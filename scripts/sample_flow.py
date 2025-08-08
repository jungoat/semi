# scripts/sample_flow.py
# 학습된 FlawMatch 체크포인트에서 클래스 조건별 샘플 생성
# - guided(uncond+CFG)와 uncond(None) 둘 다 저장
# - guidance_scale, ode_steps, cond_ids, n_per_cond 조절 가능
# - 디버그를 위해 ODE 궤적(step-by-step) 저장 옵션 제공

from __future__ import annotations

import os
import math
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


# =============================
# Embeddings & UNet (train_flow.py와 동일 설계)
# =============================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        inv = torch.exp(-math.log(10000) * torch.arange(self.half).float() / self.half)
        self.register_buffer("inv_freq", inv, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        ang = t * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, c_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.t_proj = nn.Linear(t_dim, out_ch)
        self.c_proj = nn.Linear(c_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None] + self.c_proj(c_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, C, H * W)
        k = self.k(x).view(B, C, H * W)
        v = self.v(x).view(B, C, H * W)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(C), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(B, C, H, W)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, c_dim: int):
        super().__init__()
        self.block1 = ResidualBlock(in_ch, out_ch, t_dim, c_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, t_dim, c_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb, c_emb):
        x = self.block1(x, t_emb, c_emb)
        x = self.block2(x, t_emb, c_emb)
        skip = x
        x = self.down(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, c_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block1 = ResidualBlock(out_ch * 2, out_ch, t_dim, c_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, t_dim, c_dim)

    def forward(self, x, skip, t_emb, c_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb, c_emb)
        x = self.block2(x, t_emb, c_emb)
        return x


class VectorFieldUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, ch_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 t_dim: int = 128, c_dim: int = 128, num_classes: int = 18, use_attn: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.t_embedding = SinusoidalTimeEmbedding(t_dim)
        self.c_embedding = nn.Embedding(num_classes + 1, c_dim)

        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        prev = chs[0]
        for out in chs[1:]:
            self.downs.append(Down(prev, out, t_dim, c_dim))
            prev = out

        self.mid1 = ResidualBlock(prev, prev, t_dim, c_dim)
        self.attn = SelfAttention2d(prev) if use_attn else nn.Identity()
        self.mid2 = ResidualBlock(prev, prev, t_dim, c_dim)

        self.ups = nn.ModuleList()
        for out in reversed(chs[:-1]):
            self.ups.append(Up(prev, out, t_dim, c_dim))
            prev = out

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        B = x.size(0)
        t_emb = self.t_embedding(t)
        if c is None:
            c_idx = torch.full((B,), self.num_classes, dtype=torch.long, device=x.device)
        else:
            c_idx = c
        c_emb = self.c_embedding(c_idx)

        x = self.in_conv(x)
        skips = []
        for d in self.downs:
            x, s = d(x, t_emb, c_emb)
            skips.append(s)
        x = self.mid2(self.attn(self.mid1(x, t_emb, c_emb)), t_emb, c_emb)
        for u in self.ups:
            x = u(x, skips.pop(), t_emb, c_emb)
        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


class FlawMatchSampler(nn.Module):
    def __init__(self, image_size: int, num_classes: int, guidance_scale: float, ode_steps: int,
                 base_ch: int = 32, t_dim: int = 128, c_dim: int = 128):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale
        self.ode_steps = ode_steps

        self.net = VectorFieldUNet(
            in_ch=1, base_ch=base_ch, ch_mults=(1,2,4,8),
            t_dim=t_dim, c_dim=c_dim, num_classes=num_classes, use_attn=True
        )

    def load_ckpt(self, ckpt_path: str, map_location: str = "cuda"):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state = ckpt.get("model", ckpt)
        self.net.load_state_dict(state, strict=True)
        return ckpt.get("cfg", None)

    @torch.no_grad()
    def sample_guided(self, n: int, cond_id: int, device: torch.device) -> torch.Tensor:
        self.eval()
        x = torch.randn(n, 1, self.image_size, self.image_size, device=device)
        c = torch.full((n,), cond_id, dtype=torch.long, device=device)

        def f(x, t_scalar):
            t = torch.full((x.size(0),), t_scalar, device=device)
            v_un = self.net(x, t, None)
            v_co = self.net(x, t, c)
            return v_un + self.guidance_scale * (v_co - v_un)

        steps = self.ode_steps
        dt = 1.0 / steps
        t = 0.0
        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        return x

    @torch.no_grad()
    def sample_uncond(self, n: int, device: torch.device) -> torch.Tensor:
        self.eval()
        x = torch.randn(n, 1, self.image_size, self.image_size, device=device)

        def f(x, t_scalar):
            t = torch.full((x.size(0),), t_scalar, device=device)
            return self.net(x, t, None)

        steps = self.ode_steps
        dt = 1.0 / steps
        t = 0.0
        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        return x

    @torch.no_grad()
    def sample_traj(self, cond_id: int, device: torch.device) -> List[torch.Tensor]:
        self.eval()
        x = torch.randn(1, 1, self.image_size, self.image_size, device=device)
        c = torch.full((1,), cond_id, dtype=torch.long, device=device)
        traj = [x.clone()]

        def f(x, t_scalar):
            t = torch.full((x.size(0),), t_scalar, device=device)
            v_un = self.net(x, t, None)
            v_co = self.net(x, t, c)
            return v_un + self.guidance_scale * (v_co - v_un)

        steps = self.ode_steps
        dt = 1.0 / steps
        t = 0.0
        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(x.clone())
            t += dt
        return traj


# =============================
# Utils
# =============================

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_uint01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


def parse_cond_ids(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip() != '']


# =============================
# CLI
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description="Sample from trained FlawMatch")
    ap.add_argument("--ckpt_path", type=str, default="runs/flow_match/ckpt/fm_best.pth")
    ap.add_argument("--out_dir", type=str, default="runs/flow_match/samples_eval")

    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=18)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--ode_steps", type=int, default=50)

    ap.add_argument("--cond_ids", type=str, default="0,1,2")
    ap.add_argument("--n_per_cond", type=int, default=16)

    ap.add_argument("--save_traj", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    sampler = FlawMatchSampler(
        image_size=args.image_size,
        num_classes=args.num_classes,
        guidance_scale=args.guidance_scale,
        ode_steps=args.ode_steps,
    ).to(device)
    cfg = sampler.load_ckpt(args.ckpt_path, map_location="cuda")

    cond_ids = parse_cond_ids(args.cond_ids)

    for cid in cond_ids:
        # guided
        imgs_g = sampler.sample_guided(n=args.n_per_cond, cond_id=cid, device=device)
        imgs_g = to_uint01(imgs_g)
        save_image(imgs_g, os.path.join(args.out_dir, f"guided_cond{cid}.png"), nrow=int(max(1, args.n_per_cond**0.5)))
        # 개별 저장
        for i in range(imgs_g.size(0)):
            save_image(imgs_g[i:i+1], os.path.join(args.out_dir, f"guided_cond{cid}_{i:03d}.png"))

        # uncond
        imgs_u = sampler.sample_uncond(n=args.n_per_cond, device=device)
        imgs_u = to_uint01(imgs_u)
        save_image(imgs_u, os.path.join(args.out_dir, f"uncond_like_cond{cid}.png"), nrow=int(max(1, args.n_per_cond**0.5)))
        for i in range(imgs_u.size(0)):
            save_image(imgs_u[i:i+1], os.path.join(args.out_dir, f"uncond_like_cond{cid}_{i:03d}.png"))

        # trajectory
        if args.save_traj:
            traj = sampler.sample_traj(cond_id=cid, device=device)
            traj_img = to_uint01(torch.cat(traj, dim=0))
            save_image(traj_img, os.path.join(args.out_dir, f"traj_cond{cid}.png"), nrow=len(traj))

    print("done")


if __name__ == "__main__":
    main()
