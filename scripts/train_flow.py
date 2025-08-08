# scripts/train_flow.py
# Flow Matching 학습 스크립트 (Section 4.2, 4.2.2 반영)
# - Interpolation: x_t = (1-t)x0 + t x1, Target velocity: u_t = x1 - x0
# - Loss: MSE (기본) / L1 선택 가능
# - 입력: (x_t, t, c)  ← t: time embedding, c: class embedding (조건 라벨)
# - CFG: 학습 시 확률 p로 class drop, 샘플링 시 guidance scale 적용

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.utils import save_image

# =============================
# 공통
# =============================

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project.")
    return torch.device("cuda")


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================
# Dataset
# =============================
class PatchDataset(Dataset):
    """
    CSV 형식: patch_path, cond_id, split
    이미지는 그레이스케일 가정 → [1,H,W], [-1,1] 정규화
    """
    def __init__(self, csv_path: str, image_size: int = 64):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        for col in ["patch_path", "cond_id"]:
            if col not in self.df.columns:
                raise ValueError(f"missing column: {col}")
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row["patch_path"]
        c = int(row["cond_id"])  # 0..(K-1)

        img = Image.open(p).convert("L").resize((self.image_size, self.image_size))
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        x = x[None, ...]  # [1,H,W]
        x = x * 2.0 - 1.0
        return x, torch.tensor(c, dtype=torch.long)


# =============================
# Embeddings
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
        # t: (B,) in [0,1]
        t = t.view(-1, 1)
        ang = t * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


# =============================
# UNet Blocks (+ optional Self-Attention at bottleneck)
# =============================
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
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(C), dim=-1)  # [B, HW, HW]
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int, c_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        # concat 후 채널은 out_ch + skip_ch
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, t_dim, c_dim)
        self.block2 = ResidualBlock(out_ch, out_ch, t_dim, c_dim)

    def forward(self, x, skip, t_emb, c_emb):
        x = self.up(x)               # [B, out_ch, H*2, W*2]
        x = torch.cat([x, skip], dim=1)  # [B, out_ch + skip_ch, ...]
        x = self.block1(x, t_emb, c_emb)
        x = self.block2(x, t_emb, c_emb)
        return x


class VectorFieldUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, ch_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 t_dim: int = 128, c_dim: int = 128, num_classes: int = 18, use_attn: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.t_embedding = SinusoidalTimeEmbedding(t_dim)
        self.c_embedding = nn.Embedding(num_classes + 1, c_dim)  # 마지막 인덱스는 null (CFG)

        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        skips_ch = []           # 각 단계의 skip 채널 수 기록
        prev = chs[0]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        for out in chs[1:]:
            self.downs.append(Down(prev, out, t_dim, c_dim))
            skips_ch.append(out)   # 이 단계의 skip 채널 수는 out
            prev = out

        self.mid1 = ResidualBlock(prev, prev, t_dim, c_dim)
        self.attn = SelfAttention2d(prev) if use_attn else nn.Identity()
        self.mid2 = ResidualBlock(prev, prev, t_dim, c_dim)

        self.ups = nn.ModuleList()
        for out, skipc in zip(reversed(chs[:-1]), reversed(skips_ch)):
            self.ups.append(Up(prev, skipc, out, t_dim, c_dim))
            prev = out

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)  # velocity

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
            x = u(x, skips.pop(), t_emb, c_emb)  # 역순 pop
        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


# =============================
# Flow Matching Wrapper
# =============================
@dataclass
class FMConfig:
    image_size: int = 64
    channels: int = 1
    base_ch: int = 32
    t_dim: int = 128
    c_dim: int = 128
    num_classes: int = 18
    p_drop: float = 0.1            # CFG drop rate
    loss_type: str = "mse"         # 'mse' or 'l1'
    lr: float = 0.005
    batch_size: int = 256
    epochs: int = 20
    guidance_scale: float = 3.0
    ode_steps: int = 50


class FlawMatch(nn.Module):
    def __init__(self, cfg: FMConfig):
        super().__init__()
        self.cfg = cfg
        self.net = VectorFieldUNet(
            in_ch=cfg.channels,
            base_ch=cfg.base_ch,
            ch_mults=(1, 2, 4, 8),
            t_dim=cfg.t_dim,
            c_dim=cfg.c_dim,
            num_classes=cfg.num_classes,
            use_attn=True,
        )

    @staticmethod
    def sample_noise_like(x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def fm_loss(self, x1: torch.Tensor, c: Optional[torch.Tensor]):
        B = x1.size(0)
        t = torch.rand(B, device=x1.device)
        x0 = self.sample_noise_like(x1)
        xt = (1.0 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
        ut = x1 - x0

        if self.training and self.cfg.p_drop > 0 and c is not None:
            drop = (torch.rand(B, device=x1.device) < self.cfg.p_drop)
            c = c.clone()
            c[drop] = -1  # forward(None) 처리용

        cin = None if (c is None or (c.dim() > 0 and (c < 0).any())) else c
        uhat = self.net(xt, t, cin)

        if self.cfg.loss_type == "mse":
            loss = F.mse_loss(uhat, ut)
        else:
            loss = F.l1_loss(uhat, ut)
        return loss

    @torch.no_grad()
    def sample(self, n: int, cond: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        self.eval()
        x = torch.randn(n, self.cfg.channels, H, W, device=device)
        c = torch.full((n,), cond, dtype=torch.long, device=device)

        def f(x, t_scalar):
            t = torch.full((x.size(0),), t_scalar, device=device)
            v_un = self.net(x, t, None)
            v_co = self.net(x, t, c)
            return v_un + self.cfg.guidance_scale * (v_co - v_un)

        steps = self.cfg.ode_steps
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


# =============================
# Trainer
# =============================
class FlowTrainer:
    def __init__(self, model: FlawMatch, cfg: FMConfig, device: torch.device, out_dir: str):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "ckpt"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "samples"), exist_ok=True)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        best = float("inf")
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            losses = []
            for x1, c in train_loader:
                x1, c = x1.to(self.device), c.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                loss = self.model.fm_loss(x1, c)
                loss.backward()
                self.opt.step()
                losses.append(float(loss.detach().item()))
            tr_loss = np.mean(losses) if losses else float("nan")

            va_loss = None
            if val_loader is not None:
                self.model.eval()
                v_losses = []
                with torch.no_grad():
                    for x1, c in val_loader:
                        x1, c = x1.to(self.device), c.to(self.device)
                        v_losses.append(float(self.model.fm_loss(x1, c).item()))
                va_loss = np.mean(v_losses) if v_losses else None

            msg = f"[Epoch {epoch:03d}] train={tr_loss:.4f}"
            if va_loss is not None:
                msg += f" val={va_loss:.4f}"
            print(msg)

            # 샘플 미리보기 저장: 데이터셋에 존재하는 cond 몇 개만
            try:
                cond_ids = sorted(list({int(c.item()) for _, c in list(train_loader)[:3]}))[:3]
            except Exception:
                cond_ids = [0]
            with torch.no_grad():
                for cid in cond_ids:
                    imgs = self.model.sample(n=8, cond=cid, H=self.cfg.image_size, W=self.cfg.image_size, device=self.device)
                    grid = (imgs.clamp(-1, 1) + 1) * 0.5
                    save_image(grid, os.path.join(self.out_dir, "samples", f"ep{epoch:03d}_cond{cid}.png"), nrow=4)

            # 체크포인트
            cur = va_loss if va_loss is not None else tr_loss
            if cur < best:
                best = cur
                ckpt_path = os.path.join(self.out_dir, "ckpt", "fm_best.pth")
                torch.save({
                    "model": self.model.state_dict(),
                    "cfg": self.cfg.__dict__,
                }, ckpt_path)


# =============================
# CLI
# =============================

def build_loaders(train_csv: str, val_csv: Optional[str], image_size: int, batch_size: int):
    tr_ds = PatchDataset(train_csv, image_size=image_size)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    va_ld = None
    if val_csv and os.path.isfile(val_csv):
        va_ds = PatchDataset(val_csv, image_size=image_size)
        va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)
    return tr_ds, tr_ld, va_ld


def parse_args():
    ap = argparse.ArgumentParser(description="Train FlawMatch (Flow Matching)")
    ap.add_argument("--train_csv", type=str, default="patches/train.csv")
    ap.add_argument("--val_csv", type=str, default="patches/test.csv")
    ap.add_argument("--out_dir", type=str, default="runs/flow_match")

    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=18)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1"])
    ap.add_argument("--p_drop", type=float, default=0.1)

    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--ode_steps", type=int, default=50)
    return ap.parse_args()


def main():
    seed_everything(0)
    device = require_cuda()
    args = parse_args()

    _, tr_ld, va_ld = build_loaders(args.train_csv, args.val_csv, args.image_size, args.batch_size)

    cfg = FMConfig(
        image_size=args.image_size,
        channels=1,
        base_ch=32,
        t_dim=128,
        c_dim=128,
        num_classes=args.num_classes,
        p_drop=args.p_drop,
        loss_type=args.loss_type,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        guidance_scale=args.guidance_scale,
        ode_steps=args.ode_steps,
    )

    model = FlawMatch(cfg)
    trainer = FlowTrainer(model, cfg, device, out_dir=args.out_dir)
    trainer.fit(tr_ld, va_ld)


if __name__ == "__main__":
    main()
