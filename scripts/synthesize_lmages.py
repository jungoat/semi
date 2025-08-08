# scripts/step6_synthesize_images.py
# 목적: 학습된 FlawMatch로 결함 패치를 샘플링하여, 음성 배경 이미지에 합성해
#       "합성 전체 이미지"(image, mask, bbox 메타)를 생성한다.
# 입력:
#   - data/meta/index.csv          : 배경으로 쓸 음성 이미지(label==0)
#   - patches/defect_instances_labeled.csv : cond_id ↔ (xbin,ybin,obin) 조회용
#   - runs/flow_match/ckpt/fm_best.pth      : 학습된 모델 가중치
# 출력:
#   - synthetic/train/img/*.png, synthetic/train/mask/*.png
#   - synthetic/meta.csv (img_path, mask_path, split, cond_id, x,y,w,h)

from __future__ import annotations

import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# UNet (train_flow.py와 동일, Up의 skip 채널 버그 패치 반영)
# -----------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        inv = torch.exp(-math.log(10000) * torch.arange(self.half).float() / self.half)
        self.register_buffer("inv_freq", inv, persistent=False)
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int, c_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block1 = ResidualBlock(out_ch + skip_ch, out_ch, t_dim, c_dim)
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
        self.c_embedding = nn.Embedding(num_classes + 1, c_dim)  # 마지막은 null

        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        skips_ch = []
        prev = chs[0]
        for out in chs[1:]:
            self.downs.append(Down(prev, out, t_dim, c_dim))
            skips_ch.append(out)
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
        self.net = VectorFieldUNet(1, base_ch, (1,2,4,8), t_dim, c_dim, num_classes, True)

    def load_ckpt(self, ckpt_path: str, map_location: str = "cuda"):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state = ckpt.get("model", ckpt)
        self.net.load_state_dict(state, strict=True)
        return ckpt.get("cfg", None)

    @torch.no_grad()
    def _ode(self, x: torch.Tensor, f):
        steps = self.ode_steps
        dt = 1.0 / steps
        t = 0.0
        for _ in range(steps):
            k1 = f(x, t)
            k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = f(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
        return x

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
        return self._ode(x, f)


# -----------------
# 합성기
# -----------------
@dataclass
class SynthConfig:
    index_csv: str = "data/meta/index.csv"
    cond_csv: str = "patches/defect_instances_labeled.csv"
    ckpt_path: str = "runs/flow_match/ckpt/fm_best.pth"
    out_dir: str = "synthetic"
    split: str = "train"           # 배경으로 사용할 split
    image_size: int = 64            # 패치 크기 (생성 모델 크기)
    guidance_scale: float = 3.0
    ode_steps: int = 50
    per_bg: int = 1                 # 배경 한 장당 생성할 결함 수
    n_limit: Optional[int] = None   # 합성 총 개수 제한
    seed: int = 0
    defect_is_darker: bool = True
    alpha: float = 1.0              # 패치 혼합 알파(1.0이면 paste)


class CondMapper:
    """cond_id ↔ (xbin,ybin,obin) 매핑 및 좌표 샘플 도우미"""
    def __init__(self, cond_csv: str):
        df = pd.read_csv(cond_csv)
        need = {"cond_id", "xbin", "ybin", "obin"}
        if not need.issubset(df.columns):
            raise ValueError(f"cond csv must contain {need}")
        df = df.drop_duplicates(subset=["cond_id"])  # 대표 매핑만
        self.map = {int(r.cond_id): (int(r.xbin), int(r.ybin), int(r.obin)) for _, r in df.iterrows()}

    @staticmethod
    def bin_center(bin_idx: int, bins: int, W: int) -> int:
        # 0..bins-1 → 1/(2*bins), 3/(2*bins), ... 위치로 매핑
        return int((2*bin_idx + 1) * (W / (2*bins)))

    def propose_xy(self, cond_id: int, H: int, W: int, patch: int, jitter: int = 8) -> Tuple[int, int]:
        xbin, ybin, _ = self.map[int(cond_id)]
        cx = self.bin_center(xbin, 3, W)
        cy = self.bin_center(ybin, 3, H)
        # 패치가 이미지 안에 들어오도록 경계 보정 + 소량 지터
        half = patch // 2
        cx = max(half, min(W - half - 1, cx + random.randint(-jitter, jitter)))
        cy = max(half, min(H - half - 1, cy + random.randint(-jitter, jitter)))
        x1, y1 = cx - half, cy - half
        return x1, y1


class ImageComposer:
    def __init__(self, defect_is_darker: bool = True, alpha: float = 1.0):
        self.defect_is_darker = defect_is_darker
        self.alpha = float(alpha)

    @staticmethod
    def to_uint8(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def make_soft_mask(self, patch_u8: np.ndarray) -> np.ndarray:
        # 그레이 패치에서 결함 영역을 소프트마스크로 추정
        if self.defect_is_darker:
            inv = 255 - patch_u8
            inv = cv2.GaussianBlur(inv, (3,3), 0)
            m = inv.astype(np.float32) / 255.0
        else:
            m = cv2.GaussianBlur(patch_u8, (3,3), 0).astype(np.float32) / 255.0
        m = np.clip(m, 0.0, 1.0)
        return m

    def paste(self, bg_u8: np.ndarray, patch_u8: np.ndarray, top: int, left: int) -> Tuple[np.ndarray, np.ndarray]:
        H, W = bg_u8.shape[:2]
        ph, pw = patch_u8.shape[:2]
        roi = bg_u8[top:top+ph, left:left+pw]
        m = self.make_soft_mask(patch_u8)
        if self.alpha < 1.0:
            m = m * self.alpha
        comp = (roi.astype(np.float32) * (1.0 - m) + patch_u8.astype(np.float32) * m)
        out = bg_u8.copy()
        out[top:top+ph, left:left+pw] = self.to_uint8(comp)
        # 바이너리 마스크도 함께 생성(평가/학습용)
        bin_m = (m > 0.2).astype(np.uint8) * 255
        mask = np.zeros_like(bg_u8)
        mask[top:top+ph, left:left+pw] = bin_m
        return out, mask


class Synthesizer:
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = FlawMatchSampler(cfg.image_size, num_classes=18, guidance_scale=cfg.guidance_scale, ode_steps=cfg.ode_steps).to(self.device)
        self.sampler.load_ckpt(cfg.ckpt_path, map_location="cuda" if self.device.type == "cuda" else "cpu")
        self.mapper = CondMapper(cfg.cond_csv)
        self.composer = ImageComposer(defect_is_darker=cfg.defect_is_darker, alpha=cfg.alpha)

    def _load_backgrounds(self) -> List[Tuple[str, int, int]]:
        df = pd.read_csv(self.cfg.index_csv)
        df = df[(df["split"] == self.cfg.split) & (df["label"] == 0)].copy()
        paths = []
        for _, r in df.iterrows():
            p = r["img_path"]
            if isinstance(p, str) and os.path.isfile(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape[0] >= self.cfg.image_size and img.shape[1] >= self.cfg.image_size:
                    paths.append((p, img.shape[0], img.shape[1]))
        return paths

    def _choose_cond(self, cond_counts: Optional[pd.Series] = None) -> int:
        if cond_counts is None or cond_counts.sum() == 0:
            # 균등
            return random.randint(0, 17)
        ids = cond_counts.index.tolist()
        probs = (cond_counts.values / cond_counts.values.sum()).tolist()
        return int(np.random.choice(ids, p=probs))

    def run(self, out_split: str = "train", match_real_dist: bool = True):
        os.makedirs(os.path.join(self.cfg.out_dir, out_split, "img"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.out_dir, out_split, "mask"), exist_ok=True)

        # real 분포 기반 cond 샘플링을 원하면 train.csv에서 분포 추출
        cond_counts = None
        if match_real_dist:
            tr_csv = os.path.join("patches", f"{out_split}.csv")
            if os.path.isfile(tr_csv):
                dfp = pd.read_csv(tr_csv)
                if "cond_id" in dfp.columns:
                    cond_counts = dfp["cond_id"].value_counts().sort_index()

        bgs = self._load_backgrounds()
        metas = []
        total = 0
        for p, H, W in bgs:
            img_bg = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img_bg is None:
                continue
            base = os.path.splitext(os.path.basename(p))[0]

            for k in range(self.cfg.per_bg):
                if self.cfg.n_limit is not None and total >= self.cfg.n_limit:
                    break
                cond = self._choose_cond(cond_counts)
                # 패치 생성
                patch = self.sampler.sample_guided(n=1, cond_id=cond, device=self.device)
                patch = (patch.clamp(-1, 1) + 1) * 0.5  # [0,1]
                patch = (patch[0,0].cpu().numpy() * 255.0).astype(np.uint8)

                # 위치 결정
                x1, y1 = self.mapper.propose_xy(cond, H, W, self.cfg.image_size)

                # 합성
                img_syn, mask_syn = self.composer.paste(img_bg, patch, y1, x1)

                save_base = f"{base}_c{cond}_{k:03d}"
                ipath = os.path.join(self.cfg.out_dir, out_split, "img", save_base + ".png")
                mpath = os.path.join(self.cfg.out_dir, out_split, "mask", save_base + ".png")
                cv2.imwrite(ipath, img_syn)
                cv2.imwrite(mpath, mask_syn)

                # bbox 계산
                ys, xs = np.where(mask_syn > 0)
                if xs.size > 0 and ys.size > 0:
                    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
                else:
                    x = y = w = h = 0

                metas.append({
                    "img_path": ipath,
                    "mask_path": mpath,
                    "split": out_split,
                    "cond_id": cond,
                    "x": x, "y": y, "w": w, "h": h,
                })
                total += 1

            if self.cfg.n_limit is not None and total >= self.cfg.n_limit:
                break

        meta_df = pd.DataFrame(metas)
        out_csv = os.path.join(self.cfg.out_dir, "meta.csv")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        meta_df.to_csv(out_csv, index=False)
        print(f"saved: {out_csv} ({len(meta_df)} rows)")


# -----------------
# CLI
# -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Synthesize full images by composing generated defect patches onto negative backgrounds")
    ap.add_argument("--index_csv", type=str, default=SynthConfig.index_csv)
    ap.add_argument("--cond_csv", type=str, default=SynthConfig.cond_csv)
    ap.add_argument("--ckpt_path", type=str, default=SynthConfig.ckpt_path)
    ap.add_argument("--out_dir", type=str, default=SynthConfig.out_dir)
    ap.add_argument("--split", type=str, default=SynthConfig.split, choices=["train", "test"])  # 배경 선택 split
    ap.add_argument("--image_size", type=int, default=SynthConfig.image_size)
    ap.add_argument("--guidance_scale", type=float, default=SynthConfig.guidance_scale)
    ap.add_argument("--ode_steps", type=int, default=SynthConfig.ode_steps)
    ap.add_argument("--per_bg", type=int, default=SynthConfig.per_bg)
    ap.add_argument("--n_limit", type=int, default=SynthConfig.n_limit if SynthConfig.n_limit is not None else 0)
    ap.add_argument("--seed", type=int, default=SynthConfig.seed)
    ap.add_argument("--alpha", type=float, default=SynthConfig.alpha)
    args = ap.parse_args()
    if args.n_limit == 0:
        args.n_limit = None
    return args


def main():
    args = parse_args()
    cfg = SynthConfig(
        index_csv=args.index_csv,
        cond_csv=args.cond_csv,
        ckpt_path=args.ckpt_path,
        out_dir=args.out_dir,
        split=args.split,
        image_size=args.image_size,
        guidance_scale=args.guidance_scale,
        ode_steps=args.ode_steps,
        per_bg=args.per_bg,
        n_limit=args.n_limit,
        seed=args.seed,
        alpha=args.alpha,
    )
    syn = Synthesizer(cfg)
    syn.run(out_split=args.split, match_real_dist=True)


if __name__ == "__main__":
    main()
