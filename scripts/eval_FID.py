# scripts/eval_fid.py
# FID 평가 (Table 1 지표용)
# - 입력1: real_csv (patches/train.csv 또는 patches/test.csv)
# - 입력2: gen_dir  (sample_flow.py가 만든 샘플 이미지 폴더)
# - 옵션: 전체 FID + cond별 FID
# - grayscale 패치 → RGB 변환 후 299×299로 리사이즈해 InceptionV3 pool3(2048) 피처 사용
# - real 통계는 캐시(npz) 가능

from __future__ import annotations

import os
import argparse
import glob
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    from torchvision import models, transforms
except Exception as e:
    raise RuntimeError("torchvision이 필요합니다: pip install torchvision")

# sqrtm 백엔드: SciPy가 있으면 사용, 없으면 뉴튼-슐츠 근사 사용
try:
    from scipy.linalg import sqrtm as scipy_sqrtm
except Exception:
    scipy_sqrtm = None


def to_uint01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


class RealCSVSet(Dataset):
    def __init__(self, csv_path: str, transform, cond_filter: Optional[List[int]] = None):
        import pandas as pd
        df = pd.read_csv(csv_path)
        if not {"patch_path", "cond_id"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: patch_path, cond_id")
        if cond_filter is not None:
            df = df[df["cond_id"].isin(cond_filter)].copy()
        # 실제 파일만 남김
        df = df[df["patch_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))]
        self.paths = df["patch_path"].tolist()
        self.conds = df["cond_id"].astype(int).tolist()
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        c = self.conds[idx]
        img = Image.open(p).convert("L").resize((299, 299), Image.BICUBIC)
        img = Image.merge("RGB", (img, img, img))
        return self.t(img), c


class GenDirSet(Dataset):
    def __init__(self, root_dir: str, transform, cond_filter: Optional[List[int]] = None):
        # guided_cond{cid}_###.png 규칙과 cond{cid}_*.png를 탐색
        pats = [
            os.path.join(root_dir, "guided_cond*.png"),
            os.path.join(root_dir, "cond*_*.png"),
            os.path.join(root_dir, "*.png"),
        ]
        files = []
        for pat in pats:
            files.extend(glob.glob(pat))
        files = sorted(set(files))

        paths, conds = [], []
        for p in files:
            base = os.path.basename(p)
            cid = None
            # guided_cond{cid}_###.png
            if base.startswith("guided_cond"):
                try:
                    mid = base[len("guided_cond"):]
                    mid = mid.split("_")[0]  # cond{cid}
                    if mid.startswith("cond"):
                        cid = int(mid[len("cond"):])
                except Exception:
                    cid = None
            # cond{cid}_*.png 형태도 허용
            if cid is None and base.startswith("cond"):
                try:
                    cid = int(base.split("_")[0][len("cond"):])
                except Exception:
                    cid = None
            # 마지막 수단: 파일명에서 숫자만 뽑아 첫 덩어리 사용
            if cid is None:
                nums = [int(s) for s in base.replace(".", "_").split("_") if s.isdigit()]
                cid = nums[0] if nums else -1

            if cond_filter is not None and cid not in cond_filter:
                continue
            paths.append(p)
            conds.append(cid)

        self.paths = paths
        self.conds = conds
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        c = self.conds[idx]
        img = Image.open(p).convert("L").resize((299, 299), Image.BICUBIC)
        img = Image.merge("RGB", (img, img, img))
        return self.t(img), c


class InceptionPool3(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()
        m = models.inception_v3(weights=weights, aux_logits=False)
        m.fc = nn.Identity()
        self.backbone = m.to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def features(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: [B,3,299,299] in [0,1]
        with torch.no_grad():
            feats = self.backbone(imgs)  # [B, 2048]
        return feats


def compute_stats(loader: DataLoader, device: torch.device, model: InceptionPool3) -> Tuple[np.ndarray, np.ndarray]:
    feats_all = []
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            f = model.features(x)
        feats_all.append(f.cpu().numpy())
    feats = np.concatenate(feats_all, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    diff = mu1 - mu2

    if scipy_sqrtm is not None:
        covmean = scipy_sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    else:
        # 뉴튼-슐츠 근사
        covmean = _sqrtm_newton_schulz(s1 @ s2, num_iters=50)

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * tr_covmean
    return float(fid)


def _sqrtm_newton_schulz(a: np.ndarray, num_iters: int = 50, eps: float = 1e-12) -> np.ndarray:
    # a: (n,n), 양의 정부호 근처라고 가정
    a_t = torch.from_numpy(a).float()
    norm = torch.linalg.norm(a_t)
    if norm < eps:
        return np.zeros_like(a)
    a_n = a_t / norm
    y = a_n
    z = torch.eye(a_n.shape[0], dtype=a_t.dtype)
    for _ in range(num_iters):
        yz = 0.5 * (3.0 * torch.eye(a_n.shape[0]) - z @ y)
        y = y @ yz
        z = yz @ z
    s = y * math.sqrt(norm.item())
    return s.cpu().numpy()


def build_transform():
    # InceptionV3 IMAGENET1K_V1 표준화 파이프라인
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    return weights.transforms()


def cache_or_compute_real_stats(csv_path: str, device: torch.device, batch_size: int, cache_npz: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    t = build_transform()
    ds = RealCSVSet(csv_path, t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    inc = InceptionPool3(device)
    mu, sigma = compute_stats(dl, device, inc)

    if cache_npz:
        os.makedirs(os.path.dirname(cache_npz) or ".", exist_ok=True)
        np.savez(cache_npz, mu=mu, sigma=sigma)
    return mu, sigma


def load_or_build_real_stats(csv_path: str, device: torch.device, batch_size: int, cache_npz: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if cache_npz and os.path.isfile(cache_npz):
        d = np.load(cache_npz)
        return d["mu"], d["sigma"]
    return cache_or_compute_real_stats(csv_path, device, batch_size, cache_npz)


def eval_fid(real_csv: str, gen_dir: str, out_txt: str, batch_size: int = 64, per_cond: bool = True, cache_real_npz: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 전체 FID
    t = build_transform()
    inc = InceptionPool3(device)

    print("computing real stats ...")
    mu_r, si_r = load_or_build_real_stats(real_csv, device, batch_size, cache_real_npz)

    print("computing gen stats ...")
    ds_g = GenDirSet(gen_dir, t)
    if len(ds_g) == 0:
        raise RuntimeError(f"generated images not found in {gen_dir}")
    dl_g = DataLoader(ds_g, batch_size=batch_size, shuffle=False, num_workers=2)
    mu_g, si_g = compute_stats(dl_g, device, inc)

    fid_all = frechet_distance(mu_r, si_r, mu_g, si_g)
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"FID_all: {fid_all:.4f}\n")

    print(f"FID_all: {fid_all:.4f}")

    # 조건별 FID
    if per_cond:
        # cond 교집합 기준
        import pandas as pd
        real_df = pd.read_csv(real_csv)
        r_conds = sorted(set(real_df["cond_id"].astype(int).tolist()))
        g_conds = sorted(set(ds_g.conds))
        commons = [c for c in r_conds if c in g_conds]

        with open(out_txt, "a", encoding="utf-8") as f:
            for c in commons:
                ds_r_c = RealCSVSet(real_csv, t, cond_filter=[c])
                ds_g_c = GenDirSet(gen_dir, t, cond_filter=[c])
                if len(ds_r_c) < 8 or len(ds_g_c) < 8:
                    continue
                dl_r_c = DataLoader(ds_r_c, batch_size=batch_size, shuffle=False, num_workers=2)
                dl_g_c = DataLoader(ds_g_c, batch_size=batch_size, shuffle=False, num_workers=2)
                mu_r_c, si_r_c = compute_stats(dl_r_c, device, inc)
                mu_g_c, si_g_c = compute_stats(dl_g_c, device, inc)
                fid_c = frechet_distance(mu_r_c, si_r_c, mu_g_c, si_g_c)
                print(f"FID_cond[{c}]: {fid_c:.4f}")
                f.write(f"FID_cond[{c}]: {fid_c:.4f}\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate FID between real patches and generated images")
    ap.add_argument("--real_csv", type=str, default="patches/test.csv")
    ap.add_argument("--gen_dir", type=str, default="runs/flow_match/samples_eval")
    ap.add_argument("--out_txt", type=str, default="runs/flow_match/fid.txt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--per_cond", action="store_true")
    ap.add_argument("--cache_real_npz", type=str, default="runs/flow_match/real_stats_test.npz")
    return ap.parse_args()


def main():
    args = parse_args()
    eval_fid(
        real_csv=args.real_csv,
        gen_dir=args.gen_dir,
        out_txt=args.out_txt,
        batch_size=args.batch_size,
        per_cond=args.per_cond,
        cache_real_npz=args.cache_real_npz,
    )


if __name__ == "__main__":
    main()
