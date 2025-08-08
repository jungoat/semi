# scripts/step3_make_condition_labels.py
# 조건 클래스 라벨 생성
# - 입력: patches/defect_instances.csv (step2에서 생성)
# - 처리:
#   1) 위치(nx, ny), 방향비(orientation=w/h)로 클래스 경계 결정
#      - 방법1: KMeans (기본)
#      - 방법2: auto-k(선택) — 히스토그램 스무딩 + peak detection로 k 추정
#   2) xbin(0..kx-1), ybin(0..ky-1), obin(0..ko-1) 계산
#   3) cond_id = xbin + kx*ybin + (kx*ky)*obin
# - 출력: patches/defect_instances_labeled.csv

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# 선택 의존성: scikit-learn
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# 선택 의존성: SciPy peak detection (없으면 naive 대체)
try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

# 선택 의존성: OpenCV (히스토그램 스무딩 대체 수단)
try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class LabelConfig:
    instances_csv: str = "patches/defect_instances.csv"
    out_csv: str = "patches/defect_instances_labeled.csv"
    method: str = "kmeans"          # 'kmeans' | 'quantile'
    kx: str = "3"                    # 'auto' | int string
    ky: str = "3"                    # 'auto' | int string
    ko: str = "2"                    # 'auto' | int string (보통 2)
    fit_split: str = "train"         # 'train' | 'all'
    seed: int = 0
    hist_bins: int = 50
    peak_prominence: float = 0.02     # auto-k 시 peak prominence 비율
    gaussian_sigma: float = 1.0       # auto-k 시 히스토그램 스무딩 정도


class AxisBinner:
    def __init__(self, k: int, method: str = "kmeans", seed: int = 0):
        assert method in {"kmeans", "quantile"}
        self.k = int(k)
        self.method = method
        self.seed = seed
        self.centers_: Optional[np.ndarray] = None  # shape (k,)

    def fit(self, x: np.ndarray):
        x = np.asarray(x).reshape(-1, 1)
        if self.method == "kmeans" and KMeans is not None:
            km = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
            labels = km.fit_predict(x)
            centers = km.cluster_centers_.reshape(-1)
            # 클러스터 중심을 오름차순 정렬하고, label 매핑을 재정의
            order = np.argsort(centers)
            inv = np.empty_like(order)
            inv[order] = np.arange(self.k)
            self.centers_ = centers[order]
            self._label_map = inv  # 원 라벨 → 순위 라벨
        else:
            # quantile fallback
            self.edges_ = np.quantile(x, q=np.linspace(0, 1, self.k + 1).tolist()).reshape(-1)
            self.centers_ = 0.5 * (self.edges_[:-1] + self.edges_[1:])
            self._label_map = None
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        if self.method == "kmeans" and KMeans is not None and self.centers_ is not None:
            # 가장 가까운 center의 순위 인덱스
            d = np.abs(x[:, None] - self.centers_[None, :])
            return np.argmin(d, axis=1).astype(np.int64)
        else:
            # quantile bins
            bins = np.digitize(x, self.edges_[1:-1], right=True)
            return bins.astype(np.int64)


def _smooth_histogram(hist: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return hist
    if cv2 is not None:
        # OpenCV 가우시안 블러: 1D에 적용 위해 (N,1)로 다루고 다시 펴기
        ksize = int(6 * sigma + 1) | 1  # 홀수로 강제
        sm = cv2.GaussianBlur(hist.astype(np.float32).reshape(-1, 1), (ksize, 1), sigma).reshape(-1)
        return sm
    # naive conv 가우시안 커널
    radius = int(3 * sigma)
    xs = np.arange(-radius, radius + 1)
    ker = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    ker /= ker.sum()
    return np.convolve(hist, ker, mode="same")


def estimate_k_auto(x: np.ndarray, bins: int, prominence_ratio: float, sigma: float, k_default: int) -> int:
    x = np.asarray(x).reshape(-1)
    if x.size < 10:
        return k_default
    hist, edges = np.histogram(x, bins=bins, range=(x.min(), x.max()))
    hist = hist.astype(np.float32)
    sm = _smooth_histogram(hist, sigma)

    if find_peaks is not None:
        prom = max(sm) * prominence_ratio
        peaks, _ = find_peaks(sm, prominence=prom)
        k = int(np.clip(len(peaks), 2, 6))
    else:
        # naive peak count: 국소 최댓값 개수
        peaks = []
        for i in range(1, len(sm) - 1):
            if sm[i] > sm[i - 1] and sm[i] > sm[i + 1]:
                peaks.append(i)
        k = int(np.clip(len(peaks), 2, 6))

    return k if k >= 2 else k_default


def parse_k(val: str, x: np.ndarray, cfg: LabelConfig) -> int:
    if val == "auto":
        return estimate_k_auto(x, bins=cfg.hist_bins, prominence_ratio=cfg.peak_prominence, sigma=cfg.gaussian_sigma, k_default=3)
    return int(val)


def main(args: Optional[argparse.Namespace] = None):
    p = argparse.ArgumentParser(description="Make conditioning labels from defect instance attributes")
    p.add_argument("--instances_csv", type=str, default=LabelConfig.instances_csv)
    p.add_argument("--out_csv", type=str, default=LabelConfig.out_csv)
    p.add_argument("--method", type=str, default=LabelConfig.method, choices=["kmeans", "quantile"])
    p.add_argument("--kx", type=str, default=LabelConfig.kx)
    p.add_argument("--ky", type=str, default=LabelConfig.ky)
    p.add_argument("--ko", type=str, default=LabelConfig.ko)
    p.add_argument("--fit_split", type=str, default=LabelConfig.fit_split, choices=["train", "all"])
    p.add_argument("--seed", type=int, default=LabelConfig.seed)
    p.add_argument("--hist_bins", type=int, default=LabelConfig.hist_bins)
    p.add_argument("--peak_prominence", type=float, default=LabelConfig.peak_prominence)
    p.add_argument("--gaussian_sigma", type=float, default=LabelConfig.gaussian_sigma)
    ns = p.parse_args() if args is None else args

    df = pd.read_csv(ns.instances_csv)
    if not set(["nx", "ny", "orientation"]).issubset(df.columns):
        raise ValueError("instances_csv must contain columns: nx, ny, orientation")

    if ns.fit_split == "train" and "split" in df.columns:
        df_fit = df[df["split"] == "train"].copy()
        if len(df_fit) < 5:
            df_fit = df.copy()
    else:
        df_fit = df.copy()

    # k 결정
    kx = parse_k(ns.kx, df_fit["nx"].values, LabelConfig(hist_bins=ns.hist_bins, peak_prominence=ns.peak_prominence, gaussian_sigma=ns.gaussian_sigma))
    ky = parse_k(ns.ky, df_fit["ny"].values, LabelConfig(hist_bins=ns.hist_bins, peak_prominence=ns.peak_prominence, gaussian_sigma=ns.gaussian_sigma))
    ko = parse_k(ns.ko, df_fit["orientation"].values, LabelConfig(hist_bins=ns.hist_bins, peak_prominence=ns.peak_prominence, gaussian_sigma=ns.gaussian_sigma))

    # binner 학습 (fit_split 기준)
    bx = AxisBinner(k=kx, method=ns.method, seed=ns.seed).fit(df_fit["nx"].values)
    by = AxisBinner(k=ky, method=ns.method, seed=ns.seed).fit(df_fit["ny"].values)
    bo = AxisBinner(k=ko, method=ns.method, seed=ns.seed).fit(df_fit["orientation"].values)

    # 전체에 적용
    df["xbin"] = bx.predict(df["nx"].values)
    df["ybin"] = by.predict(df["ny"].values)
    df["obin"] = bo.predict(df["orientation"].values)

    # cond_id 조합
    df["cond_id"] = df["xbin"] + (kx * df["ybin"]) + (kx * ky) * df["obin"]

    # 정리된 통계 출력(간단 요약)
    print(f"kx={kx}, ky={ky}, ko={ko} → total classes={kx*ky*ko}")
    print("xbin counts:", df["xbin"].value_counts().sort_index().to_dict())
    print("ybin counts:", df["ybin"].value_counts().sort_index().to_dict())
    print("obin counts:", df["obin"].value_counts().sort_index().to_dict())

    os.makedirs(os.path.dirname(ns.out_csv) or ".", exist_ok=True)
    df.to_csv(ns.out_csv, index=False)
    print(f"saved: {ns.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
