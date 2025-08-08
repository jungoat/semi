# scripts/step2_extract_patches.py
# 논문 5.2 절차 반영: 양성 샘플에서 결함 GT 또는 임계값 이진화 기반으로 인스턴스 추출
# - Opening → Dilation → Connected Components
# - 위치(x,y), 크기(w,h), 방향비(orientation=w/h) 저장
# - 패치 저장(64x64, 중심 기준), 메타 CSV 저장

import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# 고정 경로 및 기본 설정
index_csv = "data/meta/index.csv"
out_dir = "patches"
patch_size = 64
min_area = 50              # 너무 작은 잡음 제거 임계값
kernel_open = 3            # opening 커널 크기
open_iter = 1
kernel_dilate = 3
dilate_iter = 3

# 마스크 사용 정책
use_gt_only = True         # True: GT 마스크만 사용, False: GT 없으면 임계값 이진화로 대체
fallback_to_threshold = True  # GT 없을 때 thresholding fallback 허용

defect_is_darker = True    # 결함이 배경 대비 어둡게 보일 때 True → THRESH_BINARY_INV + OTSU

meta_csv_path = os.path.join(out_dir, "defect_instances.csv")


@dataclass
class InstanceMeta:
    split: str
    basename: str
    comp_id: int
    img_path: str
    mask_path: str
    mask_source: str  # 'gt' or 'threshold'
    patch_path: str
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: int
    cy: int
    nx: float
    ny: float
    nw: float
    nh: float
    orientation: float


class DefectExtractor:
    def __init__(self):
        os.makedirs(out_dir, exist_ok=True)
        self.kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
        self.kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dilate, kernel_dilate))

    @staticmethod
    def _ensure_box(cx: int, cy: int, H: int, W: int, size: int):
        half = size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = x1 + size
        y2 = y1 + size
        if x2 > W:
            x1 = W - size
            x2 = W
        if y2 > H:
            y1 = H - size
            y2 = H
        x1 = max(0, x1)
        y1 = max(0, y1)
        return x1, y1, x2, y2

    def _threshold_from_image(self, img_gray: np.ndarray) -> np.ndarray:
        if defect_is_darker:
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        # GT 마스크가 이미 이진일 가능성 고려: 0/비0 기준으로 이진화
        binary = (mask > 0).astype(np.uint8) * 255
        return binary

    def _postprocess_binary(self, binary: np.ndarray) -> np.ndarray:
        # Opening → Dilation
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_o, iterations=open_iter)
        dilated = cv2.dilate(opened, self.kernel_d, iterations=dilate_iter)
        return dilated

    def _components(self, binary: np.ndarray):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        return num_labels, labels, stats, centroids

    def process_row(self, row: pd.Series, metas: list):
        # 양성만 처리
        if int(row["label"]) != 1:
            return

        split = row["split"]
        basename = os.path.splitext(row["basename"])[0]
        img_path = row["img_path"]
        mask_path = row["mask_path"]

        if not os.path.isfile(img_path):
            return

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        H, W = img.shape[:2]

        # 마스크 선택: GT 우선, 없으면 임계값 이진화 fallback
        mask_source = None
        binary = None

        if use_gt_only:
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    return
                binary = self._binarize_mask(mask)
                mask_source = "gt"
            else:
                return
        else:
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    binary = self._binarize_mask(mask)
                    mask_source = "gt"
            if binary is None and fallback_to_threshold:
                binary = self._threshold_from_image(img)
                mask_source = "threshold"
            if binary is None:
                return

        proc = self._postprocess_binary(binary)
        num_labels, labels, stats, centroids = self._components(proc)

        save_base = os.path.join(out_dir, split, "1")  # 양성만 저장
        os.makedirs(save_base, exist_ok=True)

        for comp_id in range(1, num_labels):  # 0은 배경
            x, y, w, h, area = stats[comp_id]
            if area < min_area:
                continue

            cx, cy = int(centroids[comp_id][0]), int(centroids[comp_id][1])
            x1, y1, x2, y2 = self._ensure_box(cx, cy, H, W, patch_size)
            patch = img[y1:y2, x1:x2]

            save_name = f"{basename}_{comp_id}.png"
            patch_path = os.path.join(save_base, save_name)
            cv2.imwrite(patch_path, patch)

            orient = w / max(h, 1e-6)
            meta = InstanceMeta(
                split=split,
                basename=basename,
                comp_id=comp_id,
                img_path=img_path,
                mask_path=mask_path,
                mask_source=mask_source,
                patch_path=patch_path,
                x=int(x), y=int(y), w=int(w), h=int(h), area=int(area),
                cx=int(cx), cy=int(cy),
                nx=float(cx / W), ny=float(cy / H),
                nw=float(w / W), nh=float(h / H),
                orientation=float(orient),
            )
            metas.append(asdict(meta))


def extract_patches():
    df = pd.read_csv(index_csv)
    metas = []
    ex = DefectExtractor()

    # 양성만 선별하되, 파일 경로가 비어있지 않은 행만 사용
    df = df[(df["label"] == 1) & df["img_path"].notna() & (df["img_path"] != "")]

    for _, row in df.iterrows():
        ex.process_row(row, metas)

    if metas:
        meta_df = pd.DataFrame(metas)
        os.makedirs(out_dir, exist_ok=True)
        meta_df.to_csv(meta_csv_path, index=False)
        print(f"saved patches and meta: {meta_csv_path} ({len(meta_df)} rows)")
    else:
        print("no defect instances extracted")


if __name__ == "__main__":
    extract_patches()
