import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from scipy.signal import find_peaks


class KolektorSDD2ClassProcessor:
    def __init__(self, csv_path, img_dir, out_csv, debug_dir, patch_dir, gaussian_sigma=2):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.out_csv = out_csv
        self.debug_dir = debug_dir
        self.patch_dir = patch_dir
        self.gaussian_sigma = gaussian_sigma
        self.records = []
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(self.patch_dir, exist_ok=True)

    def load_mask(self, mask_path):
        return np.array(Image.open(mask_path).convert("L"))

    def extract_attributes(self, mask, img_shape, filename):
        H, W = img_shape[:2]
        labeled = label(mask > 0)
        props = regionprops(labeled)
        instances = []
        for prop in props:
            y_min, x_min, y_max, x_max = prop.bbox
            w, h = x_max - x_min, y_max - y_min
            if w == 0 or h == 0:
                continue

            area = w * h
            x_norm, y_norm = x_min / W, y_min / H
            w_norm, h_norm = w / W, h / H
            area_norm = area / (W * H)
            aspect_ratio = w / h
            orientation = 0 if aspect_ratio >= 1 else 1  # 0=Horizontal, 1=Vertical

            instances.append({
                "filename": filename,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "w_norm": w_norm,
                "h_norm": h_norm,
                "area_norm": area_norm,
                "aspect_ratio": aspect_ratio,
                "orientation": orientation,
                "x": x_min, "y": y_min, "w": w, "h": h
            })
        return instances

    def auto_bin(self, values, default_bins=2, min_distance=5):
        hist, _ = np.histogram(values, bins=50)
        hist_smooth = gaussian_filter1d(hist, sigma=self.gaussian_sigma)

        # adaptive prominence
        prominence_val = max(0.05, np.std(hist_smooth) * 0.5)
        peaks, _ = find_peaks(hist_smooth, distance=min_distance, prominence=prominence_val)

        if len(peaks) > 1:
            target_bins = len(peaks)
        else:
            target_bins = default_bins

        X = np.array(values).reshape(-1, 1)
        kmeans = KMeans(n_clusters=target_bins, random_state=0).fit(X)
        cluster_ids = kmeans.predict(X)

        sort_idx = np.argsort(kmeans.cluster_centers_.ravel())
        name_map = {cid: i for i, cid in enumerate(sort_idx)}
        ordered_ids = [name_map[cid] for cid in cluster_ids]

        return ordered_ids, target_bins

    def save_patch(self, img_rgb, inst, cls_id, idx):
        x, y, w, h = inst["x"], inst["y"], inst["w"], inst["h"]
        pad_ratio = 0.2  # bbox 주변 여유 비율
        pad_w = int(w * pad_ratio)
        pad_h = int(h * pad_ratio)

        x_min = max(0, x - pad_w)
        y_min = max(0, y - pad_h)
        x_max = min(img_rgb.shape[1], x + w + pad_w)
        y_max = min(img_rgb.shape[0], y + h + pad_h)

        patch = img_rgb[y_min:y_max, x_min:x_max]
        patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)

        cls_dir = os.path.join(self.patch_dir, f"class_{cls_id}")
        os.makedirs(cls_dir, exist_ok=True)
        patch_name = f"{inst['filename']}_{idx}.png"
        save_path = os.path.join(cls_dir, patch_name)
        cv2.imwrite(save_path, patch)

        return save_path

    def run(self):
        all_x, all_y, all_aspect = [], [], []
        all_data = []

        for _, row in self.df.iterrows():
            filename = os.path.splitext(os.path.basename(row["filename"]))[0]
            img_path = os.path.join(self.img_dir, filename + ".png")
            mask_path = os.path.join(self.img_dir, filename + "_GT.png")
            if not os.path.exists(mask_path):
                continue

            img_color = cv2.imread(img_path)
            mask = self.load_mask(mask_path)
            instances = self.extract_attributes(mask, img_color.shape, filename)

            for inst in instances:
                all_x.append(inst["x_norm"])
                all_y.append(inst["y_norm"])
                all_aspect.append(inst["aspect_ratio"])
                self.records.append(inst)

        x_bins, _ = self.auto_bin(all_x, default_bins=2)
        y_bins, _ = self.auto_bin(all_y, default_bins=3)
        aspect_bins, _ = self.auto_bin(all_aspect, default_bins=2)

        for i, inst in enumerate(self.records):
            cls_id = x_bins[i] * 3 + y_bins[i]  # 예: x=2, y=3 => 총 6 class
            inst["class_id"] = cls_id

            patch_path = self.save_patch(
                cv2.imread(os.path.join(self.img_dir, inst["filename"] + ".png")),
                inst, cls_id, i
            )
            inst["patch_path"] = patch_path
            all_data.append(inst)

        df_out = pd.DataFrame(all_data)
        df_out.to_csv(self.out_csv, index=False)
        print(f"[INFO] CSV 저장 완료: {self.out_csv}")
        print(f"[INFO] 패치 저장 완료: {self.patch_dir}")


if __name__ == "__main__":
    processor = KolektorSDD2ClassProcessor(
        csv_path="data/meta/train.csv",
        img_dir="data/train/img",
        out_csv="data/meta/defect_attributes_all_classes.csv",
        debug_dir="runs/debug_features",
        patch_dir="runs/kolektor_patches"
    )
    processor.run()
