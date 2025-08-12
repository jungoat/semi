import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

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
        H, W = img_shape
        labeled = label(mask > 0)
        props = regionprops(labeled)
        instances = []
        for prop in props:
            y_min, x_min, y_max, x_max = prop.bbox
            w, h = x_max - x_min, y_max - y_min
            if w == 0 or h == 0:
                continue

            x_norm, y_norm = x_min / W, y_min / H
            orientation = 0 if w / h >= 1 else 1  # 0=Horizontal, 1=Vertical

            instances.append({
                "filename": filename,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "aspect_ratio": w / h,
                "orientation": orientation,
                "x": x_min, "y": y_min, "w": w, "h": h
            })
        return instances

    def auto_bin(self, values, target_bins):
        hist, _ = np.histogram(values, bins=50)
        gaussian_filter1d(hist, sigma=self.gaussian_sigma)
        X = np.array(values).reshape(-1, 1)
        kmeans = KMeans(n_clusters=target_bins, random_state=0).fit(X)
        cluster_ids = kmeans.predict(X)
        sort_idx = np.argsort(kmeans.cluster_centers_.ravel())
        name_map = {cid: i for i, cid in enumerate(sort_idx)}
        ordered_ids = [name_map[cid] for cid in cluster_ids]
        return ordered_ids

    def save_debug_image(self, img_path, mask_path, instances, filename):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        for inst in instances:
            x, y, w, h = inst["x"], inst["y"], inst["w"], inst["h"]
            cls_id = inst.get("12-class", None)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            if cls_id is not None:
                cv2.putText(img, str(cls_id), (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(self.debug_dir, f"{filename}_debug.png"), img)
        cv2.imwrite(os.path.join(self.debug_dir, f"{filename}_mask.png"), mask)

    def save_patch(self, img_gray, inst, cls_id, idx):
        x_center = int(inst["x"] + inst["w"] // 2)
        y_center = int(inst["y"] + inst["h"] // 2)
        half_size = 32

        x_min = max(0, x_center - half_size)
        y_min = max(0, y_center - half_size)
        x_max = min(img_gray.shape[1], x_center + half_size)
        y_max = min(img_gray.shape[0], y_center + half_size)

        patch = img_gray[y_min:y_max, x_min:x_max]

        if patch.shape != (64, 64):
            pad_y = 64 - patch.shape[0]
            pad_x = 64 - patch.shape[1]
            patch = np.pad(patch, ((0, pad_y), (0, pad_x)),
                           mode='constant', constant_values=0)

        cls_dir = os.path.join(self.patch_dir, f"class_{cls_id}")
        os.makedirs(cls_dir, exist_ok=True)
        patch_name = f"{inst['filename']}_{idx}.png"  # 패치 파일명
        save_path = os.path.join(cls_dir, patch_name)
        cv2.imwrite(save_path, patch)

        return patch_name  # 저장한 파일명 반환

    def run(self):
        all_x, all_y = [], []
        all_data = []

        for _, row in self.df.iterrows():
            filename = os.path.splitext(os.path.basename(row["filename"]))[0]
            img_path = os.path.join(self.img_dir, filename + ".png")
            mask_path = os.path.join(self.img_dir, filename + "_GT.png")
            if not os.path.exists(mask_path):
                continue

            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = self.load_mask(mask_path)
            instances = self.extract_attributes(mask, img_gray.shape, filename)

            for inst in instances:
                all_x.append(inst["x_norm"])
                all_y.append(inst["y_norm"])
                self.records.append(inst)

        # binning
        x_bins_2 = self.auto_bin(all_x, 2)
        y_bins_3 = self.auto_bin(all_y, 3)
        x_bins_3 = self.auto_bin(all_x, 3)

        for i, inst in enumerate(self.records):
            configs = {
                "2-class": inst["orientation"],
                "4-class": x_bins_2[i] * 2 + y_bins_3[i] % 2,
                "6-class": x_bins_2[i] * 3 + y_bins_3[i],
                "9-class": x_bins_3[i] * 3 + y_bins_3[i],
                "12-class": (x_bins_2[i] * 3 + y_bins_3[i]) * 2 + inst["orientation"],
                "18-class": (x_bins_3[i] * 3 + y_bins_3[i]) * 2 + inst["orientation"],
            }
            inst.update(configs)

            # 패치 저장 + 파일명 받기
            patch_name = self.save_patch(
                cv2.imread(os.path.join(self.img_dir, inst["filename"] + ".png"),
                           cv2.IMREAD_GRAYSCALE),
                inst, inst["12-class"], i
            )

            inst["patch_name"] = patch_name
            inst["class_id"] = inst["12-class"]  # 학습용 컬럼

            all_data.append(inst)

            # 디버그 이미지 저장
            self.save_debug_image(
                os.path.join(self.img_dir, inst["filename"] + ".png"),
                os.path.join(self.img_dir, inst["filename"] + "_GT.png"),
                [inst], inst["filename"]
            )

        # CSV 저장
        df_out = pd.DataFrame(all_data)
        df_out.to_csv(self.out_csv, index=False)
        print(f"[INFO] CSV 저장 완료: {self.out_csv}")
        print(f"[INFO] 디버그 이미지 저장 완료: {self.debug_dir}")
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
