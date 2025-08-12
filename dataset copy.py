# dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class KolektorSDD2Dataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=64):
        self.df = pd.read_csv(csv_path).dropna(subset=['filename'])
        self.img_dir = img_dir
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=1),  # 반드시 흑백 변환
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # filename이 float로 저장되어 있으면 int로 변환 후 문자열로
        filename = str(int(row["filename"]))

        # class_id는 LongTensor로 변환
        class_id = torch.tensor(int(row["class_id"]), dtype=torch.long)

        # 패치 파일명 규칙에 맞춰 경로 생성
        img_path = os.path.join(self.img_dir, f"{filename}_patch_0.png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 흑백 변환
        x1 = Image.open(img_path).convert("RGB")
        x1 = self.transform(x1)  # (1, H, W)

        return x1, class_id
