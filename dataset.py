# dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class KolektorSDD2Dataset(Dataset):
    def __init__(self, csv_path, img_size=64, class_column="class_id", augment=False):
        self.df = pd.read_csv(csv_path).dropna(subset=['patch_path'])
        self.class_column = class_column
        self.augment = augment  # True면 온라인 증강 적용

        # 기본 전처리
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])

        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['patch_path']
        class_id = int(row[self.class_column])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.augment:
            img = self.augment_transform(img)
        else:
            img = self.base_transform(img)

        return img, torch.tensor(class_id, dtype=torch.long)
