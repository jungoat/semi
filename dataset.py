import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class KolektorSDD2Dataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=64, class_column="class_id"):
        self.df = pd.read_csv(csv_path).dropna(subset=['patch_name'])
        self.img_dir = img_dir
        self.class_column = class_column
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        class_id = int(row[self.class_column])
        patch_name = row["patch_name"]

        img_path = os.path.join(self.img_dir, f"class_{class_id}", patch_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        x1 = Image.open(img_path).convert("RGB")
        x1 = self.transform(x1)

        return x1, torch.tensor(class_id, dtype=torch.long)
