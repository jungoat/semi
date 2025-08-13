import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from dataset import DefectPatchDataset

# 저장할 폴더 생성
output_dir = "debug_outputs"
os.makedirs(output_dir, exist_ok=True)

# 데이터셋 및 데이터로더 생성
csv_path = 'data/processed/labeled_defects.csv'
patches_path = 'data/processed/patches/'
dataset = DefectPatchDataset(csv_file=csv_path, img_dir=patches_path)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# 데이터로더에서 한 배치(5개)만 가져오기
try:
    images, labels = next(iter(dataloader))
    print("Labels in this batch:", labels.tolist())
    
    # 이미지를 저장 가능한 [0, 1] 범위로 변환
    images = (images + 1) / 2
    
    # 5개 이미지를 하나의 그리드로 저장
    save_image(images, os.path.join(output_dir, "data_check.png"), nrow=5)
    
    print(f"✅ Data check successful. Please check the 'data_check.png' file in the '{output_dir}' folder.")

except Exception as e:
    print(f"❌ Error during data check: {e}")