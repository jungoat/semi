# 필요한 라이브러리 설치
# !pip install torch torchvision pandas Pillow

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DefectPatchDataset(Dataset):
    def __init__(self, csv_file, img_dir, image_size=64):

        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # 이미지에 적용할 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # 이미지 크기 64x64로 통일
            transforms.ToTensor(),                       # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # CSV에서 이미지 파일명과 클래스 라벨 가져오기
        img_name = os.path.join(self.img_dir, self.labels_df.loc[idx, 'patch_file'])
        # PIL을 사용해 이미지 로드 (RGB로 변환하여 채널 수 통일)
        image = Image.open(img_name).convert("L")
        
        # 클래스 라벨 가져오기
        label = self.labels_df.loc[idx, 'final_class']
        label = torch.tensor(label, dtype=torch.long)

        # 이미지에 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # (이미지 텐서, 라벨) 쌍 반환
        return image, label


if __name__ == '__main__':
    # 이 파일을 직접 실행하면 데이터셋이 잘 동작하는지 테스트할 수 있습니다.
    csv_path = 'data/processed/labeled_defects.csv'
    patches_path = 'data/processed/patches/'
    
    # 데이터셋 객체 생성
    defect_dataset = DefectPatchDataset(csv_file=csv_path, img_dir=patches_path)
    
    # 첫 번째 데이터 확인
    image, label = defect_dataset[0]
    
    print(" Dataset test successful!")
    print(f"Total number of patches: {len(defect_dataset)}")
    print(f"Sample 1 - Image tensor shape: {image.shape}")
    print(f"Sample 1 - Label: {label.item()}")
    print(f"Image tensor min value: {image.min()}") # -1에 가까운지 확인
    print(f"Image tensor max value: {image.max()}") # 1에 가까운지 확인