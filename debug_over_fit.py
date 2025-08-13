import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers import UNet2DModel
from tqdm import tqdm
import os

from dataset import DefectPatchDataset # 02번 파일에서 만든 클래스 임포트


if __name__ == '__main__':
    # --- 1. 하이퍼파라미터 및 환경 설정 ---
    learning_rate = 5e-4 # 논문 값(5e-3)보다 조금 낮춰 안정적으로 시작
    batch_size = 64     # 논문 값(256)보다 작게 설정 (VRAM 용량에 따라 조절)
    num_epochs = 20     # 학습 에폭 수
    image_size = 64
    num_classes = 12     # 우리가 클러스터링으로 정한 클래스 개수
    cfg_drop_prob = 0.1  # Classifier-Free Guidance 드롭 확률
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. 데이터 준비 ---
    csv_path = 'data/processed/labeled_defects.csv'
    patches_path = 'data/processed/patches/'
    dataset = DefectPatchDataset(csv_file=csv_path, img_dir=patches_path, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --- 3. 모델, 옵티마이저 준비 ---
    # diffusers의 UNet2DModel 사용
    # CFG를 위해 클래스 개수 + 1 (unconditional용 null class)
# 03_B_overfit_test.py 에서 모델 부분 수정
    model = UNet2DModel(
        sample_size=image_size, in_channels=1, out_channels=1,
        layers_per_block=1, # 레이어 수도 1로 줄임
        block_out_channels=(32, 64), # 매우 단순하게
        down_block_types=("DownBlock2D", "DownBlock2D"), # 어텐션도 제외
        up_block_types=("UpBlock2D", "UpBlock2D"),
        num_class_embeds=num_classes + 1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. 학습 루프 ---
    print(" Training started!")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (x1, c) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            x1 = x1.to(device) # Real images
            c = c.to(device)   # Class labels
            
            # --- Flow Matching 로직 ---
            x0 = torch.randn_like(x1) # x0
            
            # 시간 t 샘플링 (0~1 사이)
            t = torch.rand(x1.size(0), device=device).view(-1, 1, 1, 1)
            
            # xt 와 ut 계산
            xt = (1 - t) * x0 + t * x1 # Interpolated image
            ut = x1 - x0              # Target velocity
            
            # --- CFG 로직 ---
            # 일정 확률로 c를 null class index(num_classes)로 변경
            context_mask = torch.rand(x1.size(0), device=device) > cfg_drop_prob
            c = torch.where(context_mask, c, num_classes)
            
            # --- 예측 및 손실 계산 ---
            # diffusers U-Net은 timestep을 1D 텐서로 기대함
            predicted_velocity = model(sample=xt, timestep=t.squeeze(), class_labels=c).sample
            loss = F.l1_loss(predicted_velocity, ut)
            
            total_loss += loss.item()
            
            # --- 모델 업데이트 ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # --- 에폭마다 모델 저장 ---
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"flawmatch_epoch_{epoch+1}.pth"))
            print(f"Model saved at epoch {epoch+1}")

    print("Training finished")