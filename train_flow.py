# 필요한 라이브러리
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import numpy as np

# 우리가 직접 구현한 모델과 데이터셋을 불러옵니다.
from model import FlawMatchUnet
from dataset import DefectPatchDataset

# 얼리 스토핑 클래스
class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.0001): # 더 너그럽게 설정
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# 메인 실행 블록
if __name__ == '__main__':
    # --- 1. 하이퍼파라미터 (안전 제일 설정) ---
    learning_rate = 3e-4      # 딥러닝에서 가장 표준적이고 안정적인 학습률
    batch_size = 64
    num_epochs = 500          # 충분한 학습 시간
    image_size = 64
    num_classes = 12
    cfg_drop_prob = 0.1
    cin = 32
    output_dir = "models_safe"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. 데이터 준비 ---
    csv_path = 'data/processed/labeled_defects.csv'
    patches_path = 'data/processed/patches/'
    dataset = DefectPatchDataset(csv_file=csv_path, img_dir=patches_path, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- 3. 모델, 옵티마이저, 얼리 스토퍼 ---
    model = FlawMatchUnet(
        in_channels=1, out_channels=1, num_classes=num_classes, cin=cin
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)

    # --- 4. 학습 루프 ---
    print("🚀 Training started in SAFE MODE!")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (x1, c) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            x1, c = x1.to(device), c.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), device=device)
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
            ut = x1 - x0
            
            context_mask = torch.rand(x1.size(0), device=device) > cfg_drop_prob
            c = torch.where(context_mask, c, num_classes)
            
            predicted_velocity = model(x=xt, time=t, class_labels=c)
            loss = F.l1_loss(predicted_velocity, ut)
            
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        if early_stopper.early_stop(avg_loss):
            print("Early stopping triggered!")
            break
        
        if avg_loss <= early_stopper.min_validation_loss:
            print(f"New best model found! Saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), os.path.join(output_dir, "flawmatch_best_model.pth"))

    print(" Training finished!")
