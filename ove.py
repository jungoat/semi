import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from tqdm import tqdm

from dataset import DefectPatchDataset

# --- 설정 (03_train_flowmatch.py와 동일하게) ---
image_size = 64
num_classes = 12
cfg_drop_prob = 0.1
device = "cuda"
batch_size = 64 # 실제 배치 사이즈
learning_rate = 5e-4 # 안정적인 학습률로 테스트

# --- 데이터 준비 (딱 한 배치만!) ---
csv_path = 'data/processed/labeled_defects.csv'
patches_path = 'data/processed/patches/'
dataset = DefectPatchDataset(csv_file=csv_path, img_dir=patches_path, image_size=image_size)
# num_workers=0 으로 설정해야 디버깅이 편합니다.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

# 데이터로더에서 딱 한 배치만 꺼내옵니다.
try:
    one_batch = next(iter(dataloader))
    print(f"Testing with one batch of size: {one_batch[0].shape}")
except StopIteration:
    print("Error: DataLoader is empty. Check your dataset.")
    exit()

# --- 모델 및 옵티마이저 준비 ---
model = UNet2DModel(
    sample_size=image_size, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(32, 64, 128, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    num_class_embeds=num_classes + 1,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- 단일 배치 과적합 테스트 루프 ---
print("🚀 Starting single batch overfitting test...")
num_iterations = 500 # 이 배치를 500번 반복해서 보여줌
model.train()

for i in tqdm(range(num_iterations)):
    optimizer.zero_grad()
    
    x1, c = one_batch
    x1 = x1.to(device)
    c = c.to(device)
    
    # Flow Matching 로직 (동일)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.size(0), device=device).view(-1, 1, 1, 1)
    xt = (1 - t) * x0 + t * x1
    ut = x1 - x0
    
    context_mask = torch.rand(x1.size(0), device=device) > cfg_drop_prob
    c = torch.where(context_mask, c, num_classes)
    
    predicted_velocity = model(sample=xt, timestep=t.squeeze(), class_labels=c).sample
    loss = F.l1_loss(predicted_velocity, ut)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 클리핑 유지
    optimizer.step()
    
    if (i + 1) % 50 == 0:
        print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")

print("✅ Overfitting test finished!")