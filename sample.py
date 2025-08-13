import torch
from torchvision.utils import save_image
import os
from tqdm import tqdm

# [핵심] diffusers 대신, 우리가 직접 구현한 모델을 불러옵니다.
from model import FlawMatchUnet

# --- 1. 설정 및 모델 로드 ---
# 학습된 모델 파일 경로
model_path = "models/fm_best.pth" # 학습이 완료된 모델 경로

# 하이퍼파라미터
num_samples_per_class = 5
image_size = 64
num_classes = 12
cfg_scale = 3.0
cin = 32                  # 학습 시 사용한 모델의 기본 채널 수
output_dir = "outputs_final" # 최종 결과 폴더
os.makedirs(output_dir, exist_ok=True)

sampling_steps = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# [핵심] UNet2DModel 대신, 우리가 직접 만든 FlawMatchUnet 모델을 생성합니다.
model = FlawMatchUnet(
    in_channels=1, 
    out_channels=1, 
    num_classes=num_classes, 
    cin=cin
).to(device)

# 저장된 가중치 불러오기
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 모델을 추론 모드로 설정
print(f"Model loaded from {model_path}")


# --- 2. 오일러 적분법(Euler Method)을 사용한 샘플링 실행 ---
print(f"Generating samples using Euler method with {sampling_steps} steps...")

for class_idx in range(num_classes):
    print(f"--- Generating samples for class {class_idx} ---")
    
    class_output_dir = os.path.join(output_dir, f"class_{class_idx}")
    os.makedirs(class_output_dir, exist_ok=True)
    
    xt = torch.randn(num_samples_per_class, 1, image_size, image_size, device=device)
    
    time_steps = torch.linspace(1, 0, sampling_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    for t in tqdm(time_steps):
        with torch.no_grad():
            t_tensor = torch.full((num_samples_per_class,), t, device=device)
            
            uncond_labels = torch.full((num_samples_per_class,), num_classes, device=device, dtype=torch.long)
            cond_labels = torch.full((num_samples_per_class,), class_idx, device=device, dtype=torch.long)
            
            # [핵심] 우리 모델의 forward 함수에 맞게 호출합니다.
            uncond_pred = model(x=xt, time=t_tensor, class_labels=uncond_labels)
            cond_pred = model(x=xt, time=t_tensor, class_labels=cond_labels)
            
            velocity = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            
            xt = xt - velocity * step_size

    generated_samples = xt
    
    generated_samples = (generated_samples + 1) / 2
    generated_samples = torch.clamp(generated_samples, 0, 1)

    for i, sample in enumerate(generated_samples):
        save_path = os.path.join(class_output_dir, f"sample_{i}.png")
        save_image(sample, save_path)

print(f"\n All class samples saved into respective folders inside '{output_dir}' directory.")
