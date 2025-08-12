# sample_flow.py
import torch
from models.flow_generator import FlowGenerator
from PIL import Image
import os
import numpy as np
import pandas as pd

class FlawMatchSamplerODE:
    def __init__(self, model, device, guidance_scale=3):
        self.model = model.eval().to(device)
        self.device = device
        self.guidance_scale = guidance_scale

    def guided_vector_field(self, x, t, class_id):
        """Classifier-Free Guidance 적용 벡터필드"""
        v_uncond = self.model(x, t, None)
        v_cond = self.model(x, t, class_id)
        return v_uncond + self.guidance_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def sample(self, class_id, image_size=64, steps=200):
        """
        Algorithm 2 기반 Sampling (0 → 1 순방향 적분)
        """
        # 초기 노이즈
        x_t = torch.randn(1, 1, image_size, image_size, device=self.device)

        # 시간 스텝
        t_steps = torch.linspace(0, 1, steps, device=self.device)

        for i in range(steps - 1):
            t = t_steps[i].unsqueeze(0)  # shape: (1,)
            dt = t_steps[i+1] - t_steps[i]

            # Guidance 적용된 벡터필드 계산
            v_guided = self.guided_vector_field(x_t, t, class_id)

            # Euler update
            x_t = x_t + v_guided * dt

        return x_t

    def tensor_to_pil(self, tensor):
        """Tensor -> PIL Image 변환"""
        img = tensor.squeeze().detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)


if __name__ == "__main__":
    # ===== CSV에서 클래스 개수 자동 감지 =====
    csv_path = "data/meta/defect_attributes_all_classes.csv"
    df = pd.read_csv(csv_path)
    unique_classes = sorted(df['class_id'].unique())
    num_classes = len(unique_classes)

    checkpoint_path = "checkpoints/fm_best.pth"
    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 모델 로드 =====
    model = FlowGenerator(in_ch=1, base_ch=32, emb_dim=128, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    sampler = FlawMatchSamplerODE(model, device=device, guidance_scale=3)

    # ===== 클래스별 20장 생성 =====
    for cls_id in unique_classes:
        class_dir = os.path.join(output_dir, f"class_{cls_id}")
        os.makedirs(class_dir, exist_ok=True)

        print(f"[Class {cls_id}] Generating samples...")

        for sample_idx in range(20):
            class_tensor = torch.tensor([cls_id], device=device)
            sample_img = sampler.sample(class_id=class_tensor, image_size=64, steps=200)
            sampler.tensor_to_pil(sample_img).save(os.path.join(class_dir, f"{sample_idx}.png"))

        print(f"[Saved] {class_dir} (20 images)")
