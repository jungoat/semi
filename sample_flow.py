# sample_flow.py
import torch
from models.flow_generator import FlowGenerator
from PIL import Image
import os
import numpy as np

class FlawMatchSamplerODE:
    def __init__(self, model, num_classes, device, guidance_scale=3):
        self.model = model.eval().to(device)
        self.num_classes = num_classes
        self.device = device
        self.guidance_scale = guidance_scale

    def guided_vector_field(self, t, x, class_id):
        """Classifier-Free Guidance 적용 벡터필드"""
        v_uncond = self.model(x, t, None)
        v_cond = self.model(x, t, class_id)
        return v_uncond + self.guidance_scale * (v_cond - v_uncond), v_uncond

    @torch.no_grad()
    def sample(self, class_id, image_size=64, steps=200):
        """ODE Solver (Euler)로 최종 샘플 생성"""
        # 초기 노이즈 동일하게 설정
        x_t_cond = torch.randn(1, 1, image_size, image_size, device=self.device)
        x_t_uncond = x_t_cond.clone()

        t_steps = torch.linspace(0, 1, steps, device=self.device)

        for i in range(steps - 1):
            dt = t_steps[i+1] - t_steps[i]

            # cond
            v_guided, v_uncond = self.guided_vector_field(t_steps[i].unsqueeze(0), x_t_cond, class_id)
            x_t_cond = x_t_cond + v_guided * dt

            # uncond
            x_t_uncond = x_t_uncond + v_uncond * dt

        return x_t_uncond, x_t_cond

    def tensor_to_pil(self, tensor):
        """Tensor -> PIL Image 변환"""
        img = tensor.squeeze().detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)


if __name__ == "__main__":
    # ======= 설정 =======
    checkpoint_path = "checkpoints/fm_best.pth"
    output_dir = "samples"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======= 모델 로드 =======
    model = FlowGenerator(in_ch=1, base_ch=32, emb_dim=128, num_classes=12)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    sampler = FlawMatchSamplerODE(model, num_classes=12, device=device, guidance_scale=3)

    # ======= 클래스별 20장 샘플 =======
    for cls_id in range(12):
        class_dir = os.path.join(output_dir, f"class_{cls_id}")
        cond_dir = os.path.join(class_dir, "cond")
        uncond_dir = os.path.join(class_dir, "uncond")
        os.makedirs(cond_dir, exist_ok=True)
        os.makedirs(uncond_dir, exist_ok=True)

        print(f"[Class {cls_id}] Generating samples...")

        for sample_idx in range(20):
            class_tensor = torch.tensor([cls_id], device=device)

            uncond_sample, cond_sample = sampler.sample(
                class_id=class_tensor,
                image_size=64,
                steps=200
            )

            # 저장
            sampler.tensor_to_pil(uncond_sample).save(os.path.join(uncond_dir, f"{sample_idx}.png"))
            sampler.tensor_to_pil(cond_sample).save(os.path.join(cond_dir, f"{sample_idx}.png"))

        print(f"[Saved] {class_dir} (cond & uncond, 20 images each)")
