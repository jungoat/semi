# sample_flow.py
import os
import torch
import argparse
from tqdm import tqdm
from models.flow_generator import FlowGenerator
from torchvision.utils import save_image

@torch.no_grad()
def ode_sampler(model, class_id, guidance_scale=3, image_size=64, steps=50, device="cuda"):
    model.eval()
    # 초기 상태: N(0, I)
    x = torch.randn(1, 1, image_size, image_size, device=device)
    t_space = torch.linspace(0, 1, steps, device=device)

    for i in range(steps - 1):
        t = t_space[i].unsqueeze(0)
        dt = t_space[i+1] - t_space[i]

        # uncond
        v_uncond = model(x, t, None)
        # cond
        c_tensor = torch.tensor([class_id], device=device)
        v_cond = model(x, t, c_tensor)

        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + v * dt  # Euler update

    return x

@torch.no_grad()
def ode_sampler_uncond(model, image_size=64, steps=50, device="cuda"):
    model.eval()
    x = torch.randn(1, 1, image_size, image_size, device=device)
    t_space = torch.linspace(0, 1, steps, device=device)

    for i in range(steps - 1):
        t = t_space[i].unsqueeze(0)
        dt = t_space[i+1] - t_space[i]
        v_uncond = model(x, t, None)
        x = x + v_uncond * dt

    return x

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowGenerator(
        in_ch=1,
        base_ch=32,
        emb_dim=128,
        num_classes=args.num_classes
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    for class_id in range(args.num_classes):
        cond_dir = os.path.join(args.save_dir, f"class_{class_id}", "cond")
        uncond_dir = os.path.join(args.save_dir, f"class_{class_id}", "uncond")
        os.makedirs(cond_dir, exist_ok=True)
        os.makedirs(uncond_dir, exist_ok=True)

        for idx in tqdm(range(args.samples_per_class), desc=f"Class {class_id}"):
            # cond
            img_cond = ode_sampler(
                model,
                class_id,
                guidance_scale=args.guidance_scale,
                image_size=args.image_size,
                steps=args.steps,
                device=device
            )
            save_image(img_cond, os.path.join(cond_dir, f"{idx}.png"))

            # uncond
            img_uncond = ode_sampler_uncond(
                model,
                image_size=args.image_size,
                steps=args.steps,
                device=device
            )
            save_image(img_uncond, os.path.join(uncond_dir, f"{idx}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--save_dir", type=str, default="samples")
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--samples_per_class", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    main(args)
