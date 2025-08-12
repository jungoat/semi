import matplotlib.pyplot as plt
from dataset import KolektorSDD2Dataset
import pandas as pd

if __name__ == "__main__":
    csv_path = "data/meta/defect_attributes.csv"

    # CSV 로드 후 결측 제거 + 문자열 변환
    df = pd.read_csv(csv_path).dropna(subset=["filename"])
    df["filename"] = df["filename"].astype(str)  # 여기서 강제 문자열 변환
    df.to_csv(csv_path, index=False)

    dataset = KolektorSDD2Dataset(
        csv_path=csv_path,
        img_dir="runs/kolektor_patches",
        timesteps=1000,
        img_size=64
    )

    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        xt, t, class_id, x0 = dataset[i]
        img_np = x0.squeeze(0).numpy() if x0.shape[0] == 1 else x0.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np, cmap="gray" if x0.shape[0] == 1 else None)
        axes[i].set_title(f"Class {int(class_id)}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
