# train_flow.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import KolektorSDD2Dataset
from models.flow_generator import FlowGenerator


def remap_labels(csv_path, save_path=None):
    """
    CSV의 class_id를 0부터 연속 번호로 리매핑
    """
    df = pd.read_csv(csv_path).dropna(subset=["class_id"])
    unique_labels = sorted(df["class_id"].unique())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    df["class_id"] = df["class_id"].map(label_map)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df, len(unique_labels)


class FlawMatchTrainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 라벨 리매핑
        remap_csv_path = args.train_csv.replace(".csv", "_remap.csv")
        df, num_classes = remap_labels(args.train_csv, remap_csv_path)
        args.num_classes = num_classes  # 자동 업데이트

        # 모델 초기화
        self.model = FlowGenerator(
            in_ch=1,  # 단일 채널 (Grayscale)
            base_ch=32,
            emb_dim=128,
            num_classes=args.num_classes
        ).to(self.device)

        # 하이퍼파라미터
        self.epochs = args.epochs
        self.lr = args.lr
        self.class_drop_rate = args.class_drop_rate

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Dataset & Loader
        dataset = KolektorSDD2Dataset(
            csv_path=remap_csv_path,
            img_size=64,
            augment=True
        )
        self.loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Loss: L1 Loss (논문 공식)
        self.criterion = nn.L1Loss()

        # 저장 경로
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        self.save_path = args.save_path

    def train(self):
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.loader, desc=f"Epoch [{epoch}/{self.epochs}]")
            total_loss = 0.0

            for x1, class_id in pbar:
                x1 = x1.to(self.device)  # (B, 1, H, W)
                class_id = class_id.to(self.device)

                B = x1.size(0)

                # 1) Sample noise x0 ~ N(0, I)
                x0 = torch.randn_like(x1)

                # 2) Class drop (CFG)
                if torch.rand(()) < self.class_drop_rate:
                    c_input = None
                else:
                    c_input = class_id

                # 3) Sample time t ~ U(0, 1)
                t = torch.rand(B, dtype=torch.float32, device=self.device)

                # 4) Interpolated xt
                xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1

                # 5) Target velocity ut
                ut = x1 - x0

                # 6) Predict velocity
                uhat_t = self.model(xt, t, c_input)

                # 7) Loss (L1)
                loss = self.criterion(uhat_t, ut)

                # 8) Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(self.loader)
            print(f"Epoch [{epoch}/{self.epochs}] Avg Loss: {avg_loss:.6f}")

            torch.save(self.model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, default="data/meta/defect_attributes_all_classes.csv")
    parser.add_argument("--save_path", type=str, default="checkpoints/fm_best.pth")

    parser.add_argument("--num_classes", type=int, default=None)  # 자동 설정됨
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--class_drop_rate", type=float, default=0.1)

    args = parser.parse_args()

    trainer = FlawMatchTrainer(args)
    trainer.train()
