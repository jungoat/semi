# scripts/eval_classifier.py
# 이 스크립트는 전체 이미지(binary) 분류기를 학습/평가한다.
# - 입력(real): data/meta/index.csv  (split in {train,test}, label in {0,1}, img_path)
# - 입력(synth): synthetic/meta.csv  (split, img_path, cond_id) → label=1로 사용
# - 학습 모드: real | synth | real_plus_synth
# - 출력: runs/cls_full/{mode}/best.pth, metrics.txt, confusion.png
# - Backbone: ResNet18 (grayscale→RGB 변환 후 학습)

from __future__ import annotations

import os
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

try:
    import pandas as pd
    from torchvision import models, transforms
    from torchvision.utils import save_image
except Exception:
    raise RuntimeError("pandas/torchvision 필요: pip install pandas torchvision")


# =============================
# Utils
# =============================

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CLSConfig:
    index_csv: str = "data/meta/index.csv"
    synth_csv: str = "synthetic/meta.csv"
    out_dir: str = "runs/cls_full"
    mode: str = "real_plus_synth"  # real | synth | real_plus_synth
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    balance: bool = True  # 클래스 불균형 시 가중 샘플링
    seed: int = 0


# =============================
# Dataset
# =============================
class FullImageCSV(Dataset):
    def __init__(self, items: List[Tuple[str, int]], image_size: int, train: bool):
        self.items = [(p, int(y)) for p, y in items if isinstance(p, str) and os.path.isfile(p)]
        self.train = train
        self.t = self._build_tf(image_size, train)

    @staticmethod
    def _build_tf(image_size: int, train: bool):
        if train:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        else:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(image_size+16),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert('L')
        x = self.t(img)
        return x, torch.tensor(y, dtype=torch.long)


def build_items_real(index_csv: str, split: str) -> List[Tuple[str, int]]:
    df = pd.read_csv(index_csv)
    df = df[(df['split'] == split) & df['img_path'].notna() & (df['img_path'] != '')]
    out = []
    for _, r in df.iterrows():
        p = r['img_path']
        y = int(r['label'])
        out.append((p, y))
    return out


def build_items_synth(synth_csv: str, split: str) -> List[Tuple[str, int]]:
    df = pd.read_csv(synth_csv)
    df = df[(df['split'] == split) & df['img_path'].notna() & (df['img_path'] != '')]
    out = [(r['img_path'], 1) for _, r in df.iterrows()]  # synth는 전부 결함(양성)
    return out


# =============================
# Model
# =============================
class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, 2)

    def forward(self, x):
        return self.backbone(x)


# =============================
# Trainer
# =============================
class Trainer:
    def __init__(self, cfg: CLSConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(os.path.join(cfg.out_dir, cfg.mode), exist_ok=True)
        self.save_dir = os.path.join(cfg.out_dir, cfg.mode)

    def _build_loaders(self):
        # train set
        real_train = build_items_real(self.cfg.index_csv, 'train')
        synth_train = build_items_synth(self.cfg.synth_csv, 'train') if os.path.isfile(self.cfg.synth_csv) else []

        if self.cfg.mode == 'real':
            tr_items = real_train
        elif self.cfg.mode == 'synth':
            # synth만으로는 음성(0)이 없으므로, 최소한 real의 음성은 섞어준다
            negs = [(p, y) for p, y in real_train if y == 0]
            tr_items = synth_train + negs
        else:
            # real_plus_synth: real 전체 + synth(양성)
            tr_items = real_train + synth_train

        te_items = build_items_real(self.cfg.index_csv, 'test')

        tr_ds = FullImageCSV(tr_items, self.cfg.image_size, train=True)
        te_ds = FullImageCSV(te_items, self.cfg.image_size, train=False)

        if self.cfg.balance:
            # 클래스 비율 보정용 샘플러
            labels = [y for _, y in tr_ds.items]
            class_sample_count = np.bincount(labels)
            class_sample_count[class_sample_count == 0] = 1
            weights = 1.0 / class_sample_count
            sample_weights = np.array([weights[y] for y in labels])
            sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
            tr_loader = DataLoader(tr_ds, batch_size=self.cfg.batch_size, sampler=sampler, num_workers=self.cfg.num_workers, pin_memory=True)
        else:
            tr_loader = DataLoader(tr_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)

        te_loader = DataLoader(te_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        return tr_loader, te_loader

    def fit(self):
        tr_loader, te_loader = self._build_loaders()
        model = ResNet18Binary().to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        best_acc = 0.0
        best_path = os.path.join(self.save_dir, 'best.pth')

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            losses = []
            for x, y in tr_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())

            te_acc, te_prec, te_rec, te_f1 = self.evaluate(model, te_loader)
            print(f"[Epoch {epoch:03d}] loss={np.mean(losses):.4f} test_acc={te_acc:.4f} f1={te_f1:.4f}")

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save({'model': model.state_dict()}, best_path)

        # 최종 평가 및 리포트 저장
        model.load_state_dict(torch.load(best_path, map_location=self.device)['model'])
        acc, prec, rec, f1, cm = self.evaluate(model, te_loader, return_cm=True)
        self._save_report(best_acc=acc, prec=prec, rec=rec, f1=f1, cm=cm)
        print(f"saved: {best_path}")

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, return_cm: bool = False):
        model.eval()
        y_true, y_pred = [], []
        for x, y in loader:
            x = x.to(self.device)
            logits = model(x)
            p = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(p.tolist())
            y_true.extend(y.numpy().tolist())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        acc = (y_true == y_pred).mean().item()
        # precision/recall/f1 (positive=1)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if return_cm:
            cm = np.array([[tn, fp],[fn, tp]], dtype=int)
            return acc, prec, rec, f1, cm
        return acc, prec, rec, f1

    def _save_report(self, best_acc: float, prec: float, rec: float, f1: float, cm: np.ndarray):
        os.makedirs(self.save_dir, exist_ok=True)
        txt = os.path.join(self.save_dir, 'metrics.txt')
        with open(txt, 'w', encoding='utf-8') as f:
            f.write(f"mode: {self.cfg.mode}\n")
            f.write(f"acc: {best_acc:.4f}\n")
            f.write(f"prec: {prec:.4f}\n")
            f.write(f"rec:  {rec:.4f}\n")
            f.write(f"f1:   {f1:.4f}\n")
            f.write(f"cm:   {cm.tolist()}\n")
        # confusion matrix 이미지 저장
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(3,3))
            plt.imshow(cm, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha='center', va='center')
            plt.xticks([0,1],["pred 0","pred 1"]) ; plt.yticks([0,1],["true 0","true 1"]) ; plt.tight_layout()
            fig.savefig(os.path.join(self.save_dir,'confusion.png'))
            plt.close(fig)
        except Exception:
            pass


# =============================
# CLI
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description='Train/Eval binary classifier on full images')
    ap.add_argument('--index_csv', type=str, default=CLSConfig.index_csv)
    ap.add_argument('--synth_csv', type=str, default=CLSConfig.synth_csv)
    ap.add_argument('--out_dir', type=str, default=CLSConfig.out_dir)
    ap.add_argument('--mode', type=str, default=CLSConfig.mode, choices=['real','synth','real_plus_synth'])
    ap.add_argument('--image_size', type=int, default=CLSConfig.image_size)
    ap.add_argument('--batch_size', type=int, default=CLSConfig.batch_size)
    ap.add_argument('--epochs', type=int, default=CLSConfig.epochs)
    ap.add_argument('--lr', type=float, default=CLSConfig.lr)
    ap.add_argument('--weight_decay', type=float, default=CLSConfig.weight_decay)
    ap.add_argument('--num_workers', type=int, default=CLSConfig.num_workers)
    ap.add_argument('--balance', action='store_true')
    ap.add_argument('--seed', type=int, default=CLSConfig.seed)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = CLSConfig(
        index_csv=args.index_csv,
        synth_csv=args.synth_csv,
        out_dir=args.out_dir,
        mode=args.mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        balance=args.balance,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    Trainer(cfg).fit()


if __name__ == '__main__':
    main()
