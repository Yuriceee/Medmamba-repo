"""
# 训练：默认跑 Swin-T、ConvNeXt-T、ResNet-50、EffNetV2-S、DeiT-Small
python compare_baselines.py \
  --mode train \
  --data_root /export/home2/junhao003/Yuqing/MedMamba/medmnist_data/bloodmnist/imagefolder_224 \
  --train_dir train --val_dir val --test_dir test \
  --epochs 50 --batch_size 64 --img_size 224 \
  --output_dir outputs/exp_baselines --amp

# 测试：读取各模型 best.pt，统一评估 acc / macro-F1 / macro-AUC
python compare_baselines.py \
  --mode test \
  --data_root /path/to/dataset \
  --test_dir test \
  --output_dir outputs/exp_baselines
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- Optional deps ----------
try:
    import timm
except Exception as e:
    raise ImportError(f"timm is required. pip install timm==0.4.12 (or newer). Error: {e}")

try:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        precision_score, recall_score, confusion_matrix
    )
except Exception:
    np = None
    accuracy_score = f1_score = roc_auc_score = None
    precision_score = recall_score = confusion_matrix = None
    warnings.warn("scikit-learn / numpy not found: metrics may be unavailable.")



# ---------- Utils ----------

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------- Model ----------

DEFAULT_ARCHS = [
    "swin_tiny_patch4_window7_224",
    "convnext_tiny",
    "resnet50",
    "tf_efficientnetv2_s",
    "deit_small_patch16_224",
]


def build_model(arch: str, num_classes: int) -> nn.Module:
    model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
    return model


# ---------- Data ----------

def make_transforms(img_size: int):
    def to_3ch(x):
        # (C,H,W) -> if gray (1,H,W) repeat to (3,H,W)
        if x.ndim == 3 and x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(to_3ch),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(to_3ch),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


@dataclass
class Datasets:
    train: datasets.ImageFolder
    val: datasets.ImageFolder
    test: Optional[datasets.ImageFolder]


def build_datasets(data_root: Path, train_dir: str, val_dir: str, test_dir: Optional[str], img_size: int) -> Tuple[Datasets, int, List[str]]:
    train_tf, eval_tf = make_transforms(img_size)
    train_ds = datasets.ImageFolder(str(data_root / train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_root / val_dir), transform=eval_tf)
    classes = train_ds.classes
    num_classes = len(classes)
    test_ds = None
    if test_dir is not None and (data_root / test_dir).exists():
        test_ds = datasets.ImageFolder(str(data_root / test_dir), transform=eval_tf)
        test_ds.class_to_idx = train_ds.class_to_idx
    return Datasets(train=train_ds, val=val_ds, test=test_ds), num_classes, classes



# ---------- Train / Eval ----------

def forward_one_epoch(model, loader, device, criterion, amp: bool = True, train: bool = False, optimizer=None):
    model.train(train)
    loss_meter = AverageMeter()
    all_probs, all_labels = [], []

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and train))

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            loss = criterion(outputs, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        if np is not None:
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    metrics = {"loss": loss_meter.avg}
    if np is not None and len(all_probs) > 0:
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        preds = probs.argmax(axis=1)
        metrics["acc"] = float(accuracy_score(labels, preds)) if accuracy_score else None
        if f1_score:
            try:
                metrics["f1_macro"] = float(f1_score(labels, preds, average="macro"))
            except Exception:
                metrics["f1_macro"] = None
        if roc_auc_score:
            try:
                metrics["auc_macro"] = float(roc_auc_score(labels, probs, multi_class="ovo", average="macro"))
            except Exception:
                metrics["auc_macro"] = None
        try:
            num_classes = probs.shape[1]
            extra = compute_overall_metrics(labels, preds, probs, num_classes)
            metrics.update(extra)
        except Exception:
            pass
    return metrics


def train_one_arch(arch: str, args, data: Datasets, num_classes: int, device: torch.device, outdir: Path) -> Dict:
    ensure_dir(outdir)
    model = build_model(arch, num_classes).to(device)

    train_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(data.val,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = -1.0
    best_ckpt = outdir / "best.pt"
    history: List[Dict] = []
    early_count = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = forward_one_epoch(model, train_loader, device, criterion, amp=args.amp, train=True, optimizer=optimizer)
        val_m   = forward_one_epoch(model, val_loader,   device, criterion, amp=False, train=False)
        scheduler.step()
        t1 = time.time()

        # Choose val score priority: AUC > F1 > ACC
        val_score = (
            (val_m.get("auc_macro") or -1.0),
            (val_m.get("f1_macro") or -1.0),
            (val_m.get("acc") or -1.0),
        )
        score_scalar = val_score[0] if val_score[0] >= 0 else (val_score[1] if val_score[1] >= 0 else val_score[2])

        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}, "time_sec": round(t1 - t0, 2)}
        history.append(row)
        print(f"[ARCH {arch}] Ep {epoch+1}/{args.epochs} | train_loss={train_m['loss']:.4f} | val_loss={val_m['loss']:.4f} | val_acc={val_m.get('acc')} | val_f1={val_m.get('f1_macro')} | val_auc={val_m.get('auc_macro')}")

        if score_scalar > best_metric:
            best_metric = score_scalar
            torch.save({
                "arch": arch,
                "model_state": model.state_dict(),
                "num_classes": num_classes,
            }, best_ckpt)
            early_count = 0
        else:
            early_count += 1

        if args.early_stop > 0 and early_count >= args.early_stop:
            print(f"[ARCH {arch}] Early stopping at epoch {epoch+1}.")
            break

    save_json({"history": history, "best_metric": best_metric}, outdir / "train_log.json")
    return {"best_metric": best_metric, "best_ckpt": str(best_ckpt)}


def test_one_arch(arch: str, args, data: Datasets, num_classes: int, device: torch.device, outdir: Path) -> Dict:
    ensure_dir(outdir)
    ckpt_path = None
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt_path = Path(args.checkpoint)
    else:
        cand = outdir / "best.pt"
        ckpt_path = cand if cand.exists() else None
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint for arch={arch}. Provide --checkpoint or train first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(arch, num_classes)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device).eval()

    assert data.test is not None, "Test dataset not found. Provide --test_dir or use --val_dir as test."
    test_loader = DataLoader(data.test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    test_m = forward_one_epoch(model, test_loader, device, criterion, amp=False, train=False)
    print(f"[ARCH {arch}] TEST | acc={test_m.get('acc')} | f1={test_m.get('f1_macro')} | auc={test_m.get('auc_macro')} | loss={test_m['loss']:.4f}")

    save_json({"checkpoint": str(ckpt_path), "test_metrics": test_m}, outdir / "test_metrics.json")
    return test_m

def compute_overall_metrics(labels: np.ndarray,
                            preds: np.ndarray,
                            probs: np.ndarray,
                            num_classes: int) -> dict:

    out = {}

    # weighted precision / recall (=sensitivity) / f1
    if precision_score is not None:
        out["precision"]   = float(precision_score(labels, preds, average="weighted", zero_division=0))
    if recall_score is not None:
        out["sensitivity"] = float(recall_score(labels, preds, average="weighted", zero_division=0))
    if f1_score is not None:
        out["f1"]          = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # specificity: weighted by class support
    if confusion_matrix is not None:
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        support = cm.sum(axis=1)  # per-class counts
        spec_per_class = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            spec_i = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            spec_per_class.append(spec_i)
        total = support.sum() if support.sum() > 0 else 1
        out["specificity"] = float(np.nansum(np.array(spec_per_class) * support) / total)

    # AUC (OvR)
    try:
        if num_classes == 2:
            out["auc_ovr"] = float(roc_auc_score(labels, probs[:, 1]))
        else:
            out["auc_ovr"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr"] = None

    return out


# ---------- Main ----------

def parse_args():
    p = argparse.ArgumentParser(description="Comparative Analysis (baselines only, no MedMamba)")
    p.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_dir", type=str, default="train")
    p.add_argument("--val_dir", type=str, default="val")
    p.add_argument("--test_dir", type=str, default="test")
    p.add_argument("--archs", type=str, default=",".join(DEFAULT_ARCHS))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Use mixed precision training")
    p.add_argument("--early_stop", type=int, default=10, help="Early stopping patience (epochs). 0 disables.")
    p.add_argument("--output_dir", type=str, default="outputs/compare_baselines")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to .pt for test-only")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    datasets_pack, num_classes, classes = build_datasets(
        data_root=data_root, train_dir=args.train_dir, val_dir=args.val_dir, test_dir=args.test_dir, img_size=args.img_size
    )
    print(f"Detected {num_classes} classes: {classes}")

    archs = [a.strip() for a in args.archs.split(',') if a.strip()]

    csv_path = out_root / ("results_" + ("test" if args.mode == "test" else "train") + ".csv")
    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            if args.mode == "test":
                writer.writerow([
                    "arch", "stage", "epoch_or_ckpt",
                    "accuracy", "precision", "sensitivity",
                    "specificity", "f1", "auc_ovr"
                ])
            else:
                writer.writerow(["arch", "stage", "epoch_or_ckpt", "acc", "f1_macro", "auc_macro", "loss", "time_or_path"])


        for arch in archs:
            arch_out = out_root / arch
            if args.mode == "train":
                result = train_one_arch(arch, args, datasets_pack, num_classes, device, arch_out)
                train_log = json.load(open(arch_out / "train_log.json", "r"))
                last = train_log["history"][-1] if train_log["history"] else {}
                writer.writerow([
                    arch, "val", last.get("epoch", "-"), last.get("val_acc"), last.get("val_f1_macro"), last.get("val_auc_macro"), last.get("val_loss"), "best_ckpt=" + result["best_ckpt"]
                ])
            else:
                metrics = test_one_arch(arch, args, datasets_pack, num_classes, device, arch_out)
                writer.writerow([
                    arch, "test", "best.pt",
                    metrics.get("acc"),
                    metrics.get("precision"),
                    metrics.get("sensitivity"),
                    metrics.get("specificity"),
                    metrics.get("f1"),
                    metrics.get("auc_ovr"),
                ])


    print(f"Done. Summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
