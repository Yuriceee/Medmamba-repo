import os
import json
import argparse
import numpy as np
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score
)

from MedMamba import VSSM

DATASETS = [
    # "bloodmnist",
    # "kvasir",
    # "pneumoniamnist",
    # "dermamnist",
    # "retinamnist",
    # "breastmnist",
    # "pad_ufes_20",
    "octmnist",
    "organamnist",
    # ...
]

DATASET_NORM = {
    # "pad_ufes_20": "half",
    # "kvasir": "half",
    # "bloodmnist": "half",
    # "dermamnist": "half",
    # "pneumoniamnist": "half",
    # "retinamnist": "half",
    "octmnist": "half",
    "organamnist": "half",
}

def build_transform(norm):
    if norm == "imagenet":
        mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    else:
        mean, std = [0.5,0.5,0.5], [0.5,0.5,0.5]
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

class NpzMedMNISTTest(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        imgs, labels = data["test_images"], data["test_labels"]
        if labels.ndim == 2: labels = labels[:,0]
        self.labels = labels.astype(np.int64)
        self.imgs = imgs
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if img.ndim == 2:
            pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        else:
            pil = Image.fromarray(img.squeeze().astype(np.uint8)).convert("RGB")
        if self.transform: pil = self.transform(pil)
        return pil, int(self.labels[idx])

def build_dataset_detect(name, data_root, transform):
    dataset_root = os.path.join(data_root, name)

    if name.endswith("mnist"):
        npz_path = os.path.join(dataset_root, "raw", f"{name}.npz")
        ds = NpzMedMNISTTest(npz_path, transform)
        uniq = np.unique(ds.labels)
        num_classes = uniq.size
        class_names = [f"class_{i}" for i in range(num_classes)]
        class_to_idx = {c:i for i,c in enumerate(class_names)}
        return ds, class_names, num_classes, npz_path, class_to_idx

    test_root = os.path.join(dataset_root, "test")
    ds = datasets.ImageFolder(test_root, transform)
    return ds, ds.classes, len(ds.classes), test_root, ds.class_to_idx

def safe_auc(labels, probs, num_classes):
    try:
        if num_classes == 2:
            return float(roc_auc_score(labels, probs[:,1]))
        else:
            return float(roc_auc_score(labels, probs, multi_class="ovr"))
    except:
        return None

def one_dataset_eval(name, data_root, weights, weights_root, weights_name,
                     bs, nw, norm_override, save_dir, device):

    norm = norm_override or DATASET_NORM.get(name, "imagenet")
    transform = build_transform(norm)

    ds, class_names, num_classes, data_src, class_to_idx = build_dataset_detect(
        name, data_root, transform
    )

    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    weight_path = weights if weights else os.path.join(weights_root, name, weights_name)
    net = VSSM(depths=[2,2,4,2], dims=[96,192,384,768], num_classes=num_classes).to(device)
    net.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    net.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            p = torch.softmax(net(x),1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(p.argmax(1).cpu().numpy())
            all_probs.extend(p.cpu().numpy())

    all_labels, all_preds, all_probs = map(np.array, (all_labels, all_preds, all_probs))

    accuracy = accuracy_score(all_labels, all_preds)
    auc = safe_auc(all_labels, all_probs, num_classes)
    cm = confusion_matrix(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    sensitivity = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    spec_list = []
    for i in range(cm.shape[0]):
        tp = cm[i,i]
        fp = cm[:,i].sum() - tp
        fn = cm[i,:].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)
    specificity = np.mean(spec_list)

    print(f"{name}\tOA: {accuracy*100:.2f}%\tAUC: {auc if auc else 'N/A'}")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{name}.json"), "w") as f:
        json.dump({
            "dataset": name,
            "data_path": data_src,
            "weights": weight_path,
            "classes": class_names,
            "overall": {
                "accuracy": accuracy,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1,
                "auc_ovr": auc,
            },
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_root", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--weights_root", default=None)
    ap.add_argument("--weights_name", default="Medmamba.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--save_dir", default="./result_batch")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("DATASET\tOA\tAUC")
    for name in DATASETS:
        one_dataset_eval(
            name,
            data_root=args.data_root_root,
            weights=args.weights,
            weights_root=args.weights_root,
            weights_name=args.weights_name,
            bs=args.batch_size,
            nw=args.num_workers,
            norm_override=args.norm,
            save_dir=args.save_dir,
            device=device
        )

if __name__ == "__main__":
    main()
