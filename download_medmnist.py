#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedMNIST 数据集下载与转换（适配 MedMamba / ImageFolder-224）
- 强制 RGB 三通道
- 稳健标签转换（int / array / one-hot）
- 不做错误的反归一化
- 写 labels.json 记录类顺序与官方类名
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import medmnist
    from medmnist import INFO, PathMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, \
                         RetinaMNIST, BreastMNIST, BloodMNIST, OrganAMNIST, \
                         OrganCMNIST, OrganSMNIST
except ImportError:
    print("❌ 错误: 未安装 medmnist 包，请先运行: pip install medmnist")
    sys.exit(1)

# 支持的数据集（单标签任务）
DATASET_MAP = {
    # 'pathmnist': PathMNIST,
    # 'dermamnist': DermaMNIST,
    # 'octmnist': OCTMNIST,
    # 'pneumoniamnist': PneumoniaMNIST,
    # 'retinamnist': RetinaMNIST,
    # 'breastmnist': BreastMNIST,
    # 'bloodmnist': BloodMNIST,
    'organamnist': OrganAMNIST,
    # 'organcmnist': OrganCMNIST,
    # 'organsmnist': OrganSMNIST,
}

def print_dataset_info():
    print("\n" + "=" * 80)
    print("MedMNIST 数据集信息".center(80))
    print("=" * 80)
    for idx, (dataset_name, info) in enumerate(INFO.items(), 1):
        if dataset_name in DATASET_MAP:
            label_names = info.get('label', {})
            print(f"\n{idx}. {dataset_name.upper()}")
            print(f"   ├─ 任务类型: {info['task']}")
            print(f"   ├─ 类别数量: {len(label_names)}")
            print(f"   ├─ 图像尺寸: 28x28 (将转换为 224x224 RGB)")
            print(f"   └─ 类别名称: {label_names}")
    print("\n" + "=" * 80 + "\n")

def _ensure_pil_rgb(img):
    """将各种输入（PIL/ndarray/torch.Tensor）稳健转换为 RGB PIL.Image。"""
    try:
        import torch
        is_tensor = torch.is_tensor(img)
    except Exception:
        is_tensor = False

    if is_tensor:
        img = img.detach().cpu().numpy()

    if isinstance(img, np.ndarray):
        # 数值域可能是 [0,1] 或 [0,255]
        if img.ndim == 2:  # H,W
            arr = img
        elif img.ndim == 3:
            # (C,H,W) 或 (H,W,C)
            if img.shape[0] in (1,3) and img.ndim == 3:
                # 假定 (C,H,W)
                arr = np.transpose(img, (1, 2, 0))
            else:
                arr = img
        else:
            raise ValueError(f"Unsupported ndarray shape: {img.shape}")

        # 归一化到 0–255 uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).round()
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L")
        elif arr.shape[2] == 1:
            pil = Image.fromarray(arr.squeeze(2), mode="L")
        elif arr.shape[2] == 3:
            pil = Image.fromarray(arr, mode="RGB")
        else:
            # 超过 3 通道，取前三个通道
            pil = Image.fromarray(arr[:, :, :3], mode="RGB")
        return pil.convert("RGB")

    # PIL.Image
    try:
        from PIL.Image import Image as PILImage
        if isinstance(img, PILImage):
            return img.convert("RGB")
    except Exception:
        pass

    # 兜底：再次尝试从 numpy 转
    return _ensure_pil_rgb(np.array(img))

def _label_to_int(label):
    """将 medmnist 的 label（int/np.array/one-hot）转为 int。"""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if hasattr(label, 'item'):
        try:
            return int(label.item())
        except Exception:
            pass
    label = np.array(label)
    if label.ndim == 0:
        return int(label)
    # one-hot 或 shape (1,)
    return int(label.argmax())

def convert_to_imagefolder(dataset, output_dir, split_name):
    """将 MedMNIST 数据集转换为 ImageFolder(split/class_i/xxx.png)"""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # 统计类别数（从元信息而非 labels 推断更稳）
    # 部分数据集 labels 可能不全覆盖；从 INFO 使用官方类数
    dataset_name = dataset.__class__.__name__.replace('MNIST', '').lower() + 'mnist'
    if dataset_name not in INFO:
        # 回退：用数据里的 labels
        num_classes = int(len(np.unique(dataset.labels)))
    else:
        num_classes = int(len(INFO[dataset_name]['label']))

    # 创建类别目录 class_0 ... class_{K-1}
    for c in range(num_classes):
        os.makedirs(os.path.join(split_dir, f"class_{c}"), exist_ok=True)

    # 遍历保存
    for idx in tqdm(range(len(dataset)), desc=f"   {split_name}", leave=False):
        img, label = dataset[idx]
        img_pil = _ensure_pil_rgb(img).resize((224, 224), Image.BILINEAR)
        y = _label_to_int(label)
        y = max(0, min(num_classes - 1, y))  # clamp 防越界
        out_path = os.path.join(split_dir, f"class_{y}", f"{split_name}_{idx:06d}.png")
        img_pil.save(out_path)

    return num_classes

def download_dataset(dataset_key, output_base_dir, raw_only=False):
    """下载并转换指定的 MedMNIST 数据集"""
    dataset_key = dataset_key.lower()
    if dataset_key not in DATASET_MAP:
        print(f"❌ 错误: 未知数据集 '{dataset_key}'，可用: {list(DATASET_MAP.keys())}")
        return False

    print("\n" + "=" * 80)
    print(f"下载数据集: {dataset_key.upper()}".center(80))
    print("=" * 80 + "\n")

    DataClass = DATASET_MAP[dataset_key]

    raw_dir = os.path.join(output_base_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    try:
        # 直接要求 RGB 三通道
        print("📥 下载原始数据 (as_rgb=True)...")
        train_ds = DataClass(split='train', download=True, root=raw_dir, as_rgb=True)
        val_ds   = DataClass(split='val',   download=True, root=raw_dir, as_rgb=True)
        test_ds  = DataClass(split='test',  download=True, root=raw_dir, as_rgb=True)

        print(f"   ✓ 训练: {len(train_ds)}   ✓ 验证: {len(val_ds)}   ✓ 测试: {len(test_ds)}")

        if raw_only:
            print(f"\n✅ 原始数据下载完成 -> {raw_dir}")
            return True

        # 转 ImageFolder-224
        print("\n🔄 转换为 ImageFolder-224 ...")
        imagefolder_dir = os.path.join(output_base_dir, 'imagefolder_224')
        K_train = convert_to_imagefolder(train_ds, imagefolder_dir, 'train')
        K_val   = convert_to_imagefolder(val_ds,   imagefolder_dir, 'val')
        K_test  = convert_to_imagefolder(test_ds,  imagefolder_dir, 'test')
        assert K_train == K_val == K_test, "训练/验证/测试 类别数不一致"

        # 写 labels.json（类顺序与文件夹顺序一致）
        labels_json = {
            "dataset": dataset_key,
            "num_classes": K_train,
            "folder_classes": [f"class_{i}" for i in range(K_train)],
            "official_class_names": INFO[dataset_key].get("label", {}),
        }
        with open(os.path.join(imagefolder_dir, "labels.json"), "w") as f:
            json.dump(labels_json, f, indent=2, ensure_ascii=False)

        # 写 dataset_info.txt
        info_file = os.path.join(imagefolder_dir, 'dataset_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"数据集: {dataset_key}\n")
            f.write(f"类别数: {K_train}\n")
            f.write(f"图像: 224x224 RGB\n")
            f.write(f"训练/验证/测试: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}\n\n")
            f.write("官方信息:\n")
            for k, v in INFO[dataset_key].items():
                f.write(f"  {k}: {v}\n")

        print("\n✅ 数据集处理完成！")
        print(f"   📁 原始: {raw_dir}")
        print(f"   📁 ImageFolder: {imagefolder_dir}")
        print(f"   📄 labels.json / dataset_info.txt 已生成")
        print(f"\n目录预览：\n  {imagefolder_dir}/train|val|test/class_*/xxx.png")
        return True

    except Exception as e:
        print(f"❌ 下载/转换失败: {e}")
        import traceback; traceback.print_exc()
        return False

def download_all_datasets(output_base_dir, raw_only=False):
    print("\n" + "=" * 80)
    print("开始下载所有 MedMNIST 数据集".center(80))
    print("=" * 80)
    succ, fail = 0, []
    for key in DATASET_MAP.keys():
        ok = download_dataset(key, os.path.join(output_base_dir, key), raw_only)
        succ += int(ok)
        if not ok: fail.append(key)
        print()
    print("=" * 80)
    print(f"下载完成：成功 {succ}/{len(DATASET_MAP)}")
    if fail: print("失败：", ", ".join(fail))
    print("=" * 80)

def main():
    ap = argparse.ArgumentParser(
        description="MedMNIST 下载/转换工具（ImageFolder-224）"
    )
    ap.add_argument("--info", action="store_true", help="显示可用数据集信息")
    ap.add_argument("--dataset", type=str, choices=list(DATASET_MAP.keys()),
                    help="指定下载的数据集键名")
    ap.add_argument("--all", action="store_true", help="下载所有支持的数据集")
    ap.add_argument("--output", type=str, default="./medmnist_data",
                    help="输出根目录（默认 ./medmnist_data）")
    ap.add_argument("--raw-only", action="store_true", help="仅下载原始数据，不做转换")
    args = ap.parse_args()

    if args.info:
        print_dataset_info()
        return

    if not args.dataset and not args.all:
        print("❌ 需要 --dataset 或 --all；用 --info 查看数据集列表")
        sys.exit(1)

    out_root = os.path.abspath(args.output)
    os.makedirs(out_root, exist_ok=True)
    print(f"\n📁 输出目录: {out_root}")

    if args.all:
        download_all_datasets(out_root, args.raw_only)
    else:
        ds_root = os.path.join(out_root, args.dataset.lower())
        download_dataset(args.dataset.lower(), ds_root, args.raw_only)

    print("\n🎉 完成！\n")

if __name__ == "__main__":
    main()
