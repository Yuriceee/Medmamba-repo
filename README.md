# MedMamba Reproduction and Comparison Experiment Guide

This document provides a complete reproduction process for the MedMamba model and a comparison experiment guide with other baseline methods.

---

## 📋 Table of Contents

- [Part 1: MedMamba Model Reproduction](#part-1-medmamba-model-reproduction)
  - [1.1 Dataset Preparation](#11-dataset-preparation)
  - [1.2 Model Training](#12-model-training)
  - [1.3 Model Testing](#13-model-testing)
  - [1.4 Result Summary and Visualization](#14-result-summary-and-visualization)
- [Part 2: Baseline Method Comparison Experiments](#part-2-baseline-method-comparison-experiments)
  - [2.1 Baseline Method Training](#21-baseline-method-training)
  - [2.2 Baseline Method Testing](#22-baseline-method-testing)
  - [2.3 Complete Result Comparison](#23-complete-result-comparison)
- [Experimental Results Summary](#experimental-results-summary)

---

## Part 1: MedMamba Model Reproduction

### 1.1 Dataset Preparation

#### 🔄 Automatic Download (MedMNIST Series)

The MedMNIST dataset contains the following 5 medical image classification datasets:

| Dataset | Classes | Samples | Task Description |
|---------|---------|---------|------------------|
| **BloodMNIST** | 8 | 17,092 | Blood cell image classification |
| **BreastMNIST** | 2 | 780 | Breast ultrasound image classification |
| **DermaMNIST** | 7 | 10,015 | Skin lesion image classification |
| **PneumoniaMNIST** | 2 | 5,856 | Pneumonia X-ray image classification |
| **RetinaMNIST** | 5 | 1,600 | Retinal OCT image classification |

**Download Command:**
```bash
# Download all MedMNIST datasets
python download_medmnist.py --all

# Data will be saved to: ./medmnist_data/
```

#### 📥 Manual Download Datasets

**1. Kvasir-V1 Dataset**
- Description: Gastrointestinal endoscopy images, 8 classes
- Download link: https://datasets.simula.no/kvasir/
- Extract to: `./data/kvasir_v1/`

**2. PAD-UFES-20 Dataset**
- Description: Skin lesion images, 6 classes
- Download link: https://data.mendeley.com/datasets/zr7vgbcyr2/1
- Extract to: `./data/pad_ufes_20/`

---

### 1.2 Model Training

#### Training Command

```bash
python train.py
```

The training script will automatically train all 7 datasets (BloodMNIST, BreastMNIST, DermaMNIST, PneumoniaMNIST, RetinaMNIST, Kvasir, PAD-UFES-20).

#### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_flag` | Dataset name (bloodmnist/breastmnist/dermamnist/pneumoniamnist/retinamnist/kvasir/pad_ufes_20) | Required |
| `--model_flag` | Model name (medmamba/resnet50/convnext_tiny/swin_tiny_patch4_window7_224/tf_efficientnetv2_s/deit_small_patch16_224) | `medmamba` |
| `--output_root` | Training output root directory | `./outputs` |
| `--num_epochs` | Number of training epochs | `100` |
| `--gpu_ids` | GPU device ID | `0` |
| `--batch_size` | Batch size | `128` |
| `--lr` | Learning rate | `0.001` |
| `--data_path` | Dataset path (for manually downloaded datasets) | None |
| `--model_path` | Pretrained weight path | None |

---

### 1.3 Model Testing

#### Testing Command

```bash
python test.py \
  --data_root ./medmnist_data \
  --weights_root ./finetuned_weights \
  --save_dir ./result_batch
```

The testing script will automatically test all trained models and save results to the `result_batch/` directory.

#### Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_root` | Dataset root directory | `./medmnist_data` |
| `--weights_root` | Model weights root directory | `./finetuned_weights` |
| `--save_dir` | Result save directory | `./result_batch` |
| `--data_flag` | Specify a dataset to test (optional) | None (test all) |
| `--model_flag` | Specify a model to test (optional) | None (test all) |
| `--gpu_ids` | GPU device ID | `0` |

---

### 1.4 Result Summary and Visualization

#### Generate Result Summary CSV

After testing is complete, results will be saved in the `result_batch/` directory, with one JSON file per dataset.

```bash
# Enter result directory
cd result_batch

# Run summary script
python generate_summary.py

# Generated file: summary_results.csv
```

**summary_results.csv** contains the following metrics:
- Dataset: Dataset name
- Precision: Precision
- Sensitivity: Sensitivity (Recall)
- Specificity: Specificity
- F1-score: F1 score
- Overall Accuracy: Overall accuracy
- AUC: Area Under ROC Curve

#### Generate Confusion Matrix Visualization

```bash
# Run in result_batch directory
cd result_batch
python plot_confusion_matrices.py

# Output directory: confusion_matrices/
```

**Generated visualization files:**

1. **Individual dataset confusion matrices** (2 images per dataset):
   - `[dataset]_confusion_matrix.png` - Original counts
   - `[dataset]_confusion_matrix_normalized.png` - Normalized (percentages)

2. **Combined plot for all datasets**:
   - `all_confusion_matrices_combined.png` - Grid layout showing all datasets

---

## Part 2: Baseline Method Comparison Experiments

This section compares MedMamba with the following 5 mainstream methods:

| Method | Model ID | Description |
|--------|----------|-------------|
| **ResNet-50** | `resnet50` | Classic residual network |
| **ConvNeXt-Tiny** | `convnext_tiny` | Modern convolutional network |
| **Swin Transformer** | `swin_tiny_patch4_window7_224` | Hierarchical vision transformer |
| **EfficientNetV2-S** | `tf_efficientnetv2_s` | Efficient convolutional network |
| **DeiT-Small** | `deit_small_patch16_224` | Data-efficient transformer |

---

### 2.1 Baseline Method Training

#### Training Command

```bash
python comparason.py \
  --mode train \
  --data_root /path/to/dataset \
  --train_dir train \
  --val_dir val \
  --test_dir test \
  --epochs 50 \
  --batch_size 64 \
  --img_size 224 \
  --output_dir outputs/exp_baselines/[dataset_name] \
  --amp
```

**Example: Training Kvasir Dataset**
```bash
python comparason.py \
  --mode train \
  --data_root /export/home2/junhao003/Yuqing/MedMamba/medmnist_data/kvasir \
  --train_dir train \
  --val_dir val \
  --test_dir test \
  --epochs 50 \
  --batch_size 64 \
  --img_size 224 \
  --output_dir outputs/exp_baselines/kvasir \
  --amp
```

#### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Run mode (train/test) | Required |
| `--data_root` | Dataset root directory | Required |
| `--train_dir` | Training set subdirectory name | `train` |
| `--val_dir` | Validation set subdirectory name | `val` |
| `--test_dir` | Test set subdirectory name | `test` |
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size | `64` |
| `--img_size` | Image size | `224` |
| `--output_dir` | Output directory | Required |
| `--amp` | Use mixed precision training | False |
| `--model` | Model name | Set in script |

---

### 2.2 Baseline Method Testing

#### Testing Command

```bash
python comparason.py \
  --mode test \
  --data_root /path/to/dataset \
  --test_dir test \
  --output_dir outputs/exp_baselines/[dataset_name]
```

**Example: Testing PAD-UFES-20 Dataset**
```bash
python comparason.py \
  --mode test \
  --data_root /export/home2/junhao003/Yuqing/MedMamba/medmnist_data/pad_ufes_20 \
  --test_dir test \
  --output_dir outputs/exp_baselines/pad_ufes_20
```

#### Testing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Run mode (set to test) | Required |
| `--data_root` | Dataset root directory | Required |
| `--test_dir` | Test set subdirectory name | `test` |
| `--output_dir` | Output directory (for model weights and results) | Required |

---

### 2.3 Complete Result Comparison

#### Generate Comparison Results

```bash
# Generate baseline result summary
cd outputs/exp_baselines
python generate_baseline_summary.py

# Generate complete comparison of MedMamba and Baselines
python generate_combined_summary.py
```

**Generated files:**
- `baseline_summary.csv`: Summary of all baseline results
- `combined_summary.csv`: Complete comparison of MedMamba + Baselines
- `combined_summaries/`: Individual comparison files for each dataset

---

## File Organization

```
MedMamba/
├── README_REPRODUCTION_EN.md        # This document
├── train.py                          # MedMamba training script
├── test.py                           # MedMamba testing script
├── comparason.py                     # Baseline training and testing script
├── download_medmnist.py             # MedMNIST data download script
├── data/                             # Dataset directory
│   ├── medmnist_data/               # MedMNIST (auto-download)
│   ├── kvasir_v1/                   # Kvasir (manual download)
│   └── pad_ufes_20/                 # PAD-UFES-20 (manual download)
├── finetuned_weights/               # MedMamba trained weights
│   ├── bloodmnist/
│   ├── breastmnist/
│   └── ...
├── outputs/                          # Output directory
│   └── exp_baselines/               # Baseline training output
│       ├── bloodmnist/
│       │   ├── resnet50/
│       │   ├── convnext_tiny/
│       │   └── ...
│       ├── baseline_summary.csv     # Baseline results summary
│       ├── combined_summary.csv     # Complete comparison results
│       ├── generate_baseline_summary.py  # Baseline summary script
│       └── generate_combined_summary.py  # Complete comparison script
├── result_batch/                     # MedMamba test results
│   ├── bloodmnist.json
│   ├── breastmnist.json
│   ├── ...
│   ├── summary_results.csv          # MedMamba results summary
│   ├── generate_summary.py          # Summary script
│   ├── plot_confusion_matrices.py   # Confusion matrix visualization script
│   └── confusion_matrices/          # Confusion matrix images
│       ├── bloodmnist_confusion_matrix.png
│       ├── bloodmnist_confusion_matrix_normalized.png
│       └── all_confusion_matrices_combined.png
```

---


## Citation

If this reproduction work is helpful to you, please cite the original MedMamba paper:

```bibtex
@article{yue2024medmamba,
  title={MedMamba: Vision Mamba for Medical Image Classification},
  author={Yue, Yubiao and Li, Zhenzhang},
  journal={arXiv preprint arXiv:2403.03849},
  year={2024}
}
```

---

## Update Log

- **2024-11-05**: Completed all experiments and documentation
  - ✅ MedMamba model reproduction
  - ✅ 7 dataset training and testing
  - ✅ 5 baseline method comparison
  - ✅ Complete documentation

---

**Good luck with your experiments!** 🎉
