#!/usr/bin/env python3
"""
脚本用于读取result_batch目录中的混淆矩阵数据并绘制成PNG图片
"""

import json
import numpy as np
# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def plot_confusion_matrix(cm, classes, dataset_name, output_path, normalize=False):
    """
    绘制混淆矩阵

    参数:
        cm: 混淆矩阵数组
        classes: 类别名称列表
        dataset_name: 数据集名称
        output_path: 输出文件路径
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'Normalized Confusion Matrix - {dataset_name}'
        fmt = '.2f'
    else:
        title = f'Confusion Matrix - {dataset_name}'
        fmt = 'd'

    # 设置图片大小
    fig_size = max(8, len(classes) * 0.8)
    plt.figure(figsize=(fig_size, fig_size))

    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                square=True, cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                linewidths=0.5, linecolor='gray')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {os.path.basename(output_path)}")


def read_and_plot_confusion_matrices(result_dir, output_dir, plot_normalized=True):
    """
    读取所有JSON文件中的混淆矩阵并绘制

    参数:
        result_dir: 结果文件目录
        output_dir: 输出图片目录
        plot_normalized: 是否同时绘制归一化版本
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 遍历所有JSON文件
    json_files = sorted(Path(result_dir).glob("*.json"))

    if not json_files:
        print(f"在 {result_dir} 中未找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个结果文件\n")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            dataset_name = data.get('dataset', json_file.stem)
            confusion_matrix = np.array(data.get('confusion_matrix', []))
            classes = data.get('classes', [])

            if confusion_matrix.size == 0:
                print(f"警告: {json_file.name} 中没有混淆矩阵数据")
                continue

            # 如果没有提供类别名称，使用默认名称
            if not classes or len(classes) != confusion_matrix.shape[0]:
                classes = [f'Class {i}' for i in range(confusion_matrix.shape[0])]

            print(f"处理: {dataset_name}")
            print(f"  类别数: {len(classes)}")
            print(f"  混淆矩阵形状: {confusion_matrix.shape}")

            # 绘制原始混淆矩阵
            output_file = output_path / f"{dataset_name}_confusion_matrix.png"
            plot_confusion_matrix(confusion_matrix, classes, dataset_name,
                                output_file, normalize=False)

            # 绘制归一化混淆矩阵
            if plot_normalized:
                output_file_norm = output_path / f"{dataset_name}_confusion_matrix_normalized.png"
                plot_confusion_matrix(confusion_matrix, classes, dataset_name,
                                    output_file_norm, normalize=True)

            print()

        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}\n")


def plot_all_confusion_matrices_grid(result_dir, output_file):
    """
    将所有混淆矩阵绘制在一个大图中（网格布局）

    参数:
        result_dir: 结果文件目录
        output_file: 输出文件路径
    """
    json_files = sorted(Path(result_dir).glob("*.json"))

    if not json_files:
        print(f"在 {result_dir} 中未找到JSON文件")
        return

    n_datasets = len(json_files)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))

    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            dataset_name = data.get('dataset', json_file.stem)
            confusion_matrix = np.array(data.get('confusion_matrix', []))
            classes = data.get('classes', [])

            if confusion_matrix.size == 0:
                continue

            if not classes or len(classes) != confusion_matrix.shape[0]:
                classes = [f'C{i}' for i in range(confusion_matrix.shape[0])]

            # 归一化
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

            ax = axes[idx]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=classes, yticklabels=classes,
                       square=True, ax=ax, cbar=True,
                       linewidths=0.5, linecolor='gray')

            ax.set_title(dataset_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('True', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.tick_params(axis='x', rotation=45)

        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")

    # 隐藏多余的子图
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('MedMamba Confusion Matrices - All Datasets',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n已保存合并图: {os.path.basename(output_file)}")


def print_summary(result_dir):
    """打印混淆矩阵统计信息"""
    json_files = sorted(Path(result_dir).glob("*.json"))

    print("\n" + "=" * 80)
    print("混淆矩阵统计信息")
    print("=" * 80)

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            dataset_name = data.get('dataset', json_file.stem)
            confusion_matrix = np.array(data.get('confusion_matrix', []))
            overall = data.get('overall', {})

            if confusion_matrix.size == 0:
                continue

            # 计算每个类别的准确率
            class_accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

            print(f"\n【{dataset_name}】")
            print(f"  总准确率: {overall.get('accuracy', 0):.4f}")
            print(f"  类别数量: {confusion_matrix.shape[0]}")
            print(f"  样本总数: {confusion_matrix.sum():.0f}")
            print(f"  各类别准确率:")
            for i, acc in enumerate(class_accuracies):
                print(f"    Class {i}: {acc:.4f}")

        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")


def main():
    # 设置路径
    result_dir = "/export/home2/junhao003/Yuqing/MedMamba/result_batch"
    output_dir = os.path.join(result_dir, "confusion_matrices")
    combined_output = os.path.join(output_dir, "all_confusion_matrices_combined.png")

    print("=" * 80)
    print("MedMamba 混淆矩阵可视化")
    print("=" * 80)
    print(f"\n输入目录: {result_dir}")
    print(f"输出目录: {output_dir}\n")

    # 设置matplotlib的字体和样式
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 绘制单独的混淆矩阵
    print("正在绘制各数据集的混淆矩阵...\n")
    read_and_plot_confusion_matrices(result_dir, output_dir, plot_normalized=True)

    # 绘制合并的网格图
    print("\n正在生成合并的混淆矩阵网格图...")
    plot_all_confusion_matrices_grid(result_dir, combined_output)

    # 打印统计信息
    print_summary(result_dir)

    print("\n" + "=" * 80)
    print("完成！所有混淆矩阵已保存到:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
