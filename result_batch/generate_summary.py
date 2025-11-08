#!/usr/bin/env python3
"""
脚本用于读取result_batch目录中的JSON结果文件，并生成汇总的CSV文件
"""

import json
import csv
import os
from pathlib import Path


def read_results(result_dir):
    """读取指定目录中的所有JSON结果文件"""
    results = []

    # 遍历目录中的所有JSON文件
    for json_file in Path(result_dir).glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 提取数据集名称和指标
                dataset_name = data.get('dataset', json_file.stem)
                overall = data.get('overall', {})

                # 构建结果字典
                result = {
                    'Dataset': dataset_name,
                    'Precision': overall.get('precision', 0),
                    'Sensitivity': overall.get('sensitivity', 0),
                    'Specificity': overall.get('specificity', 0),
                    'F1-score': overall.get('f1', 0),
                    'Overall Accuracy': overall.get('accuracy', 0),
                    'AUC': overall.get('auc_ovr', 0)
                }

                results.append(result)
                print(f"已读取: {json_file.name}")

        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")

    return results


def write_csv(results, output_file):
    """将结果写入CSV文件"""
    if not results:
        print("没有数据可写入")
        return

    # 定义CSV列名
    fieldnames = ['Dataset', 'Precision', 'Sensitivity', 'Specificity',
                  'F1-score', 'Overall Accuracy', 'AUC']

    # 写入CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 按数据集名称排序
        results.sort(key=lambda x: x['Dataset'])

        for result in results:
            writer.writerow(result)

    print(f"\nCSV文件已生成: {output_file}")


def main():
    # 设置路径
    result_dir = "/export/home2/junhao003/Yuqing/MedMamba/result_batch"
    output_file = os.path.join(result_dir, "summary_results.csv")

    print("开始读取结果文件...")
    results = read_results(result_dir)

    print(f"\n共读取 {len(results)} 个数据集的结果")

    write_csv(results, output_file)

    # 打印汇总信息
    print("\n结果汇总:")
    print("-" * 80)
    for result in sorted(results, key=lambda x: x['Dataset']):
        print(f"{result['Dataset']:20s} | "
              f"Acc: {result['Overall Accuracy']:.4f} | "
              f"F1: {result['F1-score']:.4f} | "
              f"AUC: {result['AUC']:.4f}")


if __name__ == "__main__":
    main()
