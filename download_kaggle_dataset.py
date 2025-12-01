#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载并处理Kaggle数据集：naturalassignment
这个脚本会：
1. 使用kagglehub下载数据集
2. 解压并整理图片文件
3. 准备好供API.py处理的图片目录

依赖：
  pip install kagglehub

使用方法：
  python download_kaggle_dataset.py
  
  可选参数：
  --output-dir: 指定输出目录（默认：./kaggle_images）
  --limit: 只复制前N张图片（用于测试，顺序选择）
  --random-sample: 随机抽取N张图片（随机选择）
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("错误：未安装 kagglehub")
    print("请运行: pip install kagglehub")
    sys.exit(1)


def download_dataset():
    """下载Kaggle数据集"""
    print("正在下载数据集: naufalariqpyosyam/naturalassignment")
    print("这可能需要几分钟时间...")
    
    try:
        path = kagglehub.dataset_download("naufalariqpyosyam/naturalassignment")
        print(f"✓ 数据集已下载到: {path}")
        return path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n可能的原因：")
        print("1. 未登录Kaggle账号（需要运行 kaggle configure 或设置环境变量）")
        print("2. 网络连接问题")
        print("3. 数据集不存在或已被移除")
        sys.exit(1)


def collect_images(source_dir, extensions=None):
    """收集所有图片文件"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    image_files = []
    source_path = Path(source_dir)
    
    print(f"\n正在扫描目录: {source_dir}")
    
    for ext in extensions:
        found = list(source_path.rglob(f"*{ext}"))
        found.extend(list(source_path.rglob(f"*{ext.upper()}")))
        image_files.extend(found)
    
    # 去重并排序
    image_files = sorted(set(image_files))
    
    print(f"✓ 找到 {len(image_files)} 张图片")
    return image_files


def organize_images(source_dir, output_dir, limit=None, random_sample=None):
    """整理图片到指定目录"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图片
    image_files = collect_images(source_dir)
    
    if not image_files:
        print("❌ 未找到任何图片文件")
        return 0
    
    # 应用随机抽样（优先于limit）
    if random_sample is not None:
        total_count = len(image_files)
        sample_count = random_sample
        if sample_count > total_count:
            print(f"⚠ 请求的随机抽样数量 ({sample_count}) 大于总图片数 ({total_count})，将使用所有图片")
            sample_count = total_count
        
        print(f"⚠ 随机抽样：从 {total_count} 张图片中随机抽取 {sample_count} 张")
        image_files = random.sample(image_files, sample_count)
    # 应用限制（顺序选择，仅在未使用随机抽样时）
    elif limit:
        print(f"⚠ 限制：只处理前 {limit} 张图片")
        image_files = image_files[:limit]
    
    # 复制图片
    print(f"\n正在复制图片到: {output_dir}")
    copied_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            # 生成新的文件名（保持原扩展名）
            new_name = f"image_{i:04d}{img_path.suffix.lower()}"
            dest_path = output_path / new_name
            
            # 复制文件
            shutil.copy2(img_path, dest_path)
            copied_count += 1
            
            if copied_count % 100 == 0:
                print(f"  已复制 {copied_count}/{len(image_files)} 张图片...")
                
        except Exception as e:
            print(f"⚠ 复制失败 {img_path.name}: {e}")
            continue
    
    print(f"✓ 成功复制 {copied_count} 张图片")
    return copied_count


def create_readme(output_dir, dataset_path, image_count):
    """创建README文件说明数据来源"""
    readme_path = Path(output_dir) / "README.md"
    
    content = f"""# Kaggle数据集：naturalassignment

## 数据来源
- 数据集：naufalariqpyosyam/naturalassignment
- 原始路径：{dataset_path}
- 图片数量：{image_count}

## 使用方法

### 1. 使用API处理（API embedding）
```bash
python API.py \\
  --image-dir {output_dir} \\
  --out ./kaggle_dataset.csv \\
  --axis1 "情感：消极(0)↔积极(1)" \\
  --axis2 "活力：低能量(0)↔高能量(1)"
```

### 2. 使用本地embedding模型
```bash
python API.py \\
  --image-dir {output_dir} \\
  --out ./kaggle_dataset.csv \\
  --embed-model d:/llm/huggingface/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \\
  --use-local-embedding \\
  --axis1 "情感：消极(0)↔积极(1)" \\
  --axis2 "活力：低能量(0)↔高能量(1)"
```

### 3. 保存描述并启用详细输出
```bash
python API.py \\
  --image-dir {output_dir} \\
  --out ./kaggle_dataset.csv \\
  --use-local-embedding \\
  --embed-model <your-model-path> \\
  --save-descriptions ./kaggle_descriptions.json \\
  --verbose
```

## 数据集信息
- 下载时间：{Path(dataset_path).stat().st_mtime if Path(dataset_path).exists() else 'Unknown'}
- 整理后的图片存储在当前目录
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ 已创建说明文件: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="下载并整理Kaggle数据集：naturalassignment"
    )
    parser.add_argument(
        "--output-dir",
        default="./kaggle_images",
        help="输出目录（默认：./kaggle_images）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前N张图片（用于测试，顺序选择）"
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="随机抽取N张图片进行处理（随机选择）"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载，直接使用已下载的数据集（需要指定--dataset-path）"
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="已下载的数据集路径（配合--skip-download使用）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Kaggle数据集下载和整理工具")
    print("=" * 60)
    
    # 下载数据集
    if args.skip_download:
        if not args.dataset_path:
            print("❌ 使用 --skip-download 时必须指定 --dataset-path")
            sys.exit(1)
        dataset_path = args.dataset_path
        print(f"使用已存在的数据集: {dataset_path}")
    else:
        dataset_path = download_dataset()
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 整理图片
    image_count = organize_images(dataset_path, args.output_dir, args.limit, args.random_sample)
    
    if image_count == 0:
        print("\n❌ 没有成功复制任何图片")
        sys.exit(1)
    
    # 创建README
    create_readme(args.output_dir, dataset_path, image_count)
    
    # 显示下一步操作
    print("\n" + "=" * 60)
    print("✓ 完成！")
    print("=" * 60)
    print(f"\n图片已准备好在: {args.output_dir}")
    print(f"共 {image_count} 张图片")
    print("\n下一步操作：")
    print(f"  python API.py --image-dir {args.output_dir} --out kaggle_dataset.csv")
    print("\n查看详细使用说明：")
    print(f"  cat {args.output_dir}/README.md")


if __name__ == "__main__":
    main()

