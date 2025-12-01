#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键处理Kaggle数据集的便捷脚本

这个脚本会：
1. 检查环境和依赖
2. 下载Kaggle数据集
3. 调用API.py处理图片
4. 生成最终的CSV文件

使用方法：
  # 基础用法（使用API embedding，默认随机抽取1000张图片）
  python process_kaggle_dataset.py
  
  # 使用本地embedding模型
  python process_kaggle_dataset.py --use-local
  
  # 测试模式（只处理前10张图片）
  python process_kaggle_dataset.py --test
  
  # 自定义随机抽样数量
  python process_kaggle_dataset.py --random-sample 500
  
  # 使用所有图片（不使用随机抽样）
  python process_kaggle_dataset.py --no-random-sample
  
  # 自定义坐标轴
  python process_kaggle_dataset.py \
    --axis1 "温度：冷(0)↔热(1)" \
    --axis2 "湿度：干燥(0)↔潮湿(1)"
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """检查必要的依赖是否已安装"""
    print("检查依赖...")
    
    missing = []
    
    try:
        import kagglehub
    except ImportError:
        missing.append("kagglehub")
    
    try:
        import openai
    except ImportError:
        missing.append("openai")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")
    
    if missing:
        print(f"❌ 缺少以下依赖: {', '.join(missing)}")
        print(f"\n请运行: pip install {' '.join(missing)}")
        return False
    
    print("✓ 所有基础依赖已安装")
    return True


def check_transformers(use_local):
    """检查transformers依赖（仅当使用本地模型时需要）"""
    if not use_local:
        return True
    
    try:
        import transformers
        import torch
        print("✓ transformers 和 torch 已安装（本地模型支持）")
        return True
    except ImportError:
        print("❌ 使用本地embedding需要安装 transformers 和 torch")
        print("\n请运行: pip install transformers torch")
        return False


def check_api_key():
    """检查Qwen API密钥"""
    # 尝试从.env文件加载
    try:
        from dotenv import load_dotenv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        pass
    
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("❌ 未找到 QWEN_API_KEY")
        print("\n请设置环境变量或创建 .env 文件")
        print("详见: API_KEY_SETUP.md")
        return False
    
    print("✓ QWEN_API_KEY 已配置")
    return True


def check_kaggle_auth():
    """检查Kaggle认证"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print("✓ Kaggle认证文件已找到")
        return True
    
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        print("✓ Kaggle环境变量已设置")
        return True
    
    print("❌ 未找到Kaggle认证")
    print("\n请配置Kaggle认证（详见 KAGGLE_DATASET_WORKFLOW.md）")
    return False


def run_download(output_dir, limit=None, random_sample=None):
    """下载并整理数据集"""
    print("\n" + "="*60)
    print("Step 1/2: 下载并整理数据集")
    print("="*60)
    
    # 获取脚本所在目录，确保能找到子脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_script = os.path.join(script_dir, "download_kaggle_dataset.py")
    
    # 使用 sys.executable 确保使用相同的 Python 解释器
    cmd = [sys.executable, download_script, "--output-dir", output_dir]
    
    # 随机抽样优先于limit
    if random_sample is not None:
        cmd.extend(["--random-sample", str(random_sample)])
    elif limit:
        cmd.extend(["--limit", str(limit)])
    
    try:
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n✓ 数据集下载和整理完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def run_api_processing(
    image_dir,
    output_csv,
    use_local,
    local_model_path,
    axis1,
    axis2,
    save_descriptions,
    verbose,
    limit=None
):
    """运行API.py处理图片"""
    print("\n" + "="*60)
    print("Step 2/2: 处理图片生成数据集")
    print("="*60)
    
    # 获取脚本所在目录，确保能找到子脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_script = os.path.join(script_dir, "API.py")
    
    # 使用 sys.executable 确保使用相同的 Python 解释器
    cmd = [
        sys.executable, api_script,
        "--image-dir", image_dir,
        "--out", output_csv,
        "--axis1", axis1,
        "--axis2", axis2,
    ]
    
    if use_local:
        if not local_model_path:
            print("❌ 使用本地模型需要指定 --local-model-path")
            return False
        cmd.extend([
            "--use-local-embedding",
            "--embed-model", local_model_path
        ])
    
    if save_descriptions:
        desc_file = output_csv.replace(".csv", "_descriptions.json")
        cmd.extend(["--save-descriptions", desc_file])
    
    if verbose:
        cmd.append("--verbose")
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n✓ 图片处理完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 处理失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="一键处理Kaggle数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基础用法
  python process_kaggle_dataset.py
  
  # 使用本地模型
  python process_kaggle_dataset.py --use-local --local-model-path /path/to/model
  
  # 测试模式
  python process_kaggle_dataset.py --test
  
  # 完整自定义
  python process_kaggle_dataset.py \\
    --use-local \\
    --local-model-path /path/to/model \\
    --output-dir ./my_images \\
    --output-csv ./my_dataset.csv \\
    --axis1 "温度：冷(0)↔热(1)" \\
    --axis2 "湿度：干(0)↔湿(1)" \\
    --random-sample 500 \\
    --save-descriptions \\
    --verbose
  
  # 使用所有图片（不使用随机抽样）
  python process_kaggle_dataset.py --no-random-sample
        """
    )
    
    parser.add_argument(
        "--output-dir",
        default="./kaggle_images",
        help="图片输出目录（默认：./kaggle_images）"
    )
    parser.add_argument(
        "--output-csv",
        default="./kaggle_dataset.csv",
        help="CSV输出文件（默认：./kaggle_dataset.csv）"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="使用本地embedding模型"
    )
    parser.add_argument(
        "--local-model-path",
        default=None,
        help="本地embedding模型路径"
    )
    parser.add_argument(
        "--axis1",
        default="自然度：人造(0)↔自然(1)",
        help="坐标轴1定义"
    )
    parser.add_argument(
        "--axis2",
        default="复杂度：简单(0)↔复杂(1)",
        help="坐标轴2定义"
    )
    parser.add_argument(
        "--save-descriptions",
        action="store_true",
        help="保存LLM生成的描述到JSON文件"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细处理信息"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：只处理前10张图片"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载步骤（假设图片已存在）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前N张图片（顺序选择，与--random-sample互斥）"
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=1000,
        help="随机抽取N张图片进行处理（默认：1000）"
    )
    parser.add_argument(
        "--no-random-sample",
        action="store_true",
        help="禁用随机抽样，使用所有图片（覆盖--random-sample默认值）"
    )
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test:
        args.limit = 10
        args.random_sample = None  # 测试模式不使用随机抽样
        print("🧪 测试模式：只处理前10张图片\n")
    
    # 处理随机抽样参数
    if args.no_random_sample:
        # 用户明确要求不使用随机抽样
        args.random_sample = None
    elif args.limit and args.random_sample == 1000:
        # 如果用户指定了limit，则禁用默认的random_sample（避免冲突）
        args.random_sample = None
        print(f"⚠ 检测到 --limit 参数，已禁用默认的随机抽样\n")
    
    print("="*60)
    print("Kaggle数据集一键处理工具")
    print("="*60)
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    if not check_transformers(args.use_local):
        sys.exit(1)
    
    if not check_api_key():
        sys.exit(1)
    
    if not args.skip_download:
        if not check_kaggle_auth():
            sys.exit(1)
    
    print("\n✓ 所有前置检查通过\n")
    
    # Step 1: 下载数据集
    if not args.skip_download:
        if not run_download(args.output_dir, args.limit, args.random_sample):
            sys.exit(1)
    else:
        print(f"⚠ 跳过下载，使用已存在的图片: {args.output_dir}")
        if not os.path.exists(args.output_dir):
            print(f"❌ 目录不存在: {args.output_dir}")
            sys.exit(1)
    
    # Step 2: 处理图片
    if not run_api_processing(
        args.output_dir,
        args.output_csv,
        args.use_local,
        args.local_model_path,
        args.axis1,
        args.axis2,
        args.save_descriptions,
        args.verbose,
        args.limit
    ):
        sys.exit(1)
    
    # 完成
    print("\n" + "="*60)
    print("🎉 全部完成！")
    print("="*60)
    print(f"\n输出文件：")
    print(f"  - CSV数据: {args.output_csv}")
    
    if args.save_descriptions:
        desc_file = args.output_csv.replace(".csv", "_descriptions.json")
        print(f"  - 描述JSON: {desc_file}")
    
    print(f"\n图片目录: {args.output_dir}")
    
    print("\n下一步：")
    print("  1. 使用pandas加载CSV进行分析")
    print("  2. 可视化y1, y2坐标分布")
    print("  3. 使用embedding向量训练模型")
    print("\n详见: KAGGLE_DATASET_WORKFLOW.md")


if __name__ == "__main__":
    main()

