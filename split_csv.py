"""
将CSV文件分割成两个文件：
1. 包含 x1 到 y2 的列
2. 包含 y1 到 filename 的列
"""

import pandas as pd
import argparse
import os


def split_csv(input_file, output_dir=None):
    """
    分割CSV文件
    
    Args:
        input_file: 输入的CSV文件路径
        output_dir: 输出目录，如果为None则使用输入文件所在目录
    """
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 获取列名
    columns = df.columns.tolist()
    print(f"总列数: {len(columns)}")
    
    # 找到关键列的索引
    x1_idx = columns.index('x1') if 'x1' in columns else None
    y1_idx = columns.index('y1') if 'y1' in columns else None
    y2_idx = columns.index('y2') if 'y2' in columns else None
    filename_idx = columns.index('filename') if 'filename' in columns else None
    
    if x1_idx is None:
        raise ValueError("未找到列 'x1'")
    if y1_idx is None:
        raise ValueError("未找到列 'y1'")
    if y2_idx is None:
        raise ValueError("未找到列 'y2'")
    if filename_idx is None:
        raise ValueError("未找到列 'filename'")
    
    # 创建第一个文件：x1 到 y2
    cols_file1 = columns[x1_idx:y2_idx + 1]
    df_file1 = df[cols_file1]
    
    # 创建第二个文件：y1 到 filename
    cols_file2 = columns[y1_idx:filename_idx + 1]
    df_file2 = df[cols_file2]
    
    # 确定输出目录和文件名
    if output_dir is None:
        output_dir = os.path.dirname(input_file) or '.'
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file1 = os.path.join(output_dir, f"{base_name}_x1_to_y2.csv")
    output_file2 = os.path.join(output_dir, f"{base_name}_y1_to_filename.csv")
    
    # 保存文件
    print(f"正在保存文件1: {output_file1}")
    print(f"  包含列: {len(cols_file1)} 列 (x1 到 y2)")
    df_file1.to_csv(output_file1, index=False)
    
    print(f"正在保存文件2: {output_file2}")
    print(f"  包含列: {len(cols_file2)} 列 (y1 到 filename)")
    df_file2.to_csv(output_file2, index=False)
    
    print("\n分割完成！")
    print(f"文件1: {output_file1}")
    print(f"文件2: {output_file2}")


def main():
    parser = argparse.ArgumentParser(description='将CSV文件分割成两个文件')
    parser.add_argument('input_file', type=str, help='输入的CSV文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='输出目录（默认为输入文件所在目录）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误: 文件不存在: {args.input_file}")
        return
    
    split_csv(args.input_file, args.output)


if __name__ == '__main__':
    main()

