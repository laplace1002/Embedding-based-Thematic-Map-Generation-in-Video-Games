#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理流程（对应workflow图中的两个LLM生成步骤）：
1) 读取文件夹中的图片
2) LLM生成/图像本身描述：让多模态LLM生成中文描述 description
3) LLM生成空间位置向量：让多模态LLM对两个维度进行原始评分（无范围限制），然后使用sigmoid归一化到[0,1]或[-1,1]范围
4) 使用文本向量模型对 description 做 embedding（得到x1...xn特征向量）
5) 写出 CSV：x1,...,xN,y1,y2,filename（每个样本一行）

改进说明：
  - 旧方案：要求AI直接输出0-1范围的坐标，容易导致评分不准确且缺乏区分度
  - 新方案：让AI使用-10到10的整数评分，系统自动线性映射到[0,1]
  - 评分规则：
    * -10到-1：明显偏向低端（如"冷"、"低湿度"）→ 映射到 0.0-0.45
    * 0：中性或不确定 → 映射到 0.5
    * 1到10：明显偏向高端（如"热"、"高湿度"）→ 映射到 0.55-1.0
  - 优势：评分范围明确，区分度高，AI理解更准确

依赖：
  pip install openai tqdm pillow python-dotenv
  注意：虽然安装的是openai包，但实际调用的是Qwen API
  原因：Qwen API完全兼容OpenAI API格式，可以使用OpenAI SDK直接调用
  这是Qwen官方推荐的做法，无需安装额外的SDK
  
  如需使用本地embedding模型：
  pip install transformers torch

环境变量：
  QWEN_API_KEY=你的Qwen API密钥
可选：
  QWEN_BASE_URL=自定义网关/代理（如有，默认使用 https://dashscope.aliyuncs.com/compatible-mode/v1）

示例：
  # 使用API embedding模型（默认）
  python API.py \
    --image-dir ./images \
    --out ./dataset.csv \
    --vision-model qwen-vl-max \
    --embed-model text-embedding-v2 \
    --axis1 "温度：冷(0)↔热(1)" \
    --axis2 "湿度：低(0)↔高(1)" \
    --y-range 0-1
  
  # 使用本地Qwen3-Embedding-4B模型
  python API.py \
    --image-dir ./images \
    --out ./dataset.csv \
    --vision-model qwen-vl-max \
    --embed-model d:/llm/huggingface/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
    --use-local-embedding \
    --axis1 "温度：冷(0)↔热(1)" \
    --axis2 "湿度：低(0)↔高(1)"
  

关于维度：

# 查看实际维度
python API.py --image-dir test_images --out test.csv

# 验证维度（如果期望2560维）
python API.py \
  --image-dir test_images \
  --out test.csv \
  --use-local-embedding \
  --embed-model d:/llm/huggingface/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
  --embedding-dim 2560

备注：
  - 默认会写出表头（header）。如需去掉表头，增加 --no-header。
  - 使用 --save-descriptions 时，JSON文件会包含：
    * description: 图像描述
    * y: 归一化后的坐标 [y1, y2]
    * raw_scores: AI的原始评分 [score1, score2]
"""

import os
import io
import sys
import csv
import json
import math
import base64
import argparse
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image
from tqdm import tqdm

# 支持使用transformers库加载本地Qwen embedding模型
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 支持从.env文件加载环境变量
try:
    from dotenv import load_dotenv
    # 明确指定.env文件路径（在脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    # 加载.env文件（如果存在）
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # 如果.env不存在，尝试在当前目录查找
        load_dotenv()
except ImportError:
    # 如果没有安装python-dotenv，跳过（仍可使用系统环境变量）
    pass

# 使用OpenAI SDK调用Qwen API
# 说明：Qwen API完全兼容OpenAI API格式，所以可以直接使用OpenAI的Python SDK
# 只需要修改base_url为Qwen的端点即可，这是Qwen官方推荐的做法
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def b64_of_image(path: str) -> Tuple[str, str]:
    """读取图片并返回 (mime_subtype, base64字符串)。"""
    with Image.open(path) as im:
        # 统一转成RGB，避免某些格式带alpha出问题
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        buf = io.BytesIO()
        # 用 JPEG 压缩一下，减小传输体积
        im.save(buf, format="JPEG", quality=90)
        mime_subtype = "jpeg"
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return mime_subtype, b64


def build_openai_client() -> Any:
    """
    构建Qwen API客户端
    使用OpenAI SDK是因为Qwen API完全兼容OpenAI API格式
    只需要将base_url设置为Qwen的端点即可
    """
    if OpenAI is None:
        raise RuntimeError("未安装 openai SDK，请先 `pip install openai`.")
    # Qwen API 端点（完全兼容OpenAI API格式）
    # 默认使用阿里云DashScope的兼容模式端点
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 QWEN_API_KEY")
    # 使用OpenAI SDK，但连接到Qwen的API端点
    return OpenAI(base_url=base_url, api_key=api_key)


def normalize_score(score: float, target_range: str) -> float:
    """
    将评分归一化到目标范围
    假设输入评分范围为[-10, 10]，线性映射到目标范围
    
    映射规则：
    -10 → 0.0 (最低端)
      0 → 0.5 (中性)
     10 → 1.0 (最高端)
    """
    # 线性映射：将[-10, 10]映射到[0, 1]
    # 公式：(score + 10) / 20
    normalized = (score + 10.0) / 20.0
    
    # 夹紧到有效范围
    normalized = max(0.0, min(1.0, normalized))
    
    if target_range == "0-1":
        return normalized
    else:  # -1-1
        # 从[0,1]映射到[-1,1]
        return max(-1.0, min(1.0, normalized * 2 - 1))


def llm_describe_and_position(
    client: Any,
    vision_model: str,
    image_path: str,
    axis1: str,
    axis2: str,
    y_range: str = "0-1",
    temperature: float = 0.0,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    对单张图片：
      - 生成中文描述 description（≤ 60字）
      - 让AI给出两个维度的原始评分（无范围限制）
      - 使用sigmoid归一化到目标范围 y=[y1,y2]
    
    返回：{
        "description": str, 
        "y": [float, float],
        "raw_scores": [float, float]  # 原始评分
    }
    """
    # 读图转 base64
    mime_subtype, b64 = b64_of_image(image_path)

    # 使用改进的评分+归一化方案
    return _llm_describe_and_position_original(
        client, vision_model, mime_subtype, b64,
        axis1, axis2, y_range, temperature, max_retries
    )


def _llm_describe_and_position_original(
    client: Any,
    vision_model: str,
    mime_subtype: str,
    b64: str,
    axis1: str,
    axis2: str,
    y_range: str = "0-1",
    temperature: float = 0.0,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """改进方案：一次性询问描述和两个评分，然后归一化"""
    system_prompt = (
        "你是一个图像理解与标注助手。请保持输出结构化、可解析。"
    )
    
    user_text = (
        "请阅读这张图片并完成两个任务：\n"
        "1) 用简洁中文总结图像的核心内容，<=60字；\n"
        "2) 对图像在以下两个维度上进行评分（使用-10到10的整数评分）：\n"
        f"   维度1：{axis1}\n"
        f"   - 如果图像明显符合括号中左侧（0端）的特征，给负分（-10到-1）\n"
        f"   - 如果图像明显符合括号中右侧（1端）的特征，给正分（1到10）\n"
        f"   - 如果图像处于中间状态或不确定，给0分\n"
        f"   维度2：{axis2}\n"
        f"   - 如果图像明显符合括号中左侧（0端）的特征，给负分（-10到-1）\n"
        f"   - 如果图像明显符合括号中右侧（1端）的特征，给正分（1到10）\n"
        f"   - 如果图像处于中间状态或不确定，给0分\n"
        "   请大胆评分，充分利用-10到10的范围来体现图像的明显特征。\n"
        "严格只返回 JSON，格式为：{\"description\": \"...\", \"score1\": 整数, \"score2\": 整数}。"
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=vision_model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{mime_subtype};base64,{b64}",
                                    },
                                },
                            ],
                        },
                    ],
                )
            except Exception as e:
                error_str = str(e)
                if "image_url" in error_str or "unknown variant" in error_str:
                    raise RuntimeError(
                        f"模型 {vision_model} 不支持图片输入。\n"
                        f"请确认使用的是Qwen的视觉模型（如 qwen-vl-max, qwen-vl-plus 等）\n"
                        f"如果问题持续，请检查Qwen API文档确认模型名称"
                    ) from e
                raise
            content = resp.choices[0].message.content
            data = json.loads(content)
            desc = str(data.get("description", "")).strip()
            
            # 获取原始评分
            score1 = float(data.get("score1", 0))
            score2 = float(data.get("score2", 0))
            
            # 归一化到目标范围
            y1 = normalize_score(score1, y_range)
            y2 = normalize_score(score2, y_range)
            
            return {
                "description": desc, 
                "y": [y1, y2],
                "raw_scores": [score1, score2]  # 保存原始评分
            }
        except Exception as e:
            last_err = e
            import time
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"LLM 生成描述/坐标失败：{last_err}")


def embed_texts(
    client: Any,
    embed_model: str,
    texts: List[str],
    batch_size: int = 128,
    use_local: bool = False,
    embedding_dim: int = None,
) -> Tuple[List[List[float]], int]:
    """
    对 texts 批量做 embedding，返回向量列表（与输入顺序对应）和维度。
    
    Args:
        client: API客户端（如果use_local=True则可为None）
        embed_model: 模型名称（API模型名或本地模型路径/名称）
        texts: 文本列表
        batch_size: 批处理大小
        use_local: 是否使用本地模型
        embedding_dim: 期望的embedding维度（可选，用于验证）
    
    Returns:
        (embeddings, actual_dim): 向量列表和实际维度
    """
    if use_local:
        # 使用本地embedding模型（Qwen3-Embedding）
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "使用本地embedding模型需要transformers库，请先安装：\n"
                "  pip install transformers torch"
            )
        
        print(f"加载本地embedding模型: {embed_model}")
        print("  使用transformers库加载...")
        
        model_path = embed_model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        
        # 设置为评估模式
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("  使用GPU加速")
        else:
            print("  使用CPU（建议使用GPU以获得更好性能）")
        
        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches (本地)"):
            chunk = texts[i : i + batch_size]
            
            # Qwen embedding模型的编码方式
            with torch.no_grad():
                inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model(**inputs)
                # 使用mean pooling获取句子embedding
                embeddings_batch = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(embeddings_batch.tolist())
        
        # 获取实际维度
        actual_dim = len(embeddings[0]) if embeddings else 0
        
        # 验证维度（如果指定了期望维度）
        if embedding_dim is not None and actual_dim != embedding_dim:
            print(f"警告：期望维度 {embedding_dim}，但实际维度为 {actual_dim}")
        
        print(f"  Embedding维度: {actual_dim}")
        return embeddings, actual_dim
    else:
        # 使用API进行embedding
        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches (API)"):
            chunk = texts[i : i + batch_size]
            resp = client.embeddings.create(model=embed_model, input=chunk)
            # OpenAI embeddings 返回的 data 已按输入顺序排列
            chunk_vecs = [d.embedding for d in resp.data]
            embeddings.extend(chunk_vecs)
        
        # 获取实际维度
        actual_dim = len(embeddings[0]) if embeddings else 0
        
        # 验证维度（如果指定了期望维度）
        if embedding_dim is not None and actual_dim != embedding_dim:
            print(f"警告：期望维度 {embedding_dim}，但实际维度为 {actual_dim}")
        
        print(f"  Embedding维度: {actual_dim}")
        return embeddings, actual_dim


def collect_image_files(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    files: List[str] = []
    for root, _, fnames in os.walk(image_dir):
        for fn in fnames:
            if os.path.splitext(fn.lower())[1] in exts:
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def write_csv(
    out_path: str,
    x_vectors: List[List[float]],
    y_vectors: List[Tuple[float, float]],
    filenames: List[str],
    write_header: bool = True,
):
    if not (len(x_vectors) == len(y_vectors) == len(filenames)):
        raise ValueError("长度不一致：x/y/filename")
    if not x_vectors:
        raise ValueError("没有数据可写出")

    dim = len(x_vectors[0])
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            header = [f"x{i+1}" for i in range(dim)] + ["y1", "y2", "filename"]
            writer.writerow(header)
        for xs, (y1, y2), fn in zip(x_vectors, y_vectors, filenames):
            row = list(map(lambda v: f"{v:.7f}", xs)) + [f"{y1:.7f}", f"{y2:.7f}", fn]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="批量：图像→(LLM描述, LLM二维向量)→文本embedding→CSV"
    )
    parser.add_argument("--image-dir", required=True, help="输入图片目录")
    parser.add_argument("--out", required=True, help="输出 CSV 路径")
    parser.add_argument(
        "--vision-model",
        default="qwen-vl-max",
        help="多模态模型（可看图，Qwen视觉模型，如 qwen-vl-max, qwen3-vl-plus）",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-v2",
        help="文本向量模型（API模型名如 text-embedding-v2，或本地模型路径如 d:/llm/huggingface/models--Qwen--Qwen3-Embedding-4B/snapshots/...）",
    )
    parser.add_argument(
        "--use-local-embedding",
        action="store_true",
        help="使用本地embedding模型（需要安装transformers和torch）",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="期望的embedding维度（可选，用于验证，不指定则使用模型默认维度）",
    )
    parser.add_argument(
        "--axis1",
        default="温度：冷(0)↔热(1)",
        help="二维坐标轴1定义（影响 y1 的含义）",
    )
    parser.add_argument(
        "--axis2",
        default="湿度：低(0)↔高(1)",
        help="二维坐标轴2定义（影响 y2 的含义）",
    )
    parser.add_argument(
        "--y-range",
        choices=["0-1", "-1-1"],
        default="0-1",
        help="y坐标范围：0-1（更直观）或 -1-1（对称，默认0-1）",
    )
    parser.add_argument("--no-header", action="store_true", help="不写表头")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 张图（调试用）",
    )
    parser.add_argument(
        "--save-descriptions",
        type=str,
        default=None,
        help="保存LLM生成的描述到指定文件（JSON格式）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示每张图片的LLM描述和坐标（调试用）",
    )

    args = parser.parse_args()
    
    if not os.getenv("QWEN_API_KEY"):
        raise SystemExit("请先在环境变量里设置 QWEN_API_KEY")

    client = build_openai_client()

    files = collect_image_files(args.image_dir)
    if not files:
        raise SystemExit("未在目录中发现图片文件")
    if args.limit is not None:
        files = files[: args.limit]

    # 1) 用 LLM 逐张生成（描述, y）
    records: Dict[str, Dict[str, Any]] = {}
    descriptions: List[str] = []
    y_list: List[Tuple[float, float]] = []

    print("Step 1/3: LLM 生成描述 + 2D向量 ...")
    print(f"使用y坐标范围: {args.y_range}")
    print("使用改进的评分方案：AI给出原始评分 → sigmoid归一化")
    for fp in tqdm(files, desc="Describe & position"):
        result = llm_describe_and_position(
            client, args.vision_model, fp, args.axis1, args.axis2, args.y_range
        )
        records[fp] = result
        descriptions.append(result["description"])  # 用于 embedding
        y_list.append(tuple(result["y"]))
        
        # 如果启用verbose模式，显示每张图片的描述
        if args.verbose:
            filename = os.path.basename(fp)
            print(f"\n[{filename}]")
            print(f"  描述: {result['description']}")
            print(f"  原始评分: score1={result['raw_scores'][0]:.3f}, score2={result['raw_scores'][1]:.3f}")
            print(f"  归一化坐标: y1={result['y'][0]:.3f}, y2={result['y'][1]:.3f}")

    # 2) 批量文本 embedding（可一次性送入数组）
    print("Step 2/3: 文本 Embedding（批处理）...")
    if args.use_local_embedding:
        print(f"使用本地embedding模型: {args.embed_model}")
        x_vectors, embedding_dim = embed_texts(
            None, args.embed_model, descriptions, 
            use_local=True, embedding_dim=args.embedding_dim
        )
    else:
        print(f"使用API embedding模型: {args.embed_model}")
        x_vectors, embedding_dim = embed_texts(
            client, args.embed_model, descriptions, 
            use_local=False, embedding_dim=args.embedding_dim
        )
    
    print(f"确认：Embedding维度为 {embedding_dim}")

    # 3) 写出 CSV
    print("Step 3/3: 写出 CSV ...")
    write_csv(
        args.out,
        x_vectors=x_vectors,
        y_vectors=y_list,
        filenames=files,
        write_header=(not args.no_header),
    )
    
    # 4) 如果指定了保存描述，写出描述文件
    if args.save_descriptions:
        print(f"保存LLM描述到: {args.save_descriptions}")
        with open(args.save_descriptions, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已保存 {len(records)} 条描述")

    dim = len(x_vectors[0]) if x_vectors else 0
    print(
        f"\n完成：{len(files)} 条样本 → {args.out}\n"
        f"X 向量维度：{dim}；Y 为二维；已按行输出：x1..x{dim},y1,y2,filename"
    )


if __name__ == "__main__":
    main()
