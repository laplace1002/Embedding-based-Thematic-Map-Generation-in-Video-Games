## 项目简介

本项目是一个「图像 → 文本描述 & 情绪/属性坐标 → 向量」的数据构建与处理流水线，用于：

- **从 Kaggle 数据集批量下载图片**
- **用多模态 LLM（如 `qwen-vl-max`）为图片生成中文描述和两个语义维度坐标 `y1, y2`**
- **用文本向量模型（本地embedding或 API）对描述做 embedding，得到特征向量 `x1...xN`**
- **输出统一格式的 CSV：`x1,...,xN,y1,y2,filename`，并提供分割工具与示例模型**

目录中还包含简单的本地 Embedding Demo (`00_input_output.py`) 与训练后模型权重 `best_model.pth` 等。

---

## 环境与依赖

- **Python 版本**：建议 Python 3.8+
- **基础依赖**：
  - `openai`（用 OpenAI SDK 调 Qwen API）
  - `tqdm`
  - `pillow`
  - `python-dotenv`
  - `kagglehub`
  - `pandas`
- **本地 Embedding 相关**：
  - `sentence-transformers`
  - `transformers`
  - `torch`
  - `numpy`

安装示例（可按需增删）：

```bash
pip install openai tqdm pillow python-dotenv kagglehub pandas sentence-transformers transformers torch numpy
```

### 环境变量

- **Qwen API 密钥**
  - `QWEN_API_KEY`：必须，Qwen 的 API Key
  - `QWEN_BASE_URL`：可选，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Kaggle 认证**
  - 使用 `kagglehub` 时，需要在本机完成 Kaggle 认证（`~/.kaggle/kaggle.json` 或环境变量 `KAGGLE_USERNAME` / `KAGGLE_KEY`）

也支持在项目根目录放置 `.env` 文件（`API.py`, `process_kaggle_dataset.py` 中会自动加载）。

---

## 主要脚本说明

- **`API.py`**
  - 核心流水线脚本，完成：
    - 读取指定文件夹中的图片
    - 调用多模态 LLM 为图片生成简短中文描述 `description`
    - 让 LLM 对两个自定义语义维度（`axis1`, `axis2`）在 \[-10, 10] 间给出整数评分
    - 将评分线性归一化到目标范围（`0-1` 或 `-1-1`）得到 `y1, y2`
    - 调用 Embedding 模型（API 或本地 Qwen3-Embedding）将 `description` 转成向量 `x1...xN`
    - 写出 CSV：`x1,...,xN,y1,y2,filename`
  - 支持：
    - 使用 Qwen API 的文本向量模型（如 `text-embedding-v2`）
    - 使用本地 Qwen3-Embedding 模型（通过 `transformers`/`sentence-transformers`）
    - 保存描述 JSON（`--save-descriptions`）
    - 控制处理图片数量（`--limit`）

- **`download_kaggle_dataset.py`**
  - 用 `kagglehub` 下载 Kaggle 数据集 `naufalariqpyosyam/naturalassignment`
  - 递归收集图片，整理到统一输出目录（默认 `./kaggle_images`），文件名形如 `image_0001.jpg`
  - 支持：
    - `--limit`：只复制前 N 张（顺序）
    - `--random-sample`：从全部图片中随机抽取 N 张
    - `--skip-download` + `--dataset-path`：跳过下载，直接用指定路径
  - 会在输出目录里自动生成一个针对该数据集的 `README.md`（使用说明）。

- **`process_kaggle_dataset.py`**
  - 一键流水线脚本，串联 **下载 + 图像处理**：
    1. 检查依赖与环境（`kagglehub`、`openai`、`tqdm`、`pillow` 等）
    2. 检查 `QWEN_API_KEY` 与 Kaggle 认证
    3. 调用 `download_kaggle_dataset.py` 下载并整理图片
    4. 调用 `API.py` 对整理后的图片批量生成 CSV
  - 常用参数：
    - `--output-dir`：图片输出目录（默认 `./kaggle_images`）
    - `--output-csv`：最终 CSV 路径（默认 `./kaggle_dataset.csv`）
    - `--use-local`：使用本地 Embedding 模型
    - `--local-model-path`：本地模型路径
    - `--test`：测试模式，只处理少量图片
    - `--random-sample` / `--no-random-sample`：是否随机抽样
    - `--axis1`, `--axis2`：自定义两个语义维度
    - `--save-descriptions`：保存描述 JSON
    - `--verbose`：详细日志

- **`config.py`**
  - Qwen3-Embedding 模型配置：
    - `MODEL_SIZE = "4B" | "8B"`
    - 不同模型的路径、显存占用、MTEB 指标等
  - 提供：
    - `get_model_name()`：返回当前模型路径/名称
    - `load_model()`：封装 `SentenceTransformer` 加载逻辑，可选使用 ModelScope 镜像
    - 一些信息打印函数。

- **`00_input_output.py`**
  - 最简版 Embedding Demo：
    - 从 `config.get_model_name()` 加载模型
    - 交互式输入一段文本，输出：
      - 向量维度
      - 向量范数
      - 向量前 10 维
    - 同时将完整向量写入 `result.txt`（逗号分隔）。

- **`split_csv.py`**
  - 将一个包含 `x1...xN,y1,y2,filename` 的大 CSV 分成两个：
    - 文件 1：`x1` 到 `y2` 的所有列（特征 + 坐标）
    - 文件 2：`y1` 到 `filename` 的所有列（坐标 + 文件名）
  - 自动推断关键列的索引位置（要求列名里包含 `x1`, `y1`, `y2`, `filename`）。

- **`fit.py`**
  - 使用生成好的向量数据（默认 `dataset.csv`，前 2560 维为特征，后 2 维为目标 `y1, y2`）训练一个回归神经网络，并支持预测。
  - 主要流程：
    - 读取 CSV，将前 2560 维作为特征 `X`，后 2 维作为标签 `y`
    - 使用 `StandardScaler` 标准化特征
    - 使用 `PCA(n_components=0.95)` 降维（保留 95% 方差信息），显著降低维度
    - 构建两层全连接回归模型（带 Dropout 和 L2 正则），在训练集/验证集上训练并早停
    - 训练完成后保存：
      - `best_model.pth`：包含模型权重和 PCA 后的输入维度
      - `scaler.joblib`、`pca.joblib`：预处理器
      - `loss_curves.png`：训练/验证损失曲线
  - 预测模式下：
    - 从 `00_input_output.py` 生成的 `result.txt` 读取 2560 维向量
    - 依次应用 `StandardScaler`、`PCA`，再送入训练好的模型
    - 将预测结果保存到 `result1.txt`。

- **其他文件**
  - `dataset.csv` / `kaggle.csv` / `kaggle_x1_to_y2.csv` / `kaggle_y1_to_filename.csv`：项目中生成或使用的 CSV 数据文件
  - `best_model.pth`：下游模型（可能是基于上述向量训练的模型）保存的权重
  - `final/`：整理好的图片样本（如 `image_0001.jpg` 等），可直接用于 `API.py`。

---

## 快速开始

### 1. 准备环境

1. 安装依赖：

   ```bash
   pip install openai tqdm pillow python-dotenv kagglehub pandas sentence-transformers transformers torch numpy
   ```

2. 配置环境变量：
   - 在系统环境变量或项目根目录 `.env` 中设置：

   ```bash
   QWEN_API_KEY=你的_Qwen_API_密钥
   # 可选自定义网关
   QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

3. 完成 Kaggle 认证（用于 `kagglehub`）：
   - 将 `kaggle.json` 放到 `~/.kaggle/`，或设置：
     - `KAGGLE_USERNAME`
     - `KAGGLE_KEY`

### 2. 一键处理 Kaggle 数据集

在 `group_task` 目录下运行：

```bash
python process_kaggle_dataset.py --use-local --local-model-path d:/xxxx --output-dir ./final --output-csv ./kaggle.csv --axis1 "温度：冷(0)↔热(1)" --axis2 "湿度：干(0)↔湿(1)" --save-descriptions
```

默认行为：

- 下载 `naufalariqpyosyam/naturalassignment` 数据集
- 抽样一定数量的图片到 `./kaggle_images`
- 使用 Qwen API 模型处理图片，生成描述和坐标
- 输出 `./kaggle_dataset.csv`

常见变体：

```bash
# 使用本地 Qwen3-Embedding 模型
python process_kaggle_dataset.py --use-local --local-model-path d:/llm/huggingface/models--Qwen--Qwen3-Embedding-4B/snapshots/xxxxxxxx

# 测试模式，只处理前 10 张图片
python process_kaggle_dataset.py --test

# 自定义随机抽样数量
python process_kaggle_dataset.py --random-sample 500

# 使用所有图片（不进行随机抽样）
python process_kaggle_dataset.py --no-random-sample

# 自定义语义坐标轴
python process_kaggle_dataset.py \
  --axis1 "温度：冷(0)↔热(1)" \
  --axis2 "湿度：干燥(0)↔潮湿(1)"
```

### 3. 直接使用 API.py 处理任意图片文件夹

假设图片在 `./images` 目录：

```bash
# 使用 Qwen API 的 embedding 模型
python API.py \
  --image-dir ./images \
  --out ./dataset.csv \
  --vision-model qwen-vl-max \
  --embed-model text-embedding-v2 \
  --axis1 "温度：冷(0)↔热(1)" \
  --axis2 "湿度：低(0)↔高(1)" \
  --y-range 0-1

# 使用本地 Qwen3-Embedding 模型
python API.py \
  --image-dir ./images \
  --out ./dataset.csv \
  --vision-model qwen-vl-max \
  --embed-model d:/xxxxxxxx \
  --use-local-embedding \
  --axis1 "温度：冷(0)↔热(1)" \
  --axis2 "湿度：低(0)↔高(1)"
```

如需检查实际向量维度，可使用 `--embedding-dim` 参数校验。

### 4. 查看和分割生成的 CSV

假设生成了 `kaggle_dataset.csv`：

```bash
# 将 CSV 分割成两份：x1..y2 和 y1..filename
python split_csv.py kaggle_dataset.csv
```

分割后通常会得到：

- `kaggle_dataset_x1_to_y2.csv`
- `kaggle_dataset_y1_to_filename.csv`

---

## 本地 Embedding Demo

执行：

```bash
python 00_input_output.py
```

功能：

- 加载 `config.py` 中配置的 Qwen3-Embedding 模型
- 交互式输入文本，输出向量统计信息
- 将完整向量写入 `result.txt`，便于后续使用fit.py。

---

## 注意事项

- **显存与模型大小**：在 `config.py` 中可切换 4B / 8B 模型，需保证显存&内存充足。
- **速率与费用**：使用 Qwen API 时，请注意 QPS 限制和调用费用，可用 `--limit` 或随机抽样控制图片数量。
- **数据隐私**：如涉及敏感图片/文本，请确认符合所在环境的合规要求。

# 图像描述与二维坐标生成系统

一个基于多模态LLM的图像处理pipeline，能够为图像生成中文描述和二维语义坐标，并输出embedding向量用于机器学习任务。

## ✨ 核心特性

- 🖼️ **多模态理解**：使用Qwen视觉模型分析图像内容
- 📝 **智能描述**：生成简洁的中文图像描述（≤60字）
- 📊 **语义坐标**：在自定义的二维空间中定位图像（改进的评分+归一化方案）
- 🔢 **向量化**：生成高质量的文本embedding（支持API和本地模型）
- 📂 **批量处理**：高效处理大规模图像数据集
- 🎯 **Kaggle集成**：一键处理Kaggle数据集

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install openai tqdm pillow python-dotenv kagglehub
```

如果使用本地embedding模型：
```bash
pip install transformers torch
```

### 2. 配置API密钥

创建 `.env` 文件：
```bash
QWEN_API_KEY=你的Qwen_API密钥
```

详细配置：查看 [API_KEY_SETUP.md](API_KEY_SETUP.md)

### 3. 运行示例

#### 测试脚本（验证环境）
```bash
python test_env.py
```

#### 处理本地图片
```bash
python API.py \
  --image-dir ./test_images \
  --out ./output.csv \
  --axis1 "情感：消极(0)↔积极(1)" \
  --axis2 "活力：低(0)↔高(1)"
```

#### 处理Kaggle数据集（一键运行）
```bash
python process_kaggle_dataset.py --test
```

## 📖 文档导航

### 快速入门
- 📘 [快速启动 - Kaggle数据集](QUICK_START_KAGGLE.md) - 最快上手方式
- 📗 [API密钥设置指南](API_KEY_SETUP.md) - 配置Qwen API
- 📙 [测试数据集指南](TEST_DATASET_GUIDE.md) - 使用test_images测试

### 深入指南
- 📕 [Kaggle数据集完整工作流](KAGGLE_DATASET_WORKFLOW.md) - 详细的Kaggle集成说明
- 📔 [Embedding维度说明](EMBEDDING_DIMENSION.md) - 向量维度和模型选择
- 📓 [描述JSON格式说明](DESCRIPTIONS_JSON_FORMAT.md) - 输出格式和原始评分

### 脚本说明
- 🔧 `API.py` - 主处理脚本（支持批量处理）
- 🔧 `download_kaggle_dataset.py` - 下载并整理Kaggle数据
- 🔧 `process_kaggle_dataset.py` - 一键处理Kaggle数据集
- 🔧 `test_env.py` - 环境检测脚本

## 🎯 使用场景

### 场景1：处理Kaggle数据集（推荐）

**最简单方式：**
```bash
# 测试模式（10张图）
python process_kaggle_dataset.py --test

# 完整处理
python process_kaggle_dataset.py
```

**详见：** [QUICK_START_KAGGLE.md](QUICK_START_KAGGLE.md)

### 场景2：处理本地图片

**基础用法：**
```bash
python API.py \
  --image-dir ./your_images \
  --out ./dataset.csv \
  --axis1 "温度：冷(0)↔热(1)" \
  --axis2 "湿度：干(0)↔湿(1)"
```

**使用本地embedding模型：**
```bash
python API.py \
  --image-dir ./your_images \
  --out ./dataset.csv \
  --use-local-embedding \
  --embed-model /path/to/Qwen3-Embedding-4B \
  --axis1 "温度：冷(0)↔热(1)" \
  --axis2 "湿度：干(0)↔湿(1)"
```

**保存描述并显示详细信息：**
```bash
python API.py \
  --image-dir ./your_images \
  --out ./dataset.csv \
  --save-descriptions ./descriptions.json \
  --verbose
```

### 场景3：自定义语义空间

你可以根据任务需求自定义坐标轴：

**情感分析：**
```bash
--axis1 "情感倾向：消极(0)↔积极(1)" \
--axis2 "情感强度：弱(0)↔强(1)"
```

**自然场景：**
```bash
--axis1 "自然度：人造(0)↔自然(1)" \
--axis2 "生物多样性：无生物(0)↔丰富(1)"
```

**艺术风格：**
```bash
--axis1 "写实度：抽象(0)↔写实(1)" \
--axis2 "色彩：冷色调(0)↔暖色调(1)"
```

## 🔬 工作原理

### 改进的评分方案

**旧方案问题：** 要求AI直接输出0-1范围的坐标，容易导致评分不准确

**新方案优势：**
1. AI使用-10到10的整数评分（明确的评分范围）
2. 系统自动线性映射到[0,1]或[-1,1]
3. 区分度高，语义明确，提高准确度

**归一化函数：**
```python
normalized = (score + 10.0) / 20.0  # 线性映射
# -10 → 0.0, 0 → 0.5, 10 → 1.0
```

### 处理流程

```
图像文件 → 【多模态LLM】 → 描述 + 原始评分
                          ↓
                    sigmoid归一化
                          ↓
     描述 → 【文本Embedding】 → 向量 (x1...xN)
                          ↓
          CSV输出: x1,...,xN, y1, y2, filename
```

## 📊 输出格式

### CSV文件
```csv
x1,x2,...,xN,y1,y2,filename
0.123,0.456,...,0.789,0.65,0.72,image_0001.jpg
...
```

- **x1-xN**: embedding向量（1024维或2560维）
- **y1, y2**: 归一化后的二维坐标（0-1或-1-1范围）
- **filename**: 原始图片文件名

### 描述JSON（可选）
```json
{
  "image_0001.jpg": {
    "description": "阳光明媚的海滩，人们在玩耍",
    "y": [0.65, 0.72],
    "raw_scores": [8.5, 12.3]
  }
}
```

- **description**: LLM生成的图像描述
- **y**: 归一化后的二维坐标
- **raw_scores**: AI的原始评分（归一化前）

详见：[DESCRIPTIONS_JSON_FORMAT.md](DESCRIPTIONS_JSON_FORMAT.md)

## 🔧 高级选项

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image-dir` | 图片目录 | 必填 |
| `--out` | 输出CSV路径 | 必填 |
| `--vision-model` | 视觉模型 | qwen-vl-max |
| `--embed-model` | Embedding模型 | text-embedding-v2 |
| `--use-local-embedding` | 使用本地模型 | False |
| `--axis1` | 坐标轴1定义 | 温度：冷(0)↔热(1) |
| `--axis2` | 坐标轴2定义 | 湿度：低(0)↔高(1) |
| `--y-range` | y坐标范围 | 0-1 |
| `--save-descriptions` | 保存描述JSON | None |
| `--verbose` | 显示详细信息 | False |
| `--limit` | 只处理前N张 | None |
| `--no-header` | 不写CSV表头 | False |

### 性能优化

**API模式（快速）：**
- 约 6-10秒/图片
- 成本：约 ￥0.01-0.02/图片
- 适合中小规模数据集

**本地模式（经济）：**
- 约 10-15秒/图片（GPU）
- 成本：仅电费
- 适合大规模数据集

## 📈 典型处理时间

| 数据量 | API模式 | 本地模式 | 成本估算 |
|--------|---------|----------|---------|
| 10张（测试） | ~2分钟 | ~3分钟 | ￥0.1-0.2 |
| 100张 | ~15分钟 | ~25分钟 | ￥1-2 |
| 1000张 | ~2.5小时 | ~4小时 | ￥10-20 |

*注：实际时间取决于网络、硬件和API响应速度*

## 🐛 故障排查

### 常见错误

**1. QWEN_API_KEY未找到**
```bash
解决：查看 API_KEY_SETUP.md 配置API密钥
```

**2. Kaggle认证失败**
```bash
解决：查看 KAGGLE_DATASET_WORKFLOW.md 配置Kaggle认证
```

**3. 模型不支持图片输入**
```bash
解决：确认使用 qwen-vl-max 或 qwen-vl-plus
```

**4. 内存不足**
```bash
解决：使用 --limit 参数减少批量大小
```

### 获取帮助

```bash
# 查看所有参数
python API.py --help

# 查看环境状态
python test_env.py

# 查看详细文档
cat KAGGLE_DATASET_WORKFLOW.md
```

## 📝 示例应用

### 数据分析
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('kaggle_dataset.csv')

# 可视化坐标分布
plt.scatter(df['y1'], df['y2'], alpha=0.5)
plt.xlabel('维度1')
plt.ylabel('维度2')
plt.show()
```

### 机器学习
```python
import numpy as np
from sklearn.model_selection import train_test_split

# 准备特征和标签
X = df.iloc[:, :-3].values  # embedding向量
y = df[['y1', 'y2']].values  # 二维坐标

# 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# ... 你的模型代码
```

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可

MIT License

## 📞 联系

如有问题，请查看相关文档或提交Issue。

---

**最后更新：** 2024-11

**核心改进：** 采用原始评分+sigmoid归一化方案，提高坐标准确度

