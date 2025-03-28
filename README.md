# SimCSE-PyTorch中文医疗文本标准化

本项目是SimCSE（Simple Contrastive Learning of Sentence Embeddings，简单对比学习句子嵌入）的PyTorch实现，专注于中文医疗文本标准化。通过监督对比学习方法，实现医疗术语和症状表述的标准化。

## 项目介绍

### 背景与问题

在医疗领域，术语标准化是一个至关重要的问题。同一种疾病、症状或医疗概念可能有多种不同的表述方式，例如：
- 患者可能使用"头疼"、"头痛"、"脑壳痛"等方式描述同一症状
- 不同医生可能使用不同的表述记录同一疾病
- 医疗记录中可能混合使用正式术语和俗语表达

这种表述的多样性给医疗信息处理带来了巨大挑战，影响了：
- 医疗数据的结构化和统计分析
- 智能诊断系统的准确性
- 医疗记录的互操作性
- 临床决策支持系统的有效性

本项目旨在解决这一问题，将各种非标准的医疗文本表述映射到标准化的医学术语，提高医疗信息处理的准确性和一致性。

### 解决方案

本项目采用监督对比学习方法，基于SimCSE框架，训练模型学习将各种医疗表述映射到统一的语义空间，实现医疗文本的标准化。具体而言：
- 利用大规模中文医疗文本语料作为训练数据
- 采用监督对比学习方法，使模型能够区分相似和不相似的医疗表述
- 将相同医疗概念的不同表述在语义空间中映射到接近的位置
- 结合BM25检索和语义相似度匹配，实现准确的医疗术语标准化

## 模型结构与原理

### SimCSE模型基本原理

SimCSE是一种简单但有效的对比学习方法，用于学习高质量的句子嵌入。本项目实现的是监督版本的SimCSE，主要包括：

1. **基础编码器**：使用预训练的BERT模型作为编码器基础，将输入文本映射到高维向量空间
2. **对比学习框架**：
   - 正样本对：原始医疗表述和对应的标准表述构成正样本对
   - 负样本对：原始医疗表述和不相关的标准表述构成负样本对
   - 对比损失函数：让模型学习将相似的表述映射到相近的向量表示，将不相似的表述映射到远离的向量表示

### 模型架构详细说明

本项目的模型架构主要由以下部分组成：

1. **基础编码器层**：
   - 使用BERT模型（默认使用bert-base-chinese）
   - 支持自定义预训练模型路径
   - 可配置的dropout率，用于防止过拟合

2. **特征提取策略**：
   - CLS Token提取：使用BERT的[CLS]token作为整句表示
   - 支持多种池化策略：
     - cls：使用[CLS]标记的隐藏状态
     - pooler：使用BERT的pooler输出
     - last-avg：使用最后一层隐藏状态的平均值
     - first-last-avg：使用第一层和最后一层隐藏状态的平均值

3. **对比学习损失函数**：
   - 使用修改版的InfoNCE损失函数
   - 温度系数设置为0.05，用于控制相似度分布的平滑程度
   - 批处理中使用成三元组（原始表述、标准表述、无关表述）进行训练

### 训练与推理流程

1. **训练流程**：
   - 加载医疗术语数据，构建三元组（原始表述、标准表述、无关表述）
   - 使用BERT tokenizer对文本进行预处理
   - 通过SimCSE监督对比学习损失函数训练模型
   - 使用Spearman相关系数评估模型性能
   - 保存性能最佳的模型

2. **推理流程**：
   - 使用训练好的模型提取医疗文本的语义向量
   - 结合BM25检索，缩小候选标准术语范围
   - 计算原始表述与候选标准术语的余弦相似度
   - 选择相似度最高的标准术语作为映射结果

## 主要特性

- 基于监督SimCSE框架的语义文本嵌入实现
- 集成BM25检索用于候选标准术语选择
- 使用余弦相似度计算找到最相似的标准术语
- 专门针对中文医疗术语标准化的优化
- 完整的模型评估工具

## 环境要求

- Python 3.6+
- PyTorch
- Transformers
- pandas
- fastbm25
- jsonlines
- loguru
- tqdm
- scipy

## 安装方法

克隆仓库并安装所需的依赖包：

```bash
git clone https://github.com/yourusername/simcse-pytorch.git
cd simcse-pytorch
pip install -r requirements.txt
```

## 数据准备

项目使用的数据格式要求：

1. 训练数据：JSONL文件，包含"origin"、"entailment"和"contradiction"字段
2. 开发数据：文本文件，每行包含三元组(原始术语||标准术语||得分)
3. 标准术语字典：TSV文件，包含code和name列

## 使用方法

### 训练模型

运行以下命令训练模型：

```bash
python train.py --device cuda --num_epoch 10 --batch_size 64 --model_path bert-base-chinese --save_path sup_saved.pt
```

参数说明：
- `--device`：训练设备（cpu, cuda）
- `--num_epoch`：训练轮数
- `--batch_size`：批处理大小
- `--max_len`：最大序列长度
- `--model_path`：预训练BERT模型路径
- `--save_path`：模型保存路径
- `--lr`：学习率
- `--dropout`：Dropout率
- `--toy`：是否使用小数据子集进行测试

### 预测

使用以下命令进行预测：

```bash
python predict.py
```

您可能需要在脚本中调整模型路径和其他参数。

### 评估

评估BM25性能：

```bash
python -m utils.bm25_eval
```

## 项目结构

- `train.py`：模型训练脚本
- `predict.py`：预测脚本
- `src/`：核心实现
  - `model.py`：SimCSE模型定义
  - `trainer.py`：训练循环实现
  - `dataset.py`：数据集和数据加载工具
- `utils/`：工具函数
  - `bm25_eval.py`：BM25评估脚本
  - `data_process.py`：数据处理工具
- `data/`：数据存储目录

## 许可证

MIT License

Copyright (c) 2023 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

