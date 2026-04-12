# 社交情感追踪器 · Social Sentiment Tracker

![CI](https://github.com/lvzhuojun/social-sentiment-tracker/actions/workflows/ci.yml/badge.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://social-sentiment-tracker.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)

> 一个端到端的 NLP 情感分析平台。从原始社交媒体文本出发，经过数据清洗与预处理，训练两种互补模型——快速的 TF-IDF + 逻辑回归基线模型与经过微调的 **BERT**（`bert-base-uncased`）——并通过交互式四页 **Streamlit** Web 演示和 **FastAPI** REST 接口提供预测服务。使用 **TweetEval** 基准数据集（59,899 条真实推文，三分类情感），并在数据文件缺失时自动生成模拟数据用于离线开发。

![Streamlit Demo](reports/figures/screenshot_home.png)

**Language / 语言：** [English](README.md) · [中文](#)

---

## 目录

- [主要功能](#主要功能)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [安装指南](#安装指南)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [模型性能](#模型性能)
- [Notebook 说明](#notebook-说明)
- [API 概览](#api-概览)
- [界面截图](#界面截图)
- [未来计划](#未来计划)
- [文档维护标准](#文档维护标准)
- [更新日志](#更新日志)
- [作者](#作者)

---

## 主要功能

| # | 功能 | 说明 |
|---|------|------|
| 1 | **双模型流水线** | TF-IDF + 逻辑回归（CPU 秒级训练）与 BERT 微调（RTX 5060 约 90 分钟） |
| 2 | **TweetEval 基准数据** | 59,899 条真实推文，三分类标签；`download_data.py` 自动从 HuggingFace 下载 |
| 3 | **FastAPI REST 接口** | `api/serve.py` — `/predict`、`/predict/batch`、`/health`；Pydantic 校验；uvicorn 就绪 |
| 4 | **SHAP 可解释性** | `src/explain.py` — 基线模型 SHAP `LinearExplainer`；Streamlit 内 token 级归因展示 |
| 5 | **超参数调优** | `scripts/tune_baseline.py` — 3 折 GridSearchCV 覆盖 TF-IDF 与 LR 参数，附热力图 |
| 6 | **错误分析 Notebook** | `04_error_analysis.ipynb` — 高置信错误、否定词影响、类别难度、错误 SHAP 分析 |
| 7 | **可复现实验** | `set_seed(42)` 在 `config.py` 中全局固定 `random`、`numpy`、`torch` 随机种子 |
| 8 | **完整工程栈** | 121 个 pytest 测试 · GitHub Actions CI · Docker · Streamlit Cloud · Google 文档风格 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                           原始文本输入                                │
│             Sentiment140 CSV  /  自动生成的模拟数据                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   data_loader.py    │
                    │  load_sentiment140()│
                    │  clean_text()       │
                    │  preprocess_df()    │
                    │  split_data()       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
   ┌──────────▼──────────┐           ┌──────────▼──────────┐
   │   preprocess.py     │           │   preprocess.py     │
   │   tokenize()        │           │   add_text_features │
   │   remove_stopwords()│           │   word_count        │
   │   lemmatize()       │           │   char_count        │
   └──────────┬──────────┘           └──────────┬──────────┘
              │                                 │
   ┌──────────▼──────────┐           ┌──────────▼──────────┐
   │  baseline_model.py  │           │    bert_model.py    │
   │  TfidfVectorizer    │           │  SentimentDataset   │
   │  LogisticRegression │           │  SentimentClassifier│
   │  build_pipeline()   │           │  bert-base-uncased  │
   │  train_baseline()   │           │  Dropout(0.3)       │
   │  predict()          │           │  Linear Head        │
   └──────────┬──────────┘           │  train_bert()       │
              │                      │  predict_bert()     │
              │                      └──────────┬──────────┘
              └──────────────┬──────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │     evaluate.py     │
                  │  evaluate_model()   │
                  │  confusion_matrix() │
                  │  plot_roc_curve()   │
                  │  compare_models()   │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │    visualize.py     │
                  │  sentiment_dist()   │
                  │  text_length_dist() │
                  │  plot_wordcloud()   │
                  │  sentiment_time()   │
                  │  top_keywords()     │
                  │  confidence_gauge() │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ app/streamlit_app   │
                  │  页面1 · 首页        │
                  │  页面2 · 数据分析    │
                  │  页面3 · 实时预测    │
                  │  页面4 · 模型对比    │
                  └─────────────────────┘
```

---

## 技术栈

| 层次 | 技术 | 版本要求 |
|------|------|---------|
| 编程语言 | Python | 3.10 |
| 数据集 | Sentiment140 (Twitter) / 模拟 CSV | — |
| 机器学习基线 | scikit-learn · TF-IDF + 逻辑回归 | ≥ 1.3.0 |
| 深度学习框架 | PyTorch | ≥ 2.1.0 |
| 预训练模型 | HuggingFace `transformers`（`bert-base-uncased`） | ≥ 4.35.0 |
| NLP 工具 | NLTK（分词 · 停用词 · 词形还原） | ≥ 3.8.1 |
| 交互可视化 | Plotly | ≥ 5.18.0 |
| 静态可视化 | Matplotlib · Seaborn | ≥ 3.7.0 / 0.12.0 |
| 词云 | wordcloud | ≥ 1.9.2 |
| Web 演示 | Streamlit | ≥ 1.28.0 |
| 模型序列化 | joblib | ≥ 1.3.0 |
| 环境管理 | Conda（`sentiment-tracker`） | — |

---

## 安装指南

### 环境要求

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
- Git

### 克隆与配置

```bash
# 1. 克隆仓库
git clone https://github.com/lvzhuojun/social-sentiment-tracker.git
cd social-sentiment-tracker

# 2. 创建并激活 conda 环境（Python 3.10 + 全部依赖）
conda env create -f environment.yml
conda activate sentiment-tracker

# 3. （可选）使用真实数据集
#    从 https://www.kaggle.com/datasets/kazanova/sentiment140 下载
#    重命名并放置到：data/raw/twitter_training.csv
#    若文件不存在，首次运行时自动生成 500 条均衡模拟数据。

# 4. 启动 Streamlit Web 演示
streamlit run app/streamlit_app.py
# 在浏览器中访问 http://localhost:8501
```

### Streamlit Cloud（零安装公开演示）

1. Fork 本仓库
2. 打开 [share.streamlit.io](https://share.streamlit.io) → **New app**
3. 选择你的 Fork，分支 `main`，文件 `app/streamlit_app.py`
4. 将 **Python version** 设为 `3.10`，**Requirements file** 设为 `requirements-cloud.txt`
5. 点击 **Deploy** — 基线模型会在首次启动时自动训练（约 10 秒）

> 免费套餐内存限制，BERT 推理功能在云端不可用，其他页面完全正常。

---

### Docker（一键启动演示）

```bash
# 构建镜像（首次构建因下载 torch 约需 5 分钟）
docker build -t social-sentiment-tracker .

# 启动 Streamlit 演示
docker run -p 8501:8501 social-sentiment-tracker
# 在浏览器访问 http://localhost:8501
```

如需在容器内使用已训练好的模型，挂载 `models/` 目录：

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/models:/app/models" \
  social-sentiment-tracker
```

---

### GPU 支持（可选）

打开 `environment.yml`，进行以下修改：

```yaml
# 删除这一行：
- cpuonly
# 添加这一行（根据你的 CUDA 版本调整）：
- pytorch-cuda=12.1
```

然后重新创建环境：

```bash
conda env remove -n sentiment-tracker
conda env create -f environment.yml
```

所有训练超参数均在 `config.py` 中集中管理：

| 参数 | 默认值 | 配置键 |
|------|--------|--------|
| BERT 模型 | `bert-base-uncased` | `BERT_MODEL_NAME` |
| 最大序列长度 | 128 | `MAX_LENGTH` |
| 批次大小 | 16 | `BATCH_SIZE` |
| 训练轮数 | 3 | `EPOCHS` |
| 学习率 | 2e-5 | `LEARNING_RATE` |
| 预热比例 | 0.1 | `WARMUP_RATIO` |
| TF-IDF 最大特征数 | 50,000 | `TFIDF_MAX_FEATURES` |
| 随机种子 | 42 | `RANDOM_SEED` |

---

## 项目结构

```
social-sentiment-tracker/
│
├── app/
│   └── streamlit_app.py        # 四页 Web 演示（首页 · 数据分析 · 实时预测 · 模型对比）
│
├── data/
│   ├── raw/                    # 原始 CSV（已 git 忽略）；mock_data.csv 自动生成
│   └── processed/              # 清洗后的 DataFrame（运行时生成）
│
├── models/                     # 已训练模型文件（已 git 忽略）
│   ├── baseline_tfidf_lr.pkl   # sklearn Pipeline 序列化文件（joblib）
│   └── bert_sentiment.pt       # BERT 状态字典检查点（最优 val_acc）
│
├── notebooks/
│   ├── 01_eda.ipynb            # EDA：类别分布、文本统计、词云
│   ├── 02_baseline_model.ipynb # 训练 TF-IDF + LR、特征重要性、评估
│   └── 03_bert_finetune.ipynb  # BERT 微调、训练曲线、误差分析
│
├── reports/
│   └── figures/                # 自动保存的 PNG：混淆矩阵、ROC、词云
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # load_sentiment140 · clean_text · split_data · generate_mock_data
│   ├── preprocess.py           # tokenize · remove_stopwords · lemmatize · add_text_features
│   ├── baseline_model.py       # build_pipeline · train_baseline · predict · load_baseline_model
│   ├── bert_model.py           # SentimentDataset · SentimentClassifier · train_bert · predict_bert
│   ├── evaluate.py             # evaluate_model · plot_confusion_matrix · plot_roc_curve · compare_models
│   └── visualize.py            # plot_sentiment_distribution · plot_wordcloud · plot_top_keywords · …
│
├── config.py                   # 集中管理路径、超参数、set_seed()、get_logger()
├── environment.yml             # Conda 环境规格（Python 3.10，pytorch 频道）
├── requirements.txt            # pip 依赖列表（含版本约束）
│
├── README.md                   # 英文文档
├── README_CN.md                # 中文文档（本文件）
├── CHANGELOG.md                # 版本历史与更新记录
└── CONTRIBUTING.md             # 贡献指南与文档维护标准
```

---

## 使用方法

### 训练模型

```bash
conda activate sentiment-tracker

# ── 基线模型（TF-IDF + 逻辑回归）────────────────────────────────────────
# CPU 上约需 10 秒。
# 输出：models/baseline_tfidf_lr.pkl
python src/baseline_model.py

# ── BERT 微调 ─────────────────────────────────────────────────────────────
# 强烈建议使用 GPU（≥ 6 GB 显存）。
# 无 GPU 时自动回退到 CPU，每 epoch 约需 20–40 分钟。
# 输出：models/bert_sentiment.pt（保存最优 val_acc 检查点）
python src/bert_model.py
```

### 启动 Web 演示

```bash
conda activate sentiment-tracker
streamlit run app/streamlit_app.py
# 在浏览器中访问 http://localhost:8501
```

**各页面说明：**

| 页面 | 功能描述 |
|------|---------|
| **首页** | 数据集统计（总样本数、各类别数量）、情感分布饼图、架构总览 |
| **数据分析** | 文本长度直方图、词云（按情感类别切换）、TF-IDF 关键词柱状图、情感时间趋势图 |
| **实时预测** | 单条文本或批量输入；选择基线或 BERT 模型；输出情感标签 + 置信度仪表盘；批量结果支持下载 CSV |
| **模型对比** | 准确率 / 精确率 / 召回率 / F1 / AUC 对比表、柱状图、双模型 ROC 曲线对比及文字分析 |

### 运行 Notebook

```bash
conda activate sentiment-tracker
jupyter lab
# 按顺序打开 notebooks/ 目录下的文件：01 → 02 → 03
```

---

## 模型性能

在 **TweetEval 测试集**上的评估结果（12,264 条样本，`RANDOM_SEED=42`）。
三分类：0 = 负面 · 1 = 正面 · 2 = 中性。

| 指标 | 基线（TF-IDF + LR） | BERT（bert-base-uncased） |
|------|---------------------|--------------------------|
| 准确率 (Accuracy) | 0.5935 | **0.6889** |
| 精确率 (Precision, weighted) | 0.6096 | **0.6973** |
| 召回率 (Recall, weighted) | 0.5935 | **0.6889** |
| F1 分数 (weighted) | 0.5788 | **0.6873** |
| ROC-AUC（宏平均 OvR） | 0.7724 | **0.8660** |

> 数据集：来自 HuggingFace 的 `tweet_eval/sentiment`（45,615 训练 / 2,000 验证 / 12,284 测试）。
> BERT 在 RTX 5060 Laptop GPU 上微调 3 个 epoch（约 28.5 分钟）；第 2 个 epoch 为最优检查点（val_acc 0.7175）。
> 所有指标由 `src/evaluate.evaluate_model()` 计算；支持 OvR 多类别 ROC 曲线。

**基线模型各类别表现：**

| 类别 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|-------|
| 负面 (0) | 0.68 | 0.35 | 0.46 | 3,968 |
| 正面 (1) | 0.53 | 0.62 | 0.57 | 2,371 |
| 中性 (2) | 0.59 | 0.75 | 0.66 | 5,925 |

**BERT 各类别表现：**

| 类别 | 精确率 | 召回率 | F1 | 样本数 |
|------|--------|--------|-----|-------|
| 负面 (0) | 0.67 | 0.79 | 0.72 | 3,968 |
| 正面 (1) | 0.62 | 0.73 | 0.67 | 2,371 |
| 中性 (2) | 0.75 | 0.61 | 0.67 | 5,925 |

> **BERT 相比基线提升：准确率 +9.5pp、F1 +10.9pp、AUC +9.4pp。**
> 负面类召回率从 0.35 大幅提升至 0.79，得益于上下文表示的强大能力。
> 详细的失败模式分析请参见 `notebooks/04_error_analysis.ipynb`。

---

## Notebook 说明

| Notebook | 内容描述 | 主要输出 |
|----------|---------|---------|
| `01_eda.ipynb` | 数据加载、缺失值分析、类别分布、文本长度统计、高频词分析、TF-IDF 关键词、时间趋势 | Plotly 交互图表 · `reports/figures/wordcloud_*.png` |
| `02_baseline_model.ipynb` | 训练 TF-IDF + LR 流水线、测试集评估、LR 系数特征重要性、混淆矩阵、ROC 曲线 | `models/baseline_tfidf_lr.pkl` · ROC 曲线 PNG |
| `03_bert_finetune.ipynb` | BERT 微调、逐 epoch 训练曲线、测试集评估、与基线对比 | `models/bert_sentiment.pt` · 模型对比图 |
| `04_error_analysis.ipynb` | 高置信度错误分析、否定词影响、类别难度、错误 SHAP 分析、双模型分歧 | `reports/figures/error_*.png` · 洞察摘要 |

---

## API 概览

### `config.py`

| 符号 | 类型 | 说明 |
|------|------|------|
| `set_seed(seed=42)` | 函数 | 固定 `random` / `numpy` / `torch` 随机种子，保证实验可复现 |
| `get_logger(name)` | 函数 | 返回格式统一的 `logging.Logger` 实例 |
| `BERT_MODEL_NAME` | 常量 | `"bert-base-uncased"` |
| `MAX_LENGTH` | 常量 | 最大 token 序列长度（128） |
| `BATCH_SIZE` | 常量 | 训练批次大小（16） |
| `EPOCHS` | 常量 | 微调轮数（3） |
| `LEARNING_RATE` | 常量 | AdamW 学习率（2e-5） |

---

### `src/data_loader.py`

| 函数 | 签名 | 说明 |
|------|------|------|
| `load_data` | `load_data(real_path=None)` | 自动选择 Sentiment140 或模拟数据；返回预处理后的 DataFrame |
| `load_sentiment140` | `load_sentiment140(filepath)` | 读取原始 CSV；将标签 4 映射为 1（二分类） |
| `clean_text` | `clean_text(text: str) -> str` | 转小写；去除 URL、@提及、#话题标签、非字母字符 |
| `preprocess_dataframe` | `preprocess_dataframe(df)` | 应用 `clean_text`；删除空行和重复行 |
| `split_data` | `split_data(df, test_size=0.2, val_size=0.1)` | 分层 train / val / test 划分；返回三个 DataFrame |
| `generate_mock_data` | `generate_mock_data(n=500, save_path=None)` | 生成含日期列的均衡模拟情感 CSV |

---

### `src/preprocess.py`

| 函数 | 签名 | 说明 |
|------|------|------|
| `tokenize` | `tokenize(text: str) -> List[str]` | NLTK `word_tokenize`；不可用时回退为空格分词 |
| `remove_stopwords` | `remove_stopwords(tokens, language='english')` | 过滤 NLTK 英语停用词表 |
| `lemmatize` | `lemmatize(tokens: List[str]) -> List[str]` | 对 token 列表进行词形还原 |
| `add_text_features` | `add_text_features(df, text_col='clean_text')` | 新增 `word_count`、`char_count`、`avg_word_len`、`unique_word_ratio` 列 |

---

### `src/baseline_model.py`

| 函数 | 签名 | 说明 |
|------|------|------|
| `build_pipeline` | `build_pipeline() -> Pipeline` | 构建 TF-IDF（max_features=5万，ngram=(1,2)）→ 逻辑回归 sklearn Pipeline |
| `train_baseline` | `train_baseline(train_df, val_df) -> Pipeline` | 拟合模型，记录验证集指标，保存至 `models/baseline_tfidf_lr.pkl` |
| `predict` | `predict(pipeline, texts) -> (labels, probs)` | 返回预测标签数组和概率矩阵 |
| `load_baseline_model` | `load_baseline_model(path=None) -> Pipeline` | 通过 joblib 从磁盘加载序列化 Pipeline |

---

### `src/bert_model.py`

| 类 / 函数 | 说明 |
|----------|------|
| `SentimentDataset` | PyTorch `Dataset`；使用 HuggingFace tokenizer 分词；返回 `input_ids`、`attention_mask`、`label` 张量 |
| `SentimentClassifier(nn.Module)` | BERT 编码器 → Dropout(0.3) → `Linear(hidden_size, num_labels)` 分类头 |
| `train_bert(train_df, val_df, config=None)` | AdamW + 线性预热调度；逐 epoch 记录日志；保存最优 val_acc 检查点 |
| `predict_bert(model, tokenizer, texts, device)` | 批量推理；返回 `(labels, confidences)` numpy 数组 |
| `load_bert_model(path=None, num_labels=2)` | 从磁盘加载状态字典 + tokenizer；返回 `(model, tokenizer)` |

---

### `src/evaluate.py`

| 函数 | 签名 | 说明 |
|------|------|------|
| `evaluate_model` | `evaluate_model(y_true, y_pred, model_name, y_scores=None)` | 计算准确率、精确率、召回率、F1、ROC-AUC；打印分类报告 |
| `plot_confusion_matrix` | `plot_confusion_matrix(y_true, y_pred, model_name, labels=None)` | 归一化 seaborn 热力图；保存 PNG 至 `reports/figures/` |
| `plot_roc_curve` | `plot_roc_curve(y_true, y_scores, model_name)` | 带 AUC 标注的 ROC 曲线；保存 PNG |
| `compare_models` | `compare_models(baseline_results, bert_results)` | 生成双模型对比柱状图和 DataFrame |

---

### `src/visualize.py`

| 函数 | 说明 |
|------|------|
| `plot_sentiment_distribution(df)` | Plotly 交互式环形 / 饼图，展示各情感类别数量分布 |
| `plot_text_length_distribution(df)` | 按情感类别着色的重叠词数直方图 |
| `plot_wordcloud(df, sentiment=1)` | 保存词云 PNG；颜色方案随情感类别变化（绿 / 红 / 蓝） |
| `plot_sentiment_over_time(df, freq='D')` | 日 / 周情感趋势折线图 |
| `plot_top_keywords(df, n=20, sentiment=None)` | TF-IDF 高频词水平柱状图（可按情感过滤） |
| `plot_confidence_gauge(confidence, sentiment_label)` | Plotly 圆形仪表盘，用于 Streamlit 实时预测页 |

---

### `src/explain.py`

| 函数 | 签名 | 说明 |
|------|------|------|
| `explain_baseline_prediction` | `explain_baseline_prediction(pipeline, text, n_top=12)` | 使用 SHAP `LinearExplainer` 对 TF-IDF+LR 模型进行解释；返回 `(contributions, predicted_class, classes)`，其中 contributions 为按 `abs(shap_value)` 降序排列的 `(token, shap_value)` 列表 |
| `shap_to_plotly_bar` | `shap_to_plotly_bar(contributions, predicted_class, label_names=None)` | 将 SHAP token 归因渲染为水平 Plotly 柱状图（绿色 = 正向推动，红色 = 负向推动） |

---

## 界面截图

### 首页
![首页](reports/figures/screenshot_home.png)
*数据集统计信息、情感分布饼图及项目架构总览。*

### 数据分析页
![数据分析](reports/figures/screenshot_eda.png)
*交互式词云、文本长度直方图及 TF-IDF 关键词柱状图。*

### 实时预测 — 单条输入
![实时预测](reports/figures/screenshot_live_demo.png)
*实时预测并展示置信度仪表盘，支持单条文本输入和批量 CSV 下载。*

### 模型对比页
![模型对比](reports/figures/screenshot_comparison.png)
*基线模型与 BERT 的指标对比表、性能柱状图及 ROC 曲线对比。*

---

## 未来计划

- [x] **模型可解释性** — 基线模型 SHAP `LinearExplainer`（`src/explain.py`）
- [x] **Docker 部署** — `python:3.10-slim` 镜像，端口 8501，健康检查
- [x] **CI/CD 流水线** — GitHub Actions：flake8 代码检查 + 121 个 pytest 测试
- [x] **FastAPI REST 接口** — `/predict`、`/predict/batch`、`/health`（`api/serve.py`）
- [x] **超参数调优** — 3 折 GridSearchCV 附热力图（`scripts/tune_baseline.py`）
- [x] **错误分析** — 高置信度错误、否定词模式、类别难度（`04_error_analysis.ipynb`）
- [ ] **量化推理** — 基于 ONNX 的 INT8 BERT CPU 加速（速度提升 3–4 倍）
- [ ] **多类别细粒度情感** — 5 类标签（极消极 → 极积极）
- [ ] **实时数据接入** — Twitter / Reddit API 数据流管道
- [ ] **基于方面的情感分析（ABSA）** — 实体级别的观点挖掘
- [ ] **MLflow 实验追踪** — 记录每次训练的参数、指标与模型文件

---

## 文档维护标准

本仓库坚持**双语文档**策略。`README.md`（英文）和 [`README_CN.md`](README_CN.md)（中文）必须在**同一次 commit** 中同步更新，适用于所有影响用户可见行为的变更，包括 API 变更、新功能、安装步骤及项目结构调整。

所有公共函数遵循 **Google 风格文档字符串**，包含 `Args`、`Returns`、`Raises` 和 `Example` 四个标准块。Commit 消息遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。

详细的文档维护策略、文档字符串模板、Commit 格式指南及 PR 检查清单，请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

---

## 更新日志

完整的版本历史请参阅 [`CHANGELOG.md`](CHANGELOG.md)。

---

## 作者

**吕卓俊（Zhuojun Lyu）**
[GitHub](https://github.com/lvzhuojun) · [LinkedIn](https://www.linkedin.com/in/zhuojun-lyu/) · [邮箱](mailto:lzj2729033776@gmail.com)

---

*基于 Python 3.10 · HuggingFace Transformers · PyTorch · scikit-learn · Streamlit 构建*
