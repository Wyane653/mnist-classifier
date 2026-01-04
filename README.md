# MNIST Handwritten Digit Classifier / MNIST手写数字分类器

<https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg?logo=pytorch>  
<https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python>  
<https://img.shields.io/badge/License-MIT-yellow.svg>  
<https://img.shields.io/github/stars/Wyane653/mnist-classifier?style=social>  

---
**English** | [中文](#中文)
---

## English Version
### 📌 Overview
A complete, modular, and production-ready PyTorch implementation for MNIST handwritten digit recognition. This project demonstrates best practices in deep learning project structure, training pipelines, evaluation, and visualization.

### ✨ Features
- Modular Design: Clean separation of concerns (data, model, training, evaluation)

- Dual CNN Architectures: SimpleCNN (~99.2% accuracy) and ImprovedCNN (~99.4% accuracy)

- Complete Training Pipeline: Learning rate scheduling, checkpointing, TensorBoard logging

- Comprehensive Evaluation: Confusion matrix, error analysis, per-class accuracy visualization

- Professional Visualization: Training curves, sample predictions, misclassification analysis

- Extensible Codebase: Easy to add new models, datasets, or evaluation metrics

### 📁 Project Structure
>mnist_classifier/  
>├── src/                    # Source code  
>│   ├── train.py           # Main training script  
>│   ├── evaluate.py        # Model evaluation and analysis  
>│   ├── model.py           # CNN model definitions  
>│   ├── dataset.py         # Data loading and preprocessing  
>│   └── utils.py           # Utilities (visualization, checkpointing)  
>├── notebooks/             # Jupyter notebooks  
>│   └── exploration.ipynb  # Exploratory data analysis  
>├── models/                # Saved model checkpoints (.pth files)  
>├── data/                  # MNIST dataset (auto-downloaded)  
>├── runs/                  # TensorBoard logs  
>├── requirements.txt       # Dependencies  
>└── README.md             # This file

### 🚀 Quick Start
1. Installation
  ```
# Clone repository
git clone https://github.com/Wyane653/mnist-classifier.git
cd mnist-classifier

# Install dependencies
pip install -r requirements.txt
  ```
2. Train a Model
  ```
# Train SimpleCNN (default)
python src/train.py

# Train ImprovedCNN
python src/train.py  # Modify model_name in train.py config
  ```
3. Evaluate Model
  ```
# Evaluate the best saved model
python src/evaluate.py
  ```
4. Visualize Training
  ```
# Launch TensorBoard
tensorboard --logdir=runs
  ```

### 📊 Model Performance
| Model | Parameters | Accuracy | Training Time (GPU) |
| :----:| :----: | :----: | :----: |
| SimpleCNN | ~390K | 99.2% | ~5 minutes |
| ImprovedCNN | ~1.2M | 99.4% | ~10 minutes |

### 🔧 Detailed Usage

*Configuration*
Edit `train.py` to modify training parameters:
```
config = {
    'model_name': 'simple_cnn',  # 'simple_cnn' or 'improved_cnn'
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 20,
    'weight_decay': 1e-5,
}
```

*Checkpoints*
- *Timestamped files*: Full training state (model + optimizer + epoch)
- _best.pth: Best model weights only (for deployment)
Load a trained model:
```
checkpoint = torch.load('models/simple_cnn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 📈 Visualization Examples
The evaluation script generates three key visualizations:  
1.*Per-Class Accuracy*: Bar chart showing accuracy for each digit (0-9)  
2.*Confusion Matrix*: Heatmap visualizing misclassifications  
3.*Error Analysis*: Grid of misclassified samples with true/predicted labels

## 中文版本
### 📌 项目概述
一个完整、模块化、生产就绪的PyTorch手写数字识别项目。本项目展示了深度学习项目结构、训练流程、评估和可视化的最佳实践。  

### ✨ 核心特性
- *模块化设计*：清晰的职责分离（数据、模型、训练、评估）
- *双CNN架构*：`SimpleCNN`（约99.2%准确率）和`ImprovedCNN`（约99.4%准确率）
- *完整训练流程*:学习率调度、检查点保存、TensorBoard日志记录
- *全面评估系统*:混淆矩阵、错误分析、逐类别准确率可视化
- *专业可视化工具*:训练曲线、样本预测、错误分类分析
- *可扩展代码库*:易于添加新模型、数据集或评估指标

### 📁 项目结构
>mnist_classifier/  
>├── src/                    # Source code  
>│   ├── train.py           # Main training script  
>│   ├── evaluate.py        # Model evaluation and analysis  
>│   ├── model.py           # CNN model definitions  
>│   ├── dataset.py         # Data loading and preprocessing  
>│   └── utils.py           # Utilities (visualization, checkpointing)  
>├── notebooks/             # Jupyter notebooks  
>│   └── exploration.ipynb  # Exploratory data analysis  
>├── models/                # Saved model checkpoints (.pth files)  
>├── data/                  # MNIST dataset (auto-downloaded)  
>├── runs/                  # TensorBoard logs  
>├── requirements.txt       # Dependencies  
>└── README.md             # This file

### 🚀 快速开始
*1. 安装*
```
# 克隆仓库
git clone https://github.com/Wyane653/mnist-classifier.git
cd mnist-classifier

# 安装依赖
pip install -r requirements.txt
```
*2. 训练模型*
```
# 训练SimpleCNN（默认）
python src/train.py

# 训练ImprovedCNN
python src/train.py  # 在train.py配置中修改model_name
```
*3. 评估模型*
```
# 评估保存的最佳模型
python src/evaluate.py
```
*4. 可视化训练过程*
```
# 启动TensorBoard
tensorboard --logdir=runs
```

### 📊 模型性能
| 模型 | 参数 | 准确率 | 训练时间 |
| :----:| :----: | :----: | :----: |
| SimpleCNN | ~39万 | 99.2% | ~5 分钟 |
| ImprovedCNN | ~120万 | 99.4% | ~10 分钟 |

### 🔧 详细使用说明
*配置训练*
编辑`train.py`修改训练参数：
```
config = {
    'model_name': 'simple_cnn',  # 'simple_cnn' 或 'improved_cnn'
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 20,
    'weight_decay': 1e-5,
}
```
*检查点文件*
- *带时间戳的文件*:完整训练状态（模型+优化器+训练轮次）  
- `_best.pth`*文件*:仅最佳模型权重（用于部署）
加载训练好的模型：
```
checkpoint = torch.load('models/simple_cnn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```
### 📈 可视化示例
评估脚本生成三种关键可视化图表：  
1.*逐类别准确率*:显示每个数字（0-9）准确率的条形图  
2.*混淆矩阵*:可视化错误分类的热力图  
3.*错误分析*:被错误分类的样本网格，显示真实/预测标签
