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
#### 1. Installation
  ```
# Clone repository
git clone https://github.com/Wyane653/mnist-classifier.git
cd mnist-classifier

# Install dependencies
pip install -r requirements.txt
  ```
#### 2. Train a Model
  ```
# Train SimpleCNN (default)
python src/train.py

# Train ImprovedCNN
python src/train.py  # Modify model_name in train.py config
  ```
#### 3. Evaluate Model
  ```
# Evaluate the best saved model
python src/evaluate.py
  ```
#### 4. Visualize Training
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

#### Configuration
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

#### Checkpoints
- *Timestamped files*: Full training state (model + optimizer + epoch)
- _best.pth: Best model weights only (for deployment)
Load a trained model:
```
checkpoint = torch.load('models/simple_cnn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 📈 Visualization Examples
#### The evaluation script generates three key visualizations:  
1.*Per-Class Accuracy*: Bar chart showing accuracy for each digit (0-9)  
2.*Confusion Matrix*: Heatmap visualizing misclassifications  
3.*Error Analysis*: Grid of misclassified samples with true/predicted labels

### 🎮 Real-time Interactive Demo
Experience the power of the model firsthand with our real-time recognition application! Draw digits with your mouse and watch the model's predictions and confidence levels update instantly.  
#### Features
- *Interactive Canvas*：Draw digits freely in a dedicated drawing area.
- *Dual Trigger Modes*：
  - *Auto-recognition*：Predicts automatically when you release the mouse button.
  - *Manual recognition*：Press the `s` key anytime to get a prediction for the current drawing.
- *Live Visualization*:Displays the top prediction, a real-time confidence bar chart for all digits (0-9), and a preview of the processed image the model sees.
- *Adaptive Interface*:The OpenCV-based GUI supports dynamic window resizing – all elements scale smoothly.
#### How to Run the Demo
- 1、*Install Extra Dependency*
  The demo requires OpenCV. Install it via pip:
  ```
  pip install opencv-python
  ```
  (Add `opencv-python` to your `requirements.txt` for future use.)
- 2、*Launch the Application*
  Make sure you have a trained model (e.g., `simple_cnn_best.pth`), then run:
  ```
  python src/realtime_demo.py
  ```
  or if using the newer OpenCV version:
  ```
  python src/realtime_opencv.py
  ```
#### Controls & Interface
|  Action   | Effect  |
|  Left-click & Drag  | Draw a digit on the canvas.  |
| Release Mouse Button  | Triggers auto-recognition. |
| Press `S` Key  | Triggers manual recognition at any time. |
| Press `C` Key  | Clears the drawing canvas. |
| Press `ESC` Key  | Exits the application cleanly. |
#### Demo Highlights
This interactive demo perfectly bridges the training pipeline and the static evaluation, allowing you to:
- Intuitively understand the model's strengths and weaknesses.
- Observe how subtle changes in your drawing affect the prediction confidence.
- Showcase the project's capabilities in an engaging way.


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
#### 1. 安装
```
# 克隆仓库
git clone https://github.com/Wyane653/mnist-classifier.git
cd mnist-classifier

# 安装依赖
pip install -r requirements.txt
```
#### 2. 训练模型
```
# 训练SimpleCNN（默认）
python src/train.py

# 训练ImprovedCNN
python src/train.py  # 在train.py配置中修改model_name
```
#### 3. 评估模型
```
# 评估保存的最佳模型
python src/evaluate.py
```
#### 4. 可视化训练过程
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
#### 配置训练
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
#### 检查点文件
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

### 🎮 实时交互式演示
通过我们的实时识别应用，亲身体验模型的强大能力！用鼠标绘制数字，即可实时查看模型的预测结果和置信度变化。
#### 功能特性
- *交互式画布*：在专门的绘图区域自由绘制数字。
- *双触发模式*：
  - *自动识别*：松开鼠标按键时自动触发预测。
  - *手动识别*：随时按下 `S` 键获取当前绘图的预测结果。
- *实时可视化*:显示最佳预测结果、所有数字（0-9）的实时置信度条形图，以及模型所看到的处理后图像预览。
- *自适应界面*:基于OpenCV的图形界面支持动态窗口缩放，所有元素平滑适应。
#### 如何运行演示
- 1、*安装额外依赖*
  演示程序需要OpenCV。通过pip安装：:
  ```
  pip install opencv-python
  ```
  (建议将 `opencv-python` 添加到 `requirements.txt` 文件中以便后续使用。)
- 2、*启动应用程序*
  确保已有训练好的模型（例如 `simple_cnn_best.pth`），然后运行：
  ```
  python src/realtime_demo.py
  ```
  或使用更新的OpenCV版本：
  ```
  python src/realtime_opencv.py
  ```
#### 控制与界面
|  操作   | 效果  |
|  *鼠标左键点击并拖动*  | 在画布上绘制数字。  |
| *松开鼠标按键*  | 触发自动识别。 |
| *按下 `S` 键*  | 随时触发手动识别。 |
| *按下 `c` 键*  | 清除绘图画布。 |
| *按下 `ESC` 键*  | 干净利落地退出应用程序。 |
#### 演示亮点
这个交互式演示完美地连接了训练流程和静态评估，让你能够：
- 直观理解模型的优势和薄弱环节。
- 观察绘图的细微变化如何影响预测置信度。
- 以一种引人入胜的方式展示项目成果。
