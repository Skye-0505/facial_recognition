# 面部表情识别与特效系统

## 📌 项目简介

系统包含三个核心模块：表情识别模型训练、面部关键点检测模型训练、以及完整的摄像头/图片特效合成流水线。

---

## 🚀 快速开始

### 运行顺序

```bash
# 1. 训练表情识别模型
python emotion_models.py

# 2. 训练人脸关键点检测模型（5点 + 98点）
python landmarks_models.py

# 3. 运行特效合成系统
python special_effects.py
```

---

## 📁 文件说明

### 1️⃣ `emotion_models.py` —— 表情识别模型训练

**功能**  
在RAF-DB数据集上训练面部表情分类模型，支持ResNet-18和ViT，包含数据增强、Focal Loss处理类别不平衡、训练可视化及模型保存。

**模型架构**  
- `ResNet-18`（ImageNet / Celeb-1M 预训练）  
- `ViT-small-patch16-224`（可选，timm库实现）

**核心参数**  
| 参数 | 值 |
|------|-----|
| 输入尺寸 | 224×224 |
| Batch Size | 8 |
| Epochs | 20 |
| 学习率 | 5e-5 |
| Loss | Focal Loss (γ=2.0) |
| 优化器 | Adam |
| 类别 | 7类（surprise, fear, disgust, happy, sad, anger, natural） |

**输出文件**  
- `fer_resnet18_best.pth`：最佳模型权重  
- `training_config.json`：训练配置  
- `outputs/emotion/visualize/`：训练曲线图、预测可视化

---

### 2️⃣ `landmarks_models.py` —— 面部关键点检测模型训练

**功能**  
训练两个独立的人脸关键点检测模型：
1. **5点模型**（双眼中心、鼻尖、嘴角）—— 适用于基础应用
2. **WFLW 98点模型** —— 高精度面部网格，用于特效精准定位

**模型架构**  
- `MobileNetV2` + 全连接回归头  
- 输出维度：`关键点数 × 2`（归一化坐标）

**数据集**  
- 5点：CUHK 数据集（自动下载）
- 98点：WFLW 数据集（需手动放置）

**核心参数**  
| 参数 | 5点模型 | 98点模型 |
|------|---------|----------|
| 输入尺寸 | 224×224 | 224×224 |
| Batch Size | 16 | 16 |
| Epochs | 3 | 3 |
| Loss | MSE | MSE |
| 优化器 | Adam (lr=0.001) | Adam (lr=0.001) |

**输出文件**  
- `fivepoint_model.pth`：5点模型权重  
- `wflw_model.pth`：98点模型权重  
- `outputs/landmarks/`：数据集可视化、预测结果对比图、统计分析图表

---

### 3️⃣ `special_effects.py` —— 实时特效合成系统

**功能**  
完整的人脸特效处理流水线，支持单张图片和批量处理：

1. **人脸检测**：MTCNN（保证单人脸）
2. **表情识别**：调用训练好的ResNet-18模型
3. **关键点检测**：调用WFLW 98点模型
4. **特效合成**：根据表情类型在特定面部位置叠加图标

**表情特效规则**  
| 表情 | 图标位置 | 缩放系数 |
|------|---------|----------|
| angry | 右眉上方 | 0.8 |
| disgust | 右嘴角下方 | 0.6 |
| fear | 右眉上方偏左 | 0.7 |
| happy | 双眼外侧（双图标） | 0.5 |
| sad | 双眼外眼角下方（双图标） | 0.4 |
| surprise | 右眉上方较大 | 0.9 |
| neutral | 鼻尖左侧 | 0.7 |

**输入输出**  
- **输入图片**：`../../data/application/test_img/`
- **表情图标**：`../../data/application/icons/{emotion}.png`（需RGBA格式）
- **输出目录**：`../../outputs/application/`
  - `images/`：合成结果
  - `visualization/`：原图+结果对比图
  - `logs/`：处理日志

---

## ⚙️ 环境依赖

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
pillow>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.2.0
timm>=0.5.4      # 仅ViT需要
facenet-pytorch>=2.5.0  # MTCNN
gdown>=4.5.0     # 下载Celeb-1M权重
tqdm>=4.62.0
```

---

## 📊 模型性能参考

| 模型 | 数据集 | 准确率/误差 |
|------|--------|------------|
| ResNet-18 | RAF-DB | ~85% 验证准确率 |
| MobileNetV2-5pt | CUHK | MSE < 0.01 |
| MobileNetV2-98pt | WFLW | 平均误差 ~8像素 |

---

## 📝 注意事项

1. **运行顺序**：必须先训练模型，再运行特效系统
2. **WFLW数据集**：已手动下载并放置到指定目录，程序不会自动下载
3. **测试图片**：确保为单人正面人脸，避免多人脸或多角度
4. **设备要求**：建议CUDA加速，CPU模式下速度较慢

---