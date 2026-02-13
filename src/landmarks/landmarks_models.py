"""
人脸关键点检测 - 支持5点数据集和WFLW 98点数据集
功能：
1. 5点人脸关键点检测（MobileNetV2）
2. WFLW 98点人脸关键点检测（MobileNetV2）
"""

import os
import warnings
import urllib.request
import zipfile
from typing import Tuple, Dict, List, Optional, Union, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免终端显示问题
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 5点数据集路径配置
FIVEPOINT_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, './data/fivepoint'))
FIVEPOINT_TRAIN_ZIP_URL = "http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip"
FIVEPOINT_TRAIN_ANNOTATION = os.path.join(FIVEPOINT_DATA_DIR, 'trainImageList.txt')
FIVEPOINT_TEST_ANNOTATION = os.path.join(FIVEPOINT_DATA_DIR, 'testImageList.txt')
FIVEPOINT_IMG_DIR = os.path.join(FIVEPOINT_DATA_DIR, './')

# WFLW数据集路径配置
WFLW_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, './data/wflw'))
WFLW_IMG_DIR = os.path.join(WFLW_DATA_DIR, 'WFLW_images')
WFLW_ANNOTATIONS_DIR = os.path.join(WFLW_DATA_DIR, 'WFLW_annotations')
WFLW_TRAIN_ANNOTATION = os.path.join(
    WFLW_ANNOTATIONS_DIR, 
    'list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'
)
WFLW_TEST_ANNOTATION = os.path.join(
    WFLW_ANNOTATIONS_DIR, 
    'list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
)

# 模型保存路径
MODEL_SAVE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, './models'))
WFLW_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'wflw_model.pth')
FIVEPOINT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'fivepoint_model.pth')

# 输出图像保存路径 - 修改为../../outputs/landmarks
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../outputs/landmarks'))
FIVEPOINT_VIS_DIR = os.path.join(OUTPUT_DIR, 'fivepoint')
WFLW_VIS_DIR = os.path.join(OUTPUT_DIR, 'wflw')
WFLW_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, 'analysis')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')

# 创建必要的目录
os.makedirs(FIVEPOINT_DATA_DIR, exist_ok=True)
os.makedirs(WFLW_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(WFLW_IMG_DIR, exist_ok=True)
os.makedirs(WFLW_ANNOTATIONS_DIR, exist_ok=True)

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIVEPOINT_VIS_DIR, exist_ok=True)
os.makedirs(WFLW_VIS_DIR, exist_ok=True)
os.makedirs(WFLW_ANALYSIS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ==================== 工具函数 ====================

def download_and_extract_zip(url: str, extract_path: str = None) -> None:
    """
    下载并解压ZIP文件
    
    Args:
        url: ZIP文件URL
        extract_path: 解压路径，默认为5点数据集路径
    """
    if extract_path is None:
        extract_path = FIVEPOINT_DATA_DIR
    
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_path, filename)
    
    try:
        print(f"📥 正在下载: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"📦 正在解压到: {extract_path}")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(filepath)
        print(f"✅ 成功下载并解压: {filename}")
    except Exception as e:
        print(f"❌ 下载/解压失败: {e}")


def euclidean_dist(vector_x: np.ndarray, vector_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算欧氏距离和归一化距离
    
    Args:
        vector_x: 预测坐标
        vector_y: 真实坐标
    
    Returns:
        distance: 欧氏距离
        normalized_distance: 归一化距离
    """
    vector_x, vector_y = np.array(vector_x), np.array(vector_y)
    distance = np.sqrt(np.sum((vector_x - vector_y) ** 2, axis=-1))
    
    # 计算归一化因子（两个真实点之间的距离）
    if len(vector_y) >= 2:
        distance_y = np.sqrt(np.sum((vector_y[0] - vector_y[1]) ** 2))
    else:
        distance_y = 1.0
    
    normalized_distance = distance / (distance_y + 1e-8)
    return distance, normalized_distance


def save_figure(fig, save_path: str, dpi=150, bbox_inches='tight'):
    """保存图像到指定路径"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"💾 图像已保存: {save_path}")


# ==================== 通用数据变换类 ====================

class NormalizeBBoxFormat(object):
    """标准化边界框格式 - 转换为统一的[x1, y1, x2, y2]格式"""
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """将不同格式的边界框统一为[x1, y1, x2, y2]格式"""
        if 'bbox' not in sample or sample['bbox'] is None:
            return sample
            
        bbox = sample['bbox']
        
        # 转换为统一的[x1, y1, x2, y2]格式
        if isinstance(bbox, (list, np.ndarray, tuple)):
            bbox = np.array(bbox).flatten()
            
            if len(bbox) == 4:
                # 已经是[x1,y1,x2,y2]格式
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            elif len(bbox) == 2 and len(bbox[0]) == 2:  # [[x1,y1], [x2,y2]]
                bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
        
        sample['bbox'] = list(map(float, bbox))
        return sample


class BBoxCrop(object):
    """通用边界框裁剪 - 支持任意数量关键点"""
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sample: 包含'image', 'landmarks', 'bbox'的字典
                   bbox格式: [x1, y1, x2, y2]
        Returns:
            裁剪后的图像和调整后的关键点
        """
        image = sample['image']
        landmarks = sample['landmarks']
        bbox = sample['bbox']
        
        if bbox is None or len(bbox) != 4:
            print(f"⚠️ 跳过裁剪: 无效的边界框 {bbox}")
            return sample
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保边界框在图像范围内
        h, w = image.shape[1], image.shape[2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 裁剪图像并调整关键点
        if x2 > x1 and y2 > y1:
            image = image[:, y1:y2, x1:x2]
            if landmarks is not None:
                landmarks = landmarks - [x1, y1]
        else:
            print(f"⚠️ 无效的边界框: [{x1}, {y1}, {x2}, {y2}]")
        
        # 保留原始sample中可能存在的其他字段
        result = {'image': image, 'landmarks': landmarks}
        for key in sample:
            if key not in ['image', 'landmarks', 'bbox']:
                result[key] = sample[key]
        
        return result


class Rescale(object):
    """通用图像缩放 - 支持任意数量关键点"""
    
    def __init__(self, output_size: Tuple[int, int]):
        """
        Args:
            output_size: 目标尺寸 (height, width)
        """
        assert isinstance(output_size, (tuple, list)), "output_size必须是tuple/list类型"
        self.output_size = output_size

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sample: 包含'image', 'landmarks'的字典
        Returns:
            缩放后的图像和调整后的关键点
        """
        image = sample['image']
        landmarks = sample['landmarks']
        
        h, w = image.shape[1], image.shape[2]
        new_h, new_w = self.output_size
        
        # 调整图像大小
        image = F.resize(image.clone() if isinstance(image, torch.Tensor) else image, 
                        (new_h, new_w))
        
        # 调整关键点坐标
        if h > 0 and w > 0 and landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]

        result = {'image': image, 'landmarks': landmarks}
        for key in sample:
            if key not in ['image', 'landmarks']:
                result[key] = sample[key]
        
        return result


class ToTensor(object):
    """通用Tensor转换和归一化 - 支持任意数量关键点"""
    
    def __init__(self, device='cpu', image_size=224, normalize_landmarks=True):
        """
        Args:
            device: 设备 ('cpu' 或 'cuda')
            image_size: 图像尺寸，用于关键点归一化
            normalize_landmarks: 是否将关键点归一化到[0,1]
        """
        self.device = device
        self.image_size = image_size
        self.normalize_landmarks = normalize_landmarks

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            sample: 包含'image', 'landmarks'的字典
        Returns:
            归一化后的Tensor
        """
        image = sample['image']
        landmarks = sample['landmarks']
        
        # 图像归一化
        if isinstance(image, torch.Tensor):
            image = image.float() / 255.0
        else:
            image = torch.tensor(image, dtype=torch.float32) / 255.0
        
        image = image.to(self.device)
        
        # 关键点处理
        if landmarks is not None:
            if isinstance(landmarks, np.ndarray):
                landmarks = torch.tensor(landmarks, dtype=torch.float32)
            elif not isinstance(landmarks, torch.Tensor):
                landmarks = torch.tensor(landmarks, dtype=torch.float32)
            
            # 归一化关键点
            if self.normalize_landmarks:
                landmarks = landmarks / self.image_size
            
            landmarks = landmarks.to(self.device)

        result = {'image': image, 'landmarks': landmarks}
        for key in sample:
            if key not in ['image', 'landmarks']:
                result[key] = sample[key]
        
        return result


def get_transform_pipeline(image_size=224, device='cpu', include_bbox=True):
    """
    获取统一的数据变换流水线
    
    Args:
        image_size: 目标图像尺寸
        device: 计算设备
        include_bbox: 是否包含边界框处理
    """
    transforms = []
    
    if include_bbox:
        transforms.append(NormalizeBBoxFormat())
        transforms.append(BBoxCrop())
    
    transforms.extend([
        Rescale((image_size, image_size)),
        ToTensor(device=device, image_size=image_size)
    ])
    
    return Compose(transforms)


# ==================== 5点数据集类 ====================

class FaceLandmarksDataset(Dataset):
    """5点人脸关键点数据集"""
    
    COLUMNS = [
        'image_path', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2',
        'landmark1_x', 'landmark1_y', 'landmark2_x', 'landmark2_y',
        'landmark3_x', 'landmark3_y', 'landmark4_x', 'landmark4_y',
        'landmark5_x', 'landmark5_y'
    ]
    
    def __init__(self, annotations_file: str, img_dir: str, transform=None):
        """
        Args:
            annotations_file: 标注文件绝对路径
            img_dir: 图像目录绝对路径
            transform: 数据变换
        """
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        
        # 检查文件是否存在
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"标注文件不存在: {annotations_file}")
        
        self.df = pd.read_csv(annotations_file, delimiter=' ', names=self.COLUMNS)
        
        # 处理图像路径 - 转换为绝对路径
        self.df['image_path'] = self.df['image_path'].str.replace('\\', '/')
        self.df['image_path'] = self.df['image_path'].apply(
            lambda x: os.path.join(self.img_dir, os.path.basename(x))
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx, 0]
        
        # 读取图像
        try:
            image = read_image(img_path)
        except Exception as e:
            print(f"⚠️ 读取图像失败: {img_path}, 错误: {e}")
            image = torch.zeros((3, 224, 224), dtype=torch.uint8)

        # 解析边界框 [x1, y1, x2, y2] 格式
        bbox = self.df.iloc[idx, 1:5].values.astype(float)
        bbox = [bbox[0], bbox[2], bbox[1], bbox[3]]  # 转换为[x1,y1,x2,y2]
        
        # 解析关键点 (5个点)
        landmarks = self.df.iloc[idx, 5:].values.astype(float).reshape(-1, 2)

        sample = {
            'image': image, 
            'bbox': bbox, 
            'landmarks': landmarks,
            'img_path': img_path
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# ==================== WFLW 98点数据集类 ====================

class WFLWDataset(Dataset):
    """WFLW 98点人脸关键点数据集"""
    
    NUM_LANDMARKS = 98
    ATTR_NAMES = ['pose', 'expression', 'illumination', 'make-up', 'occlusion', 'blur']
    
    def __init__(self, annotation_file: str, img_dir: str, transform=None):
        """
        Args:
            annotation_file: 标注文件绝对路径
            img_dir: 图像目录绝对路径
            transform: 数据变换
        """
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        self._load_annotations(annotation_file)
        print(f"  ✅ 加载完成: {len(self.samples)} 个样本")

    def _load_annotations(self, annotation_file: str) -> None:
        """加载标注文件"""
        if not os.path.exists(annotation_file):
            print(f"❌ 标注文件不存在: {annotation_file}")
            return

        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 207:
                    continue
                
                try:
                    sample = self._parse_annotation_line(parts)
                    if sample:
                        self.samples.append(sample)
                except Exception as e:
                    print(f"⚠️ 解析失败: {e}")
                    continue

    def _parse_annotation_line(self, parts: List[str]) -> Optional[Dict]:
        """解析单行标注"""
        image_name = parts[-1]
        
        # 处理图像路径 - 转换为绝对路径
        img_path = os.path.join(self.img_dir, image_name)
        
        # 如果默认路径不存在，尝试其他可能的路径
        if not os.path.exists(img_path):
            alt_path = os.path.join(self.img_dir, 'WFLW_images', image_name)
            if os.path.exists(alt_path):
                img_path = alt_path
        
        # 98个关键点 (196个值)
        landmarks = np.array(list(map(float, parts[:196]))).reshape(self.NUM_LANDMARKS, 2)
        
        # 边界框 [x_min, y_min, x_max, y_max]
        bbox = list(map(float, parts[196:200]))
        
        # 属性
        attributes = list(map(int, parts[200:206]))
        attr_dict = dict(zip(self.ATTR_NAMES, attributes))
        
        return {
            'img_path': img_path,
            'landmarks': landmarks,
            'bbox': bbox,
            'attributes': attr_dict,
            'image_name': image_name
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # 读取图像
        try:
            image = read_image(sample['img_path'])
        except Exception as e:
            print(f"⚠️ 读取图像失败: {sample['img_path']}, 错误: {e}")
            image = torch.zeros((3, 224, 224), dtype=torch.uint8)
        
        data = {
            'image': image,
            'landmarks': sample['landmarks'].astype(np.float32),
            'bbox': sample['bbox'],
            'attributes': sample['attributes'],
            'img_path': sample['img_path']
        }
        
        if self.transform:
            data = self.transform(data)
        
        return data


# ==================== 模型构建 ====================

def build_model(num_landmarks: int, pretrained: bool = True) -> nn.Module:
    """
    构建MobileNetV2模型
    
    Args:
        num_landmarks: 关键点数量
        pretrained: 是否使用预训练权重
    
    Returns:
        修改后的MobileNetV2模型
    """
    model = torchvision.models.mobilenet_v2(pretrained=pretrained, progress=False)
    model.classifier[1] = nn.Linear(1280, num_landmarks * 2)
    return model


# ==================== 训练和评估函数 ====================

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                device: torch.device, epoch: int, num_epochs: int) -> float:
    """
    训练一个epoch
    
    Returns:
        平均损失
    """
    model.train()
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader), 
               desc=f"Epoch [{epoch+1}/{num_epochs}]")
    
    for i, data in bar:
        inputs = data['image'].to(device)
        targets = data['landmarks'].to(device)
        
        # 展平关键点
        batch_size_i = inputs.size(0)
        targets_flat = targets.view(batch_size_i, -1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets_flat)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        bar.set_postfix(loss=running_loss / (i + 1))
    
    return running_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, 
             criterion: nn.Module, device: torch.device) -> float:
    """
    评估模型
    
    Returns:
        平均损失
    """
    model.eval()
    test_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="评估"):
            inputs = data['image'].to(device)
            targets = data['landmarks'].to(device)
            
            batch_size_i = inputs.size(0)
            targets_flat = targets.view(batch_size_i, -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets_flat)
            
            test_loss += loss.item() * batch_size_i
            total_samples += batch_size_i
    
    return test_loss / total_samples


# ==================== 可视化函数（保存图像） ====================

def visualize_5point_dataset(dataset: Dataset, num_samples: int = 4, save_dir: str = None) -> None:
    """
    可视化5点数据集并保存图像
    
    Args:
        dataset: 数据集
        num_samples: 可视化样本数
        save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = FIVEPOINT_VIS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
            
        ax = plt.subplot(1, num_samples, i + 1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}')
        ax.axis('off')

        # 显示图像
        img_display = sample['image'].permute(1, 2, 0).numpy()
        plt.imshow(img_display)

        # 绘制关键点
        plt.scatter(sample['landmarks'][:, 0], sample['landmarks'][:, 1], 
                   s=50, marker='.', c='r')

        # 绘制边界框
        if 'bbox' in sample and sample['bbox'] is not None:
            bbox = sample['bbox']
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                               linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    
    plt.suptitle('5点人脸关键点数据集示例')
    
    # 保存图像
    save_path = os.path.join(save_dir, f'fivepoint_dataset_vis_{timestamp}.png')
    save_figure(fig, save_path)


def visualize_wflw_dataset(dataset: Dataset, num_samples: int = 3, save_dir: str = None) -> None:
    """
    可视化WFLW原始数据集并保存图像
    
    Args:
        dataset: 数据集
        num_samples: 可视化样本数
        save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = WFLW_VIS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    if num_samples == 1:
        axes = [axes]
    
    # 定义不同部位的颜色
    colors = {
        'contour': 'blue',      # 面部轮廓
        'eyebrow': 'green',     # 眉毛
        'nose': 'red',         # 鼻子
        'eyes': 'yellow',      # 眼睛
        'mouth': 'purple',     # 嘴巴
        'special': 'white'     # 特殊点
    }
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        landmarks = sample['landmarks']
        bbox = sample['bbox']
        attributes = sample.get('attributes', {})
        
        ax = axes[i]
        img_display = image.permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        
        # 绘制不同部位的关键点
        if landmarks is not None:
            # 面部轮廓 (0-32)
            if len(landmarks) >= 33:
                ax.scatter(landmarks[:33, 0], landmarks[:33, 1], 
                          s=10, c=colors['contour'], marker='.', 
                          label='轮廓' if i == 0 else "")
            
            # 眉毛 (33-70)
            if len(landmarks) >= 71:
                ax.scatter(landmarks[33:71, 0], landmarks[33:71, 1], 
                          s=10, c=colors['eyebrow'], marker='.', 
                          label='眉毛' if i == 0 else "")
            
            # 鼻子 (71-87)
            if len(landmarks) >= 88:
                ax.scatter(landmarks[71:88, 0], landmarks[71:88, 1], 
                          s=10, c=colors['nose'], marker='.', 
                          label='鼻子' if i == 0 else "")
            
            # 眼睛 (88-103)
            if len(landmarks) >= 104:
                ax.scatter(landmarks[88:104, 0], landmarks[88:104, 1], 
                          s=10, c=colors['eyes'], marker='.', 
                          label='眼睛' if i == 0 else "")
            
            # 嘴巴 (104-121)
            if len(landmarks) >= 122:
                ax.scatter(landmarks[104:122, 0], landmarks[104:122, 1], 
                          s=10, c=colors['mouth'], marker='.', 
                          label='嘴巴' if i == 0 else "")
            
            # 特殊点
            special_indices = [0, 32, 33, 70, 71, 87, 88, 103, 104, 121]
            valid_indices = [idx for idx in special_indices if idx < len(landmarks)]
            if valid_indices:
                ax.scatter(landmarks[valid_indices, 0], landmarks[valid_indices, 1], 
                          s=50, c=colors['special'], marker='o', 
                          edgecolors='black', label='特殊点' if i == 0 else "")
        
        # 绘制边界框
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                               linewidth=2, edgecolor='cyan', 
                               facecolor='none', alpha=0.8)
            ax.add_patch(rect)
        
        # 添加标题
        title = f"样本 {i}"
        if attributes:
            problem_attrs = [k for k, v in attributes.items() if v == 1]
            if problem_attrs:
                title += f"\n问题: {','.join(problem_attrs)}"
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('WFLW数据集原始样本', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'wflw_dataset_vis_{timestamp}.png')
    save_figure(fig, save_path)


def visualize_predictions(model: nn.Module, dataset: Dataset, 
                         device: torch.device, num_samples: int = 6,
                         image_size: int = 224, save_dir: str = None) -> None:
    """
    可视化预测结果并保存图像
    
    Args:
        model: 模型
        dataset: 数据集
        device: 设备
        num_samples: 可视化样本数
        image_size: 图像尺寸
        save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = PREDICTIONS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model.eval()
    
    rows = (num_samples + 2) // 3
    cols = min(3, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            
            # 预测关键点
            pred_landmarks = model(image)
            pred_landmarks = pred_landmarks.cpu().numpy().reshape(-1, 2) * image_size
            
            # 真实关键点
            true_landmarks = sample['landmarks'].cpu().numpy() * image_size
            
            # 显示图像
            img_display = image.squeeze().cpu().permute(1, 2, 0).numpy()
            
            axes[i].imshow(img_display)
            
            # 绘制真实关键点（绿色）
            axes[i].scatter(true_landmarks[:, 0], true_landmarks[:, 1], 
                          s=5, c='g', marker='.', alpha=0.6, label='真实')
            
            # 绘制预测关键点（红色）
            axes[i].scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], 
                          s=5, c='r', marker='x', alpha=0.6, label='预测')
            
            # 计算平均误差
            if len(true_landmarks) == len(pred_landmarks):
                error = np.sqrt(np.sum((pred_landmarks - true_landmarks) ** 2, axis=1)).mean()
                axes[i].set_title(f'样本 {i+1}\n平均误差: {error:.1f} 像素')
            else:
                axes[i].set_title(f'样本 {i+1}')
            
            axes[i].axis('off')
            axes[i].legend(loc='upper right', fontsize='small')
    
    # 隐藏多余的子图
    for i in range(min(num_samples, len(dataset)), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('WFLW人脸关键点检测结果\n绿色: 真实值, 红色: 预测值', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'wflw_predictions_{timestamp}.png')
    save_figure(fig, save_path)


# ==================== 数据分析函数 ====================

def analyze_wflw_dataset(annotation_file: str, save_dir: str = None) -> None:
    """
    分析WFLW数据集统计信息并保存图表
    
    Args:
        annotation_file: 标注文件路径
        save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = WFLW_ANALYSIS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(annotation_file):
        print(f"❌ 文件不存在: {annotation_file}")
        return
    
    data = []
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 207:
                continue
            
            try:
                landmarks = np.array(list(map(float, parts[:196]))).reshape(98, 2)
                bbox = list(map(float, parts[196:200]))
                attributes = list(map(float, parts[200:206]))
                data.append({'landmarks': landmarks, 'bbox': bbox, 'attributes': attributes})
            except:
                continue
    
    if len(data) == 0:
        print("❌ 没有读取到有效数据")
        return
    
    print(f"✅ 共读取 {len(data)} 条数据")
    
    # 基础统计
    print("\n📊 基础统计:")
    all_landmarks = np.vstack([d['landmarks'] for d in data])
    print(f"   关键点总数: {all_landmarks.shape[0]:,}")
    print(f"   坐标范围: X[{all_landmarks[:,0].min():.0f}, {all_landmarks[:,0].max():.0f}]")
    print(f"            Y[{all_landmarks[:,1].min():.0f}, {all_landmarks[:,1].max():.0f}]")
    
    # 人脸尺寸分布
    bboxes = np.array([d['bbox'] for d in data])
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    
    print(f"\n📏 人脸尺寸统计:")
    print(f"   平均尺寸: {widths.mean():.0f}×{heights.mean():.0f} 像素")
    print(f"   最小人脸: {widths.min():.0f}×{heights.min():.0f}")
    print(f"   最大人脸: {widths.max():.0f}×{heights.max():.0f}")
    
    # 图像质量统计
    attributes_array = np.array([d['attributes'] for d in data])
    occlusion = attributes_array[:, 4]
    blur = attributes_array[:, 5]
    
    good = ((occlusion <= 0.3) & (blur <= 0.3)).sum()
    has_occlusion = (occlusion > 0.3).sum()
    has_blur = (blur > 0.3).sum()
    has_both = ((occlusion > 0.3) & (blur > 0.3)).sum()
    
    print(f"\n🖼️ 图像质量统计:")
    print(f"   清晰图像: {good} ({good/len(data)*100:.1f}%)")
    print(f"   遮挡: {has_occlusion} ({has_occlusion/len(data)*100:.1f}%)")
    print(f"   模糊: {has_blur} ({has_blur/len(data)*100:.1f}%)")
    print(f"   遮挡+模糊: {has_both} ({has_both/len(data)*100:.1f}%)")
    
    # 绘制图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 人脸尺寸分布
    ax1.hist(widths, bins=30, alpha=0.7, label=f'宽度 (平均{widths.mean():.0f})', color='blue')
    ax1.hist(heights, bins=30, alpha=0.7, label=f'高度 (平均{heights.mean():.0f})', color='red')
    ax1.set_xlabel('像素值')
    ax1.set_ylabel('图像数量')
    ax1.set_title('人脸尺寸分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图像质量饼图
    quality_counts = [good, has_occlusion, has_blur, has_both]
    quality_labels = ['清晰图像', '遮挡', '模糊', '遮挡+模糊']
    colors = ['#66b3ff', '#ff9999', '#ffcc99', '#ff6666']
    
    ax2.pie(quality_counts, labels=quality_labels, colors=colors, 
           autopct='%1.1f%%', startangle=90)
    ax2.set_title('图像质量分布')
    
    plt.suptitle('WFLW数据集分析', fontsize=14)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, f'wflw_analysis_{timestamp}.png')
    save_figure(fig, save_path)


# ==================== 主程序 ====================

def train_5point_model():
    """训练5点人脸关键点检测模型"""
    print("=" * 50)
    print("1. 5点人脸关键点检测训练")
    print("=" * 50)
    
    # 下载数据
    if not os.path.exists(FIVEPOINT_TRAIN_ANNOTATION):
        download_and_extract_zip(FIVEPOINT_TRAIN_ZIP_URL, FIVEPOINT_DATA_DIR)
    else:
        print(f"✅ 5点数据集已存在: {FIVEPOINT_DATA_DIR}")
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集和变换 - 使用统一的数据变换流水线
    transform = get_transform_pipeline(
        image_size=224, 
        device=device, 
        include_bbox=True
    )
    
    train_dataset = FaceLandmarksDataset(
        annotations_file=FIVEPOINT_TRAIN_ANNOTATION, 
        img_dir=FIVEPOINT_IMG_DIR, 
        transform=transform
    )
    
    test_dataset = FaceLandmarksDataset(
        annotations_file=FIVEPOINT_TEST_ANNOTATION, 
        img_dir=FIVEPOINT_IMG_DIR, 
        transform=transform
    )
    
    # 可视化原始数据集并保存
    print(f"可视化原始数据集，图像保存至: {FIVEPOINT_VIS_DIR}")
    raw_dataset = FaceLandmarksDataset(
        annotations_file=FIVEPOINT_TRAIN_ANNOTATION, 
        img_dir=FIVEPOINT_IMG_DIR, 
        transform=None
    )
    visualize_5point_dataset(raw_dataset, num_samples=4, save_dir=FIVEPOINT_VIS_DIR)
    
    # 数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")
    
    # 模型
    model = build_model(num_landmarks=5, pretrained=True).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 3
    print(f"\n开始训练，共 {num_epochs} 个epoch...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {train_loss:.4f}")
    
    # 评估
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"测试集平均损失: {test_loss:.4f}")
    
    # 保存模型
    torch.save(model, FIVEPOINT_MODEL_PATH)
    print(f"5点模型已保存至: {FIVEPOINT_MODEL_PATH}")


def train_wflw_model():
    """训练WFLW 98点人脸关键点检测模型"""
    print("\n" + "=" * 50)
    print("2. WFLW 98点人脸关键点检测训练")
    print("=" * 50)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 分析数据集并保存图表
    print("\n分析训练集...")
    analyze_wflw_dataset(WFLW_TRAIN_ANNOTATION, save_dir=WFLW_ANALYSIS_DIR)
    
    # 数据集变换 - 使用统一的数据变换流水线
    transform = get_transform_pipeline(
        image_size=224, 
        device=device, 
        include_bbox=True
    )
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = WFLWDataset(WFLW_TRAIN_ANNOTATION, WFLW_IMG_DIR, transform=transform)
    test_dataset = WFLWDataset(WFLW_TEST_ANNOTATION, WFLW_IMG_DIR, transform=transform)
    
    # 可视化原始数据集并保存
    print(f"可视化原始数据集，图像保存至: {WFLW_VIS_DIR}")
    raw_dataset = WFLWDataset(WFLW_TRAIN_ANNOTATION, WFLW_IMG_DIR, transform=None)
    visualize_wflw_dataset(raw_dataset, num_samples=3, save_dir=WFLW_VIS_DIR)
    
    # 数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, drop_last=True)
    
    print(f"训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")
    
    # 模型
    model = build_model(num_landmarks=98, pretrained=True).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 3
    print(f"\n开始训练，共 {num_epochs} 个epoch...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        print(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {train_loss:.4f}")
    
    # 评估
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"测试集平均损失: {test_loss:.4f}")
    
    # 可视化预测并保存
    print(f"可视化预测结果，图像保存至: {PREDICTIONS_DIR}")
    visualize_predictions(model, test_dataset, device, num_samples=6, 
                         image_size=224, save_dir=PREDICTIONS_DIR)
    
    # 保存模型
    torch.save(model, WFLW_MODEL_PATH)
    print(f"WFLW模型已保存至: {WFLW_MODEL_PATH}")


if __name__ == "__main__":
    print("=" * 50)
    print("人脸关键点检测训练程序")
    print("=" * 50)
    print(f"当前文件路径: {CURRENT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"5点数据集目录: {FIVEPOINT_DATA_DIR}")
    print(f"WFLW数据集目录: {WFLW_DATA_DIR}")
    print(f"模型保存目录: {MODEL_SAVE_DIR}")
    print("=" * 50)
    
    # 训练5点模型
    train_5point_model()
    
    # 训练WFLW 98点模型
    train_wflw_model()
    
    print("\n✅ 所有训练完成！")
    print(f"📊 可视化结果已保存至: {OUTPUT_DIR}")
    print(f"├── 5点数据集: {FIVEPOINT_VIS_DIR}")
    print(f"├── WFLW数据集: {WFLW_VIS_DIR}")
    print(f"├── WFLW分析: {WFLW_ANALYSIS_DIR}")
    print(f"└── 预测结果: {PREDICTIONS_DIR}")
    print(f"💾 模型已保存至: {MODEL_SAVE_DIR}")