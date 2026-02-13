"""
人脸表情识别与特效合成系统
功能：识别面部表情，检测98个关键点，在面部特定位置添加表情图标
结果自动保存到 outputs/application 目录
"""

import os
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免显示问题
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from typing import Optional

# ==================== 路径配置（绝对路径） ====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型文件路径
FER_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '../emotion/fer_resnet18_best.pth'))
LANDMARK_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '../landmarks/wflw_model.pth'))

# 图标目录路径
ICON_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/application/icons'))

# 测试图片目录路径
TEST_IMG_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/application/test_img'))

# 输出目录路径（保存结果）
OUTPUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../outputs/application'))
RESULT_IMG_DIR = os.path.join(OUTPUT_DIR, 'images')      # 结果图片保存目录
RESULT_VIS_DIR = os.path.join(OUTPUT_DIR, 'visualization')  # 对比图保存目录
RESULT_LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')        # 日志保存目录

# ==================== 全局配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

# 表情标签（与lab1训练顺序一致）
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 表情特效缩放系数（基于眼睛宽度）
EMOTION_SCALE_FACTORS = {
    'angry': 0.8,      # 愤怒：中等大小
    'disgust': 0.6,    # 厌恶：较小
    'fear': 0.7,       # 恐惧：中等偏小
    'happy': 0.5,      # 快乐：较小（两个图标）
    'sad': 0.4,        # 悲伤：小（眼泪）
    'surprise': 0.9,   # 惊讶：较大
    'neutral': 0.7,    # 中性：中等
}

# 图像预处理参数（与lab1保持一致）
FER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5752, 0.4495, 0.4012],
        std=[0.2086, 0.1911, 0.1827]
    )
])

# ==================== 工具函数 ====================

def ensure_dir_exists(dir_path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    return dir_path

def check_file_exists(file_path, description=""):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description}文件不存在: {file_path}")
    return True

def get_timestamp():
    """获取当前时间戳字符串（用于文件名）"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def save_result_image(image, filename, subdir='images'):
    """
    保存结果图片到指定子目录
    Args:
        image: 要保存的图像（BGR格式）
        filename: 文件名
        subdir: 子目录名称（images/visualization/logs）
    Returns:
        保存的完整路径
    """
    save_dir = ensure_dir_exists(os.path.join(OUTPUT_DIR, subdir))
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"💾 图片已保存: {save_path}")
    return save_path

# ==================== 模型定义（与训练代码一致） ====================

class BasicBlock(nn.Module):
    """ResNet基础块 - 与训练代码保持一致"""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet架构 - 与训练代码保持一致"""
    
    def __init__(self, block: nn.Module, num_blocks: list, num_classes: int = 7):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """创建残差层"""
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def create_resnet18(num_classes: int = 7):
    """创建ResNet-18模型（用于推理）"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model

# ==================== 模型加载（修复版） ====================

class ModelLoader:
    """模型加载器，负责加载表情识别和关键点检测模型"""
    
    @staticmethod
    def load_fer_model():
        """加载表情识别模型（ResNet18）- 修复checkpoint加载问题"""
        check_file_exists(FER_MODEL_PATH, "表情识别模型")
        
        # 1. 创建模型实例
        model = create_resnet18(num_classes=7)
        
        # 2. 加载checkpoint
        checkpoint = torch.load(FER_MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # 3. 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # 格式1: 完整checkpoint（包含epoch, optimizer等）
                print("📦 检测到完整checkpoint格式")
                state_dict = checkpoint['model_state_dict']
                
                # 打印模型信息（如果有）
                if 'val_acc' in checkpoint:
                    print(f"  最佳验证准确率: {checkpoint['val_acc']:.2f}%")
                if 'epoch' in checkpoint:
                    print(f"  训练轮次: {checkpoint['epoch']}")
                if 'config' in checkpoint:
                    print(f"  模型配置: {checkpoint['config']}")
                    
            elif 'state_dict' in checkpoint:
                # 格式2: 包含'state_dict'键的checkpoint
                print("📦 检测到state_dict格式")
                state_dict = checkpoint['state_dict']
            else:
                # 格式3: 直接是state_dict
                print("📦 检测到直接state_dict格式")
                state_dict = checkpoint
        else:
            # 格式4: 直接是模型对象（极少见）
            print("📦 检测到直接模型对象格式")
            model = checkpoint
            model = model.to(DEVICE)
            model.eval()
            print("✅ 表情识别模型加载完成")
            return model
        
        # 4. 处理state_dict中的键名（移除'module.'前缀等）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # DataParallel保存的模型
                new_key = k[7:]  # 移除'module.'前缀
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        # 5. 加载权重
        try:
            model.load_state_dict(new_state_dict)
            print("✅ 模型权重加载成功")
        except Exception as e:
            print(f"⚠️ 直接加载失败，尝试非严格模式: {e}")
            # 尝试非严格模式
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"  缺失的键: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  意外的键: {unexpected_keys[:5]}...")
            print("✅ 模型权重以非严格模式加载成功")
        
        model = model.to(DEVICE)
        model.eval()
        
        print(f"✅ 表情识别模型加载完成: {FER_MODEL_PATH}")
        return model
    
    @staticmethod
    def load_landmark_model():
        """加载人脸关键点检测模型（WFLW 98点）"""
        check_file_exists(LANDMARK_MODEL_PATH, "关键点检测模型")
        
        # 关键点模型通常是完整的模型对象，直接加载
        try:
            model = torch.load(LANDMARK_MODEL_PATH, map_location=DEVICE, weights_only=False)
            
            # 如果是dict，尝试提取state_dict
            if isinstance(model, dict):
                if 'model_state_dict' in model:
                    model = model['model_state_dict']
                elif 'state_dict' in model:
                    model = model['state_dict']
            
            # 确保是nn.Module
            if not isinstance(model, nn.Module):
                print(f"⚠️ 模型类型: {type(model)}，尝试包装...")
                # 如果加载出来不是nn.Module，需要根据实际情况处理
                # 这里假设是state_dict，需要创建模型实例
                from src.landmark.models import create_model  # 需要根据实际情况导入
                model_instance = create_model()  # 创建你的关键点模型
                model_instance.load_state_dict(model)
                model = model_instance
            
            model = model.to(DEVICE)
            model.eval()
            print(f"✅ 关键点检测模型加载完成: {LANDMARK_MODEL_PATH}")
            return model
            
        except Exception as e:
            print(f"❌ 关键点模型加载失败: {e}")
            raise
    
    @classmethod
    def load_all(cls):
        """加载所有模型"""
        fer_model = cls.load_fer_model()
        landmark_model = cls.load_landmark_model()
        return fer_model, landmark_model

# ==================== 图标管理 ====================

class IconManager:
    """特效图标管理器，负责加载和管理表情图标"""
    
    def __init__(self, icon_dir=ICON_DIR):
        self.icon_dir = icon_dir
        self.emotion_icons = {}
        self._load_icons()
    
    def _load_icons(self):
        """加载所有表情图标"""
        ensure_dir_exists(self.icon_dir)
        
        for emotion in EMOTION_LABELS:
            icon_path = os.path.join(self.icon_dir, f"{emotion}.png")
            if os.path.exists(icon_path):
                try:
                    # 加载图标并转换为RGBA格式
                    icon_pil = Image.open(icon_path).convert('RGBA')
                    icon_cv = cv2.cvtColor(np.array(icon_pil), cv2.COLOR_RGBA2BGRA)
                    
                    self.emotion_icons[emotion] = {
                        'icon': icon_cv,
                        'size': icon_cv.shape[:2],  # (height, width)
                        'channels': icon_cv.shape[2]
                    }
                    print(f"✅ 加载图标: {emotion}")
                except Exception as e:
                    print(f"❌ 图标加载失败 {emotion}: {e}")
                    self.emotion_icons[emotion] = None
            else:
                print(f"⚠️ 图标不存在: {icon_path}")
                self.emotion_icons[emotion] = None
    
    def get_icon(self, emotion):
        """获取指定表情的图标"""
        if emotion not in self.emotion_icons:
            print(f"❌ 未找到表情 {emotion} 的图标")
            return None
        return self.emotion_icons[emotion]
    
    def has_icon(self, emotion):
        """检查是否存在指定表情的图标"""
        return emotion in self.emotion_icons and self.emotion_icons[emotion] is not None

# ==================== 人脸检测 ====================

class FaceDetector:
    """人脸检测器，封装MTCNN"""
    
    def __init__(self, device='cpu'):
        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
    
    def detect_single_face(self, image_path):
        """
        检测单张人脸，返回边界框 [x1, y1, x2, y2]
        仅支持单人图片，多人脸返回None
        """
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            return None
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None
        
        h, w = img.shape[:2]
        
        # 转换为RGB（MTCNN需要）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, probs, landmarks = self.mtcnn.detect(img_rgb, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                print("❌ 未检测到人脸")
                return None
            
            if len(boxes) > 1:
                print(f"❌ 检测到 {len(boxes)} 张人脸，当前系统仅支持单人图片")
                return None
            
            # 获取并调整边界框
            x1, y1, x2, y2 = boxes[0].astype(int)
            bbox = [
                max(0, x1),
                max(0, y1),
                min(w, x2),
                min(h, y2)
            ]
            
            return bbox
            
        except Exception as e:
            print(f"❌ MTCNN检测出错: {e}")
            return None

# ==================== 表情识别 ====================

class EmotionRecognizer:
    """表情识别器"""
    
    def __init__(self, model):
        self.model = model
        self.labels = EMOTION_LABELS
        self.transform = FER_TRANSFORM
    
    def recognize(self, image_path):
        """
        识别图片中的表情
        返回: (emotion, confidence)
        """
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            return None, 0.0
        
        try:
            # 加载并预处理图片
            pil_image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(outputs, dim=1).item()
            
            emotion = self.labels[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            
            return emotion, confidence
            
        except Exception as e:
            print(f"❌ 表情识别失败: {e}")
            return None, 0.0

# ==================== 关键点检测 ====================

class LandmarkDetector:
    """人脸关键点检测器（WFLW 98点）"""
    
    def __init__(self, model):
        self.model = model
        self.num_points = 98
    
    def preprocess_face(self, image_path, bbox):
        """预处理人脸区域用于关键点检测"""
        if not os.path.exists(image_path):
            return None
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            return None
        
        h, w = original_image.shape[:2]
        
        # 调整bbox到有效范围
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        bbox_actual = [x1, y1, x2, y2]
        
        # 裁剪并调整大小
        cropped_face = original_image[y1:y2, x1:x2]
        face_resized = cv2.resize(cropped_face, (224, 224))
        
        # 转换为tensor并归一化
        face_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        
        return {
            'original_image': original_image,
            'cropped_face': cropped_face,
            'face_tensor': face_tensor,
            'bbox_actual': bbox_actual,
            'crop_size': (x2 - x1, y2 - y1)
        }
    
    def predict(self, face_tensor):
        """预测98个关键点"""
        if face_tensor is None:
            return None, None
        
        try:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(face_tensor)
                
                if outputs.shape[1] == 196:
                    landmarks_tensor = outputs[0].reshape(self.num_points, 2)
                else:
                    return None, None
            
            landmarks_np = landmarks_tensor.cpu().numpy()
            landmarks_normalized = landmarks_np
            landmarks_pixel = landmarks_normalized * 224
            
            return landmarks_normalized, landmarks_pixel
            
        except Exception as e:
            print(f"❌ 关键点预测失败: {e}")
            return None, None
    
    @staticmethod
    def map_to_original(landmarks_normalized, bbox_actual, original_size):
        """将归一化关键点坐标映射回原图"""
        if landmarks_normalized is None or bbox_actual is None:
            return None
        
        x1, y1, x2, y2 = bbox_actual
        w_orig, h_orig = original_size
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        num_points = landmarks_normalized.shape[0]
        landmarks_original = np.zeros((num_points, 2), dtype=np.float32)
        
        for i in range(num_points):
            x_norm, y_norm = landmarks_normalized[i]
            landmarks_original[i] = [
                x1 + x_norm * crop_w,
                y1 + y_norm * crop_h
            ]
        
        return landmarks_original


# ==================== 关键点索引定义 ====================

class WFLWIndices:
    """WFLW 98个关键点的索引定义"""
    LEFT_EYE_LEFT = 60      # 左眼左角（修正）
    LEFT_EYE_RIGHT = 63     # 左眼右角（修正）
    RIGHT_EYE_LEFT = 68
    RIGHT_EYE_RIGHT = 71
    LEFT_EYE_CENTER = 65    # 左眼中心（修正）
    RIGHT_EYE_CENTER = 78
    LEFT_EYEBROW_CENTER = 70
    RIGHT_EYEBROW_CENTER = 75
    NOSE_TIP = 86
    MOUTH_LEFT = 95
    MOUTH_RIGHT = 96
    MOUTH_UPPER_CENTER = 97
    
    @classmethod
    def get_all_indices(cls):
        return {
            'LEFT_EYE_LEFT': cls.LEFT_EYE_LEFT,
            'LEFT_EYE_RIGHT': cls.LEFT_EYE_RIGHT,
            'RIGHT_EYE_LEFT': cls.RIGHT_EYE_LEFT,
            'RIGHT_EYE_RIGHT': cls.RIGHT_EYE_RIGHT,
            'LEFT_EYE_CENTER': cls.LEFT_EYE_CENTER,
            'RIGHT_EYE_CENTER': cls.RIGHT_EYE_CENTER,
            'LEFT_EYEBROW_CENTER': cls.LEFT_EYEBROW_CENTER,
            'RIGHT_EYEBROW_CENTER': cls.RIGHT_EYEBROW_CENTER,
            'NOSE_TIP': cls.NOSE_TIP,
            'MOUTH_LEFT': cls.MOUTH_LEFT,
            'MOUTH_RIGHT': cls.MOUTH_RIGHT,
            'MOUTH_UPPER_CENTER': cls.MOUTH_UPPER_CENTER,
        }


# ==================== 特效位置计算 ====================

class EffectPositionCalculator:
    """特效位置计算器"""
    
    def __init__(self, landmarks_original, emotion, icon_size):
        self.landmarks = landmarks_original
        self.emotion = emotion
        self.icon_h, self.icon_w = icon_size[:2]
        self.indices = WFLWIndices.get_all_indices()
    
    def calculate_eye_size(self):
        """计算眼睛的平均宽度"""
        left_eye_left = self.landmarks[self.indices['LEFT_EYE_LEFT']]
        left_eye_right = self.landmarks[self.indices['LEFT_EYE_RIGHT']]
        right_eye_left = self.landmarks[self.indices['RIGHT_EYE_LEFT']]
        right_eye_right = self.landmarks[self.indices['RIGHT_EYE_RIGHT']]
        
        left_eye_width = np.linalg.norm(left_eye_right - left_eye_left)
        right_eye_width = np.linalg.norm(right_eye_right - right_eye_left)
        eye_width = (left_eye_width + right_eye_width) / 2
        return eye_width, left_eye_width, right_eye_width
    
    def calculate_scale(self, eye_width):
        """根据眼睛尺寸和表情类型计算缩放比例"""
        factor = EMOTION_SCALE_FACTORS.get(self.emotion, 0.7)
        target_width = eye_width * factor
        scale = target_width / self.icon_w
        scale = max(0.2, min(2.0, scale))
        return scale, target_width
    
    def _calculate_angry_position(self, scale):
        right_brow = self.landmarks[self.indices['RIGHT_EYEBROW_CENTER']]
        icon_w_scaled = int(self.icon_w * scale)
        icon_h_scaled = int(self.icon_h * scale)
        return {
            'x': int(right_brow[0] - icon_w_scaled / 2),
            'y': int(right_brow[1] - icon_h_scaled - 5),
            'width': icon_w_scaled,
            'height': icon_h_scaled,
            'scale': scale,
            'anchor_point': (int(right_brow[0]), int(right_brow[1])),
            'emotion': self.emotion
        }
    
    def _calculate_disgust_position(self, scale):
        mouth_right = self.landmarks[self.indices['MOUTH_RIGHT']]
        icon_w_scaled = int(self.icon_w * scale)
        icon_h_scaled = int(self.icon_h * scale)
        return {
            'x': int(mouth_right[0] - icon_w_scaled / 2),
            'y': int(mouth_right[1] + 10),
            'width': icon_w_scaled,
            'height': icon_h_scaled,
            'scale': scale,
            'anchor_point': (int(mouth_right[0]), int(mouth_right[1])),
            'emotion': self.emotion
        }
    
    def _calculate_fear_position(self, scale):
        right_brow = self.landmarks[self.indices['RIGHT_EYEBROW_CENTER']]
        icon_w_scaled = int(self.icon_w * scale)
        icon_h_scaled = int(self.icon_h * scale)
        return {
            'x': int(right_brow[0] - icon_w_scaled / 2 + 5),
            'y': int(right_brow[1] - icon_h_scaled - 40),
            'width': icon_w_scaled,
            'height': icon_h_scaled,
            'scale': scale,
            'anchor_point': (int(right_brow[0]), int(right_brow[1])),
            'emotion': self.emotion
        }
    
    def _calculate_happy_positions(self, scale, eye_width):
        scale_small = scale * 0.5
        icon_w_small = int(self.icon_w * scale_small)
        icon_h_small = int(self.icon_h * scale_small)
        left_eye_outer = self.landmarks[self.indices['LEFT_EYE_LEFT']]
        right_eye_outer = self.landmarks[self.indices['RIGHT_EYE_RIGHT']]
        vertical_offset = int(icon_h_small * 0.3)
        return [
            {
                'x': int(left_eye_outer[0] - icon_w_small - 5),
                'y': int(left_eye_outer[1] - icon_h_small // 2 + vertical_offset),
                'width': icon_w_small,
                'height': icon_h_small,
                'scale': scale_small,
                'anchor_point': (int(left_eye_outer[0]), int(left_eye_outer[1])),
                'emotion': f'{self.emotion}_left'
            },
            {
                'x': int(right_eye_outer[0] + 5),
                'y': int(right_eye_outer[1] - icon_h_small // 2 + vertical_offset),
                'width': icon_w_small,
                'height': icon_h_small,
                'scale': scale_small,
                'anchor_point': (int(right_eye_outer[0]), int(right_eye_outer[1])),
                'emotion': f'{self.emotion}_right'
            }
        ]
    def _calculate_sad_positions(self, scale):
        """悲伤：两个眼泪图标，放在眼睛外眼角下方"""
        scale_small = scale * 0.4
        icon_w_small = int(self.icon_w * scale_small)
        icon_h_small = int(self.icon_h * scale_small)
        
        # ✅ 使用已经验证正确的外眼角索引
        left_eye_outer = self.landmarks[self.indices['LEFT_EYE_LEFT']]      # 60
        right_eye_outer = self.landmarks[self.indices['RIGHT_EYE_RIGHT']]  # 71
        
        return [
            {
                'x': int(left_eye_outer[0] - icon_w_small / 2),
                'y': int(left_eye_outer[1] + 8),  # 外眼角下方
                'width': icon_w_small,
                'height': icon_h_small,
                'scale': scale_small,
                'anchor_point': (int(left_eye_outer[0]), int(left_eye_outer[1])),
                'emotion': f'{self.emotion}_left'
            },
            {
                'x': int(right_eye_outer[0] - icon_w_small / 2),
                'y': int(right_eye_outer[1] + 8),  # 外眼角下方
                'width': icon_w_small,
                'height': icon_h_small,
                'scale': scale_small,
                'anchor_point': (int(right_eye_outer[0]), int(right_eye_outer[1])),
                'emotion': f'{self.emotion}_right'
            }
        ]
        
    def _calculate_surprise_position(self, scale):
        right_brow = self.landmarks[self.indices['RIGHT_EYEBROW_CENTER']]
        icon_w_scaled = int(self.icon_w * scale)
        icon_h_scaled = int(self.icon_h * scale)
        return {
            'x': int(right_brow[0] - icon_w_scaled / 2),
            'y': int(right_brow[1] - icon_h_scaled - 60),
            'width': icon_w_scaled,
            'height': icon_h_scaled,
            'scale': scale,
            'anchor_point': (int(right_brow[0]), int(right_brow[1])),
            'emotion': self.emotion
        }
    
    def _calculate_neutral_position(self, scale):
        nose_tip = self.landmarks[self.indices['NOSE_TIP']]
        icon_w_scaled = int(self.icon_w * scale)
        icon_h_scaled = int(self.icon_h * scale)
        return {
            'x': int(nose_tip[0] - icon_w_scaled - 100),
            'y': int(nose_tip[1] - icon_h_scaled),
            'width': icon_w_scaled,
            'height': icon_h_scaled,
            'scale': scale,
            'anchor_point': (int(nose_tip[0]), int(nose_tip[1])),
            'emotion': self.emotion
        }
    
    def calculate_positions(self):
        """主方法：根据表情计算特效位置"""
        if self.landmarks is None or self.landmarks.shape[0] < 98:
            return None
        
        eye_width, _, _ = self.calculate_eye_size()
        scale, _ = self.calculate_scale(eye_width)
        
        if self.emotion == 'angry':
            return self._calculate_angry_position(scale)
        elif self.emotion == 'disgust':
            return self._calculate_disgust_position(scale)
        elif self.emotion == 'fear':
            return self._calculate_fear_position(scale)
        elif self.emotion == 'happy':
            return self._calculate_happy_positions(scale, eye_width)
        elif self.emotion == 'sad':
            return self._calculate_sad_positions(scale)
        elif self.emotion == 'surprise':
            return self._calculate_surprise_position(scale)
        elif self.emotion == 'neutral':
            return self._calculate_neutral_position(scale)
        else:
            return None


# ==================== 特效合成器 ====================

class EffectCompositor:
    """特效合成器"""
    
    @staticmethod
    def blend_icon(image, icon, position):
        if image is None or icon is None or position is None:
            return image
        
        result = image.copy()
        positions = [position] if isinstance(position, dict) else position
        
        for pos in positions:
            try:
                x = pos['x']
                y = pos['y']
                width = pos['width']
                height = pos['height']
                
                h_orig, w_orig = result.shape[:2]
                
                icon_resized = cv2.resize(icon, (width, height), interpolation=cv2.INTER_LINEAR)
                icon_h, icon_w = icon_resized.shape[:2]
                
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(w_orig, x + icon_w)
                y_end = min(h_orig, y + icon_h)
                
                icon_x_start = max(0, -x)
                icon_y_start = max(0, -y)
                icon_x_end = icon_w - max(0, x + icon_w - w_orig)
                icon_y_end = icon_h - max(0, y + icon_h - h_orig)
                
                if x_end <= x_start or y_end <= y_start:
                    continue
                
                if icon_x_end <= icon_x_start or icon_y_end <= icon_y_start:
                    continue
                
                if icon_resized.shape[2] == 4:
                    icon_rgb = icon_resized[:, :, :3]
                    icon_alpha = icon_resized[:, :, 3] / 255.0
                else:
                    icon_rgb = icon_resized
                    icon_alpha = np.ones((icon_h, icon_w))
                
                valid_icon_rgb = icon_rgb[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
                valid_icon_alpha = icon_alpha[icon_y_start:icon_y_end, icon_x_start:icon_x_end]
                target_area = result[y_start:y_end, x_start:x_end]
                
                if target_area.shape[:2] != valid_icon_rgb.shape[:2]:
                    if target_area.shape[0] > 0 and target_area.shape[1] > 0:
                        valid_icon_rgb = cv2.resize(valid_icon_rgb, (target_area.shape[1], target_area.shape[0]))
                        valid_icon_alpha = cv2.resize(valid_icon_alpha, (target_area.shape[1], target_area.shape[0]))
                
                alpha_expanded = valid_icon_alpha[:, :, np.newaxis]
                blended = valid_icon_rgb * alpha_expanded + target_area * (1 - alpha_expanded)
                result[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
                
            except Exception as e:
                print(f"  ⚠️ 合成失败: {e}")
                continue
        
        return result


# ==================== 结果保存器 ====================

class ResultSaver:
    """结果保存器"""
    
    @staticmethod
    def save_result_image(image, original_path, emotion, confidence):
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        timestamp = get_timestamp()
        filename = f"{base_name}_{emotion}_{confidence:.2f}_{timestamp}.jpg"
        return save_result_image(image, filename, 'images')
    
    @staticmethod
    def save_comparison(original_path, result_image, emotion, confidence, info):
        original_img = cv2.imread(original_path)
        if original_img is None:
            return None
        
        h1, w1 = original_img.shape[:2]
        h2, w2 = result_image.shape[:2]
        target_height = max(h1, h2)
        
        if h1 != target_height:
            scale = target_height / h1
            original_img = cv2.resize(original_img, (int(w1 * scale), target_height))
        if h2 != target_height:
            scale = target_height / h2
            result_image = cv2.resize(result_image, (int(w2 * scale), target_height))
        
        comparison = np.hstack([original_img, result_image])
        
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        timestamp = get_timestamp()
        filename = f"{base_name}_{emotion}_comparison_{timestamp}.jpg"
        return save_result_image(comparison, filename, 'visualization')
    
    @staticmethod
    def save_log(results):
        timestamp = get_timestamp()
        log_file = os.path.join(ensure_dir_exists(RESULT_LOG_DIR), f"processing_log_{timestamp}.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"处理日志 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            
            success_count = 0
            for i, (img_path, emotion, confidence, info) in enumerate(results, 1):
                status = "✅" if info.get('success') else "❌"
                f.write(f"{i}. {status} {os.path.basename(img_path)} - {emotion} ({confidence:.2%})\n")
                if info.get('success'):
                    success_count += 1
            
            f.write("=" * 50 + "\n")
            f.write(f"总计: {len(results)} 成功: {success_count} 失败: {len(results)-success_count}\n")
        
        return log_file


# ==================== 主处理流水线 ====================

class FaceEffectPipeline:
    """人脸特效处理流水线"""
    
    def __init__(self, fer_model, landmark_model, icon_manager):
        self.face_detector = FaceDetector(device='cpu')
        self.emotion_recognizer = EmotionRecognizer(fer_model)
        self.landmark_detector = LandmarkDetector(landmark_model)
        self.icon_manager = icon_manager
        self.compositor = EffectCompositor()
        self.result_saver = ResultSaver()
    
    def process(self, image_path, save_results=True):
        info = {
            'image_path': image_path,
            'success': False,
            'emotion': None,
            'confidence': 0.0,
            'error': None,
            'processing_time': 0.0
        }
        saved_paths = {}
        
        start_time = time.time()
        
        try:
            # 1. 人脸检测
            bbox = self.face_detector.detect_single_face(image_path)
            if bbox is None:
                info['error'] = "人脸检测失败"
                return None, None, info, saved_paths
            
            # 2. 预处理
            preprocess_result = self.landmark_detector.preprocess_face(image_path, bbox)
            if preprocess_result is None:
                info['error'] = "人脸预处理失败"
                return None, None, info, saved_paths
            
            # 3. 表情识别
            temp_face_path = os.path.join(CURRENT_DIR, "temp_face_crop.jpg")
            cv2.imwrite(temp_face_path, preprocess_result['cropped_face'])
            emotion, confidence = self.emotion_recognizer.recognize(temp_face_path)
            
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)
            
            if emotion is None:
                info['error'] = "表情识别失败"
                return None, None, info, saved_paths
            
            info['emotion'] = emotion
            info['confidence'] = confidence
            
            # 4. 关键点检测
            landmarks_norm, _ = self.landmark_detector.predict(preprocess_result['face_tensor'])
            if landmarks_norm is None:
                info['error'] = "关键点检测失败"
                return None, None, info, saved_paths
            
            # 5. 坐标映射
            h_orig, w_orig = preprocess_result['original_image'].shape[:2]
            landmarks_original = self.landmark_detector.map_to_original(
                landmarks_norm,
                preprocess_result['bbox_actual'],
                (w_orig, h_orig)
            )
            
            if landmarks_original is None:
                info['error'] = "坐标映射失败"
                return None, None, info, saved_paths
            
            # 6. 获取图标
            if not self.icon_manager.has_icon(emotion):
                info['error'] = f"无 {emotion} 对应的图标"
                return None, None, info, saved_paths
            
            icon_data = self.icon_manager.get_icon(emotion)
            
            # 7. 计算位置
            position_calculator = EffectPositionCalculator(
                landmarks_original,
                emotion,
                icon_data['size']
            )
            positions = position_calculator.calculate_positions()
            
            if positions is None:
                info['error'] = "位置计算失败"
                return None, None, info, saved_paths
            
            # 8. 合成特效
            result_image = self.compositor.blend_icon(
                preprocess_result['original_image'].copy(),
                icon_data['icon'],
                positions
            )
            
            if result_image is None:
                info['error'] = "特效合成失败"
                return None, None, info, saved_paths
            
            info['success'] = True
            info['num_landmarks'] = landmarks_norm.shape[0]
            info['processing_time'] = time.time() - start_time
            
            # 9. 保存结果
            if save_results:
                result_path = self.result_saver.save_result_image(
                    result_image, image_path, emotion, confidence
                )
                saved_paths['result_image'] = result_path
                
                comparison_path = self.result_saver.save_comparison(
                    image_path, result_image, emotion, confidence, info
                )
                saved_paths['comparison'] = comparison_path
                
                print(f"\n📸 处理完成: {os.path.basename(image_path)}")
                print(f"   😀 表情: {emotion} (置信度: {confidence:.1%})")
                print(f"   ⏱️  耗时: {info['processing_time']:.2f}s")
            
            return result_image, emotion, info, saved_paths
            
        except Exception as e:
            info['error'] = str(e)
            info['processing_time'] = time.time() - start_time
            return None, None, info, saved_paths


# ==================== 批处理器 ====================

class BatchProcessor:
    """批处理系统"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = []
        self.result_saver = ResultSaver() 
    
    def get_image_files(self, directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        if not os.path.exists(directory):
            return []
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)
        
        return sorted(image_files)
    def process_all(self, test_dir=TEST_IMG_DIR):
        print("\n" + "=" * 60)
        print("🚀 开始批量处理人脸表情特效")
        print(f"📁 测试目录: {test_dir}")
        print("=" * 60 + "\n")
        
        image_files = self.get_image_files(test_dir)
        
        if not image_files:
            print(f"❌ 未找到图片文件: {test_dir}")
            return
        
        print(f"📸 共找到 {len(image_files)} 张图片\n")
        
        ensure_dir_exists(RESULT_IMG_DIR)
        ensure_dir_exists(RESULT_VIS_DIR)
        ensure_dir_exists(RESULT_LOG_DIR)
        
        successful_count = 0
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理: {os.path.basename(img_path)}")
            print("-" * 50)
            
            try:
                # ✅ pipeline只负责处理，不保存
                result_img, emotion, info, saved_paths = self.pipeline.process(
                    img_path, save_results=False
                )
                
                # ✅ BatchProcessor统一负责保存
                if info.get('success') and result_img is not None:
                    # 保存结果图片
                    result_path = self.result_saver.save_result_image(
                        result_img, img_path, emotion, info.get('confidence', 0)
                    )
                    # 保存对比图
                    comparison_path = self.result_saver.save_comparison(
                        img_path, result_img, emotion, info.get('confidence', 0), info
                    )
                    
                    print(f"\n📸 处理完成: {os.path.basename(img_path)}")
                    print(f"   😀 表情: {emotion} (置信度: {info.get('confidence', 0):.1%})")
                    print(f"   ⏱️  耗时: {info['processing_time']:.2f}s")
                    print(f"   💾 结果: {result_path}")
                    print(f"   🖼️  对比: {comparison_path}")
                else:
                    print(f"   ❌ 失败: {info.get('error', '未知错误')}")
                
                self.results.append((img_path, emotion, info.get('confidence', 0), info))
                
                if info.get('success'):
                    successful_count += 1
                
                print()
                
            except Exception as e:
                print(f"   ❌ 异常: {e}\n")
                self.results.append((img_path, None, 0.0, {'success': False, 'error': str(e)}))
        
        # 保存日志
        if self.results:
            self.result_saver.save_log(self.results)
        
        print("\n" + "=" * 60)
        print(f"📊 处理完成: 总计 {len(image_files)} 张, 成功 {successful_count} 张, 失败 {len(image_files)-successful_count} 张")
        print(f"💾 结果保存目录: {OUTPUT_DIR}")
        print("=" * 60 + "\n")
# ==================== 主程序 ====================

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🎭 人脸表情识别与特效合成系统")
    print("=" * 60)
    
    try:
        # 1. 加载模型
        print("\n📦 加载模型中...")
        fer_model, landmark_model = ModelLoader.load_all()
        
        # 2. 加载图标
        print("\n🖼️  加载表情图标...")
        icon_manager = IconManager()
        
        # 3. 创建处理流水线
        print("\n🔧 初始化处理流水线...")
        pipeline = FaceEffectPipeline(fer_model, landmark_model, icon_manager)
        
        # 4. 创建批处理器并运行
        print("\n🚀 启动批处理系统...")
        processor = BatchProcessor(pipeline)
        processor.process_all()
        
        print("\n✨ 所有任务处理完成！")
        
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 确保必要的目录存在
    ensure_dir_exists(OUTPUT_DIR)
    ensure_dir_exists(RESULT_IMG_DIR)
    ensure_dir_exists(RESULT_VIS_DIR)
    ensure_dir_exists(RESULT_LOG_DIR)
    
    # 运行主程序
    main()