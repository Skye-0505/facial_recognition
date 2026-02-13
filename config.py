"""
项目全局配置文件
"""
import os
import torch

class Config:
    """配置类，集中管理所有路径和超参数"""
    
    # ==================== 基础路径配置 ====================
    # 项目根目录（自动检测）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据相关目录
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATASET_DIR = os.path.join(DATA_DIR, 'DATASET')      # RAF-DB数据集
    TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_img')    # 测试图片
    ICON_DIR = os.path.join(DATA_DIR, 'icons')           # 表情图标
    MODEL_DIR = os.path.join(DATA_DIR, 'models')         # 预训练模型
    
    # 输出目录
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    
    # ==================== 硬件设备配置 ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 数据集配置 ====================
    # RAF-DB 表情类别（7类）
    EMOTION_LABELS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
    
    # 图像尺寸（与ResNet/ViT输入保持一致）
    IMAGE_SIZE = (224, 224)
    
    # 图像归一化参数（使用你计算的RAF-DB统计值）
    NORMALIZE_MEAN = [0.5752, 0.4495, 0.4012]
    NORMALIZE_STD = [0.2086, 0.1911, 0.1827]
    
    # ==================== 模型文件路径 ====================
    MODEL_PATHS = {
        # 表情识别模型
        'fer_resnet18': os.path.join(MODEL_DIR, 'fer_resnet18.pth'),
        
        # 人脸关键点检测模型
        'landmark_wflw': os.path.join(MODEL_DIR, 'wflw_model.pth'),
    }
    
    # ==================== 训练超参数 ====================
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 20
    
    # Focal Loss参数
    FOCAL_LOSS_GAMMA = 2.0
    
    # ==================== 数据增强配置 ====================
    # 随机擦除参数（与你的设置一致）
    RANDOM_ERASING_SCALE = (0.02, 0.25)
    
    # ==================== 人脸检测配置 ====================
    # MTCNN参数
    MTCNN_MIN_FACE_SIZE = 40
    MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]
    
    # ==================== 贴纸合成配置 ====================
    # 各表情的贴纸缩放系数（相对于眼睛宽度）
    STICKER_SCALE_FACTORS = {
        'surprise': 0.9,    # 惊讶：较大
        'fear': 0.7,        # 恐惧：中等偏小
        'disgust': 0.6,     # 厌恶：较小
        'happy': 0.5,       # 快乐：较小（有两个）
        'sad': 0.4,         # 悲伤：小（眼泪）
        'anger': 0.8,       # 愤怒：中等
        'neutral': 0.7,     # 中性：中等
    }
    
    # 贴图标点映射（WFLW 98点关键点索引）
    LANDMARK_INDICES = {
        'LEFT_EYE_LEFT': 59,
        'LEFT_EYE_RIGHT': 62,
        'RIGHT_EYE_LEFT': 68,
        'RIGHT_EYE_RIGHT': 71,
        'LEFT_EYE_CENTER': 73,
        'RIGHT_EYE_CENTER': 78,
        'LEFT_EYEBROW_CENTER': 70,
        'RIGHT_EYEBROW_CENTER': 75,
        'NOSE_TIP': 86,
        'MOUTH_LEFT': 95,
        'MOUTH_RIGHT': 96,
        'MOUTH_UPPER_CENTER': 97,
    }
    
    # ==================== 工具方法 ====================    
    @classmethod
    def get_train_transform(cls):
        """获取训练数据变换（从lab1.py迁移）"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize(cls.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cls.NORMALIZE_MEAN, cls.NORMALIZE_STD),
            transforms.RandomErasing(scale=cls.RANDOM_ERASING_SCALE)
        ])
    
    @classmethod
    def get_test_transform(cls):
        """获取测试数据变换（从lab1.py迁移）"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize(cls.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(cls.NORMALIZE_MEAN, cls.NORMALIZE_STD)
        ])
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 50)
        print("项目配置信息")
        print("=" * 50)
        print(f"设备: {cls.DEVICE}")
        print(f"表情类别: {cls.EMOTION_LABELS}")
        print(f"图像尺寸: {cls.IMAGE_SIZE}")
        print(f"批大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print("=" * 50)


# 方便导入：在其他文件中可以使用 from config import cfg
cfg = Config()