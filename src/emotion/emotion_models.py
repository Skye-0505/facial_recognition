# lab1_optimized_fixed.py
"""
面部表情识别系统 - RAF-DB数据集
优化版本：模块化设计，直接内联配置值，添加模型保存功能
"""

import os
import numpy as np
import torch
from time import time
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
import seaborn as sns
import timm
import gdown
from typing import Optional, Tuple, List, Dict
import json

# ==================== 全局配置 ====================
# 设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/emotion/DATASET'))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, 'fer_resnet18_best.pth'))
CONFIG_SAVE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, 'training_config.json'))

# 图像处理
IMAGE_SIZE = (224, 224)
# RAF-DB数据集统计的均值和标准差
MEAN = [0.5752, 0.4495, 0.4012]
STD = [0.2086, 0.1911, 0.1827]

# 训练参数
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GAMMA = 2.0  # Focal Loss参数

# 模型选择
MODEL_TYPE = 'resnet18_celeb'  # 可选: 'resnet18', 'resnet18_celeb', 'vit'
PRETRAINED = True

# 类别信息
CLASS_NAMES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'natural']
NUM_CLASSES = 7

# 训练监控
LOG_INTERVAL = 50
SAVE_BEST_MODEL = True
EVAL_ON_TEST = True

# 数据增强
USE_DATA_AUGMENTATION = True

# 确保目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH) if os.path.dirname(MODEL_SAVE_PATH) else '.', exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

#可视化保存路径
VISUALIZE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '../../outputs/emotion/visualize'))

# ==================== 数据集类 ====================
class RAFDBDataset(Dataset):
    """RAF-DB面部表情数据集类（支持懒加载）"""
    
    def __init__(self, root_dir: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            train: True表示训练集，False表示测试集
            transform: 图像转换操作
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.images = []  # 存储图像路径
        self.labels = []  # 存储标签
        
        # 确定数据子目录
        data_dir = os.path.join(root_dir, 'train' if train else 'test')
        
        # 加载图像路径和标签（懒加载）
        for label_idx in range(1, 8):  # 标签从1到7
            label_dir = os.path.join(data_dir, str(label_idx))
            if not os.path.exists(label_dir):
                print(f"警告: 目录不存在 {label_dir}")
                continue
                
            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(label_dir, img_name))
                    self.labels.append(label_idx - 1)  # 转换为0-based索引
        
        print(f"数据集加载完成: {len(self.images)} 个样本 ({'训练集' if train else '测试集'})")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        img_path = self.images[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 返回黑色图像作为占位符
            image = Image.new('RGB', IMAGE_SIZE, color='black')
        
        label = self.labels[idx]
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布统计"""
        return Counter(self.labels)

# ==================== 数据预处理工具 ====================
def get_train_transform():
    """获取训练集转换（包含数据增强）"""
    train_transforms = [
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    
    # 添加随机擦除增强
    if USE_DATA_AUGMENTATION:
        train_transforms.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), value='random')
        )
    
    return transforms.Compose(train_transforms)

def get_test_transform():
    """获取测试集转换（无数据增强）"""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

def create_dataloaders(split_ratio: float = 0.8):
    """
    创建数据加载器
    
    Args:
        split_ratio: 训练集划分比例
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 检查数据集路径是否存在
    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据集路径不存在 {DATA_PATH}")
        print("请确保已下载RAF-DB数据集并将其放在正确位置")
        exit(1)
    
    # 创建完整训练集
    full_train_dataset = RAFDBDataset(
        root_dir=DATA_PATH,
        train=True,
        transform=get_train_transform()
    )
    
    # 划分训练集和验证集
    train_size = int(split_ratio * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )
    
    # 创建测试集
    test_dataset = RAFDBDataset(
        root_dir=DATA_PATH,
        train=False,
        transform=get_test_transform()
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 分析数据分布
    print("\n训练集类别分布:")
    train_labels = []
    for _, batch_labels in train_loader:
        train_labels.extend(batch_labels.numpy())
    
    class_counts = Counter(train_labels)
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 0)
        print(f"  {CLASS_NAMES[i]:>10}: {count} 样本 ({100.*count/len(train_labels):.1f}%)")
    
    return train_loader, val_loader, test_loader

# ==================== 模型定义 ====================
class BasicBlock(nn.Module):
    """ResNet基础块"""
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
    """ResNet架构"""
    
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 1000):
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

def create_resnet18(pretrained: bool = True, num_classes: int = 7, 
                   pretrained_weights: Optional[str] = None):
    """
    创建ResNet-18模型
    
    Args:
        pretrained: 是否使用预训练权重
        num_classes: 输出类别数
        pretrained_weights: 自定义预训练权重路径
    
    Returns:
        ResNet-18模型
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    
    if pretrained:
        if pretrained_weights and os.path.isfile(pretrained_weights):
            # 加载自定义预训练权重
            print(f"加载自定义预训练权重: {pretrained_weights}")
            pretrain_dict = torch.load(pretrained_weights, map_location='cpu')
            
            if 'state_dict' in pretrain_dict:
                pretrain_dict = pretrain_dict['state_dict']
            
            # 清理键名并移除全连接层权重
            state_dict = {}
            for k, v in pretrain_dict.items():
                k = k.replace('module.', '')
                if k not in ['fc.weight', 'fc.bias'] and k in model.state_dict():
                    state_dict[k] = v
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"注意: 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"注意: 意外的键: {unexpected_keys}")
        else:
            # 加载ImageNet预训练权重
            print("加载ImageNet预训练权重")
            try:
                pretrain_dict = model_zoo.load_url(
                    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                    model_dir = os.path.abspath(os.path.join(CURRENT_DIR,'./pretrained'))
                )
                
                # 移除全连接层权重
                pretrain_dict.pop("fc.weight", None)
                pretrain_dict.pop("fc.bias", None)
                
                model.load_state_dict(pretrain_dict, strict=False)
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
                print("将继续使用随机初始化的权重")
        
        print("预训练权重加载完成")
    
    return model

def create_vit(model_name: str = 'vit_small_patch16_224', 
              num_classes: int = 7, pretrained: bool = True):
    """
    创建Vision Transformer模型
    
    Args:
        model_name: ViT模型名称
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        ViT模型
    """
    print(f"创建ViT模型: {model_name}")
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"创建ViT模型失败: {e}")
        print("请确保已安装timm库: pip install timm")
        exit(1)

# ==================== 损失函数 ====================
class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重张量
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 损失减少方式 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 模型输出 (N, C)
            targets: 真实标签 (N,)
        
        Returns:
            Focal Loss值
        """
        # 计算softmax概率
        p = torch.softmax(inputs, dim=1)
        
        # 交叉熵损失
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # 获取目标类别的概率
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算Focal权重
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== 训练器类 ====================
class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 test_loader: DataLoader):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 计算类别权重
        self.class_weights = self._calculate_class_weights()
        
        # 初始化损失函数
        self.criterion = FocalLoss(
            alpha=self.class_weights,
            gamma=GAMMA,
            reduction='mean'
        ).to(DEVICE)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 最佳模型状态
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        # 获取训练集标签分布
        labels = []
        for _, batch_labels in self.train_loader:
            labels.extend(batch_labels.numpy())
        
        class_counts = Counter(labels)
        counts = torch.tensor([class_counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float32)
        
        # 计算类别权重（样本越少，权重越大）
        weights = 1.0 / (counts + 1e-6)  # 添加小值防止除零
        weights = weights / weights.sum() * len(weights)  # 归一化
        
        # 打印类别权重信息
        print("\n类别权重信息:")
        for i, (name, weight) in enumerate(zip(CLASS_NAMES, weights)):
            print(f"  {name:>10}: {weight:.4f} (样本数: {class_counts.get(i, 0)})")
        
        return weights.to(DEVICE)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:03d}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            current_loss = epoch_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 定期日志记录
            if batch_idx % LOG_INTERVAL == 0 and batch_idx > 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}: '
                      f'Loss={loss.item():.4f}, Acc={100.*predicted.eq(targets).sum().item()/targets.size(0):.2f}%')
        
        # 计算epoch平均指标
        avg_loss = epoch_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self) -> Tuple[float, float]:
        """在验证集上评估模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = val_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def test(self) -> float:
        """在测试集上评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        print(f'\n测试集准确率: {test_acc:.2f}%')
        
        return test_acc
    
    def train(self) -> Dict:
        """完整训练过程"""
        print(f"\n开始训练 {MODEL_TYPE} 模型...")
        print(f"设备: {DEVICE}")
        print(f"训练周期: {NUM_EPOCHS}")
        print(f"批大小: {BATCH_SIZE}")
        print(f"学习率: {LEARNING_RATE}")
        
        start_time = time()
        
        for epoch in range(1, NUM_EPOCHS + 1):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印epoch结果
            print(f'\nEpoch {epoch:03d} 结果:')
            print(f'  训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%')
            print(f'  验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%')
            print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc and SAVE_BEST_MODEL:
                self.best_val_acc = val_acc
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'config': {
                        'model_type': MODEL_TYPE,
                        'num_classes': NUM_CLASSES,
                        'mean': MEAN,
                        'std': STD,
                        'image_size': IMAGE_SIZE,
                    }
                }
                self.save_model(MODEL_SAVE_PATH)
                print(f'  最佳模型已保存 (验证准确率: {val_acc:.2f}%)')
        
        # 训练完成
        total_time = time() - start_time
        print(f'\n训练完成! 总用时: {total_time:.2f}秒')
        
        # 加载最佳模型进行最终测试
        if self.best_model_state and EVAL_ON_TEST:
            print("\n使用最佳模型进行测试...")
            self.load_model(MODEL_SAVE_PATH)
            test_acc = self.test()
            self.history['test_acc'] = test_acc
        
        return self.history
    
    def save_model(self, path: str):
        """保存模型"""
        if self.best_model_state:
            torch.save(self.best_model_state, path)
            print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"模型已从 {path} 加载 (epoch {checkpoint.get('epoch', 'N/A')}, "
                  f"val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
        else:
            print(f"模型文件 {path} 不存在")

# ==================== 可视化工具 ====================
def plot_training_history(history: Dict):
    """绘制训练历史图表"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练和验证损失
    axes[0, 0].plot(history['train_loss'], label='训练损失', marker='o', linewidth=2, markersize=5)
    axes[0, 0].plot(history['val_loss'], label='验证损失', marker='s', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练和验证准确率
    axes[0, 1].plot(history['train_acc'], label='训练准确率', marker='o', linewidth=2, markersize=5)
    axes[0, 1].plot(history['val_acc'], label='验证准确率', marker='s', linewidth=2, markersize=5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].set_title('训练和验证准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率变化
    axes[1, 0].plot(history['learning_rate'], label='学习率', color='green', marker='^', linewidth=2, markersize=5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('学习率')
    axes[1, 0].set_title('学习率变化')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最终测试结果（如果有）
    if 'test_acc' in history:
        axes[1, 1].bar(['测试准确率'], [history['test_acc']], color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('准确率 (%)')
        axes[1, 1].set_title(f'最终测试准确率: {history["test_acc"]:.2f}%')
        axes[1, 1].set_ylim([0, 100])
        for i, v in enumerate([history['test_acc']]):
            axes[1, 1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.suptitle(f'{MODEL_TYPE} 训练结果', fontsize=16, fontweight='bold')
    # 自动创建目录（如果不存在）
    os.makedirs(VISUALIZE_PATH, exist_ok=True)  # exist_ok=True 表示目录已存在时不报错
    # 完整的文件路径
    save_path = os.path.join(VISUALIZE_PATH, 'training_history.png')
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {save_path}")
    plt.close() # 关闭图形，释放内存 

def visualize_predictions(model: nn.Module, test_dataset: Dataset, num_samples: int = 4):
    """可视化模型预测结果"""
    model.eval()
    
    # 随机选择样本
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = [[axes[0]], [axes[1]]]
    
    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        
        # 反归一化用于显示
        image_denorm = image * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # 预测
        with torch.no_grad():
            input_tensor = image.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_label = output.argmax().item()
            confidence = probabilities[0, predicted_label].item()
        
        # 显示图像
        axes[0, i].imshow(image_denorm.permute(1, 2, 0).numpy())
        
        # 设置标题
        title_color = 'green' if predicted_label == true_label else 'red'
        title = f"真实: {CLASS_NAMES[true_label]}\n预测: {CLASS_NAMES[predicted_label]}"
        title += f"\n置信度: {confidence:.2f}"
        axes[0, i].set_title(title, color=title_color, fontsize=10)
        axes[0, i].axis('off')
        
        # 显示概率条形图
        axes[1, i].barh(CLASS_NAMES, probabilities[0].cpu().numpy(), alpha=0.7)
        axes[1, i].set_xlim([0, 1])
        axes[1, i].set_xlabel('概率')
        axes[1, i].set_title('概率分布')
        
        # 高亮预测的类别
        for j, (bar, prob) in enumerate(zip(axes[1, i].patches, probabilities[0].cpu().numpy())):
            if j == predicted_label:
                bar.set_color('red' if predicted_label != true_label else 'green')
                bar.set_alpha(1.0)
    
    plt.suptitle('模型预测结果可视化', fontsize=14, fontweight='bold')
    # 自动创建目录（如果不存在）
    os.makedirs(VISUALIZE_PATH, exist_ok=True)  # exist_ok=True 表示目录已存在时不报错

    # 完整的文件路径
    save_path = os.path.join(VISUALIZE_PATH, 'test_predictions.png')

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {save_path}")
    plt.close() # 关闭图形，释放内存 

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("面部表情识别系统 - RAF-DB数据集")
    print("优化版本 - 直接内联配置值")
    print("=" * 60)
    
    # 数据处理
    print("\n[1/4] 数据处理...")
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # 创建模型
    print("\n[2/4] 创建模型...")
    if MODEL_TYPE == 'resnet18':
        model = create_resnet18(
            pretrained=PRETRAINED,
            num_classes=NUM_CLASSES
        )
    elif MODEL_TYPE == 'resnet18_celeb':
        # 下载Celeb-1M预训练权重
        celeb_weights_path = os.path.abspath(os.path.join(CURRENT_DIR,'./resnet18_celeb.pth'))
        if not os.path.exists(celeb_weights_path):
            print("下载Celeb-1M预训练权重...")
            try:
                gdown.download(
                    'https://drive.google.com/uc?id=1e7FmEfTIB__ATpSw5oHz61N1-bTl0Dlk',
                    celeb_weights_path,
                    quiet=False
                )
            except Exception as e:
                print(f"下载预训练权重失败: {e}")
                print("将使用ImageNet预训练权重")
                celeb_weights_path = None
        
        model = create_resnet18(
            pretrained=PRETRAINED,
            num_classes=NUM_CLASSES,
            pretrained_weights=celeb_weights_path
        )
    elif MODEL_TYPE == 'vit':
        model = create_vit(
            model_name='vit_small_patch16_224',
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED
        )
    else:
        raise ValueError(f"不支持的模型类型: {MODEL_TYPE}")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {MODEL_TYPE}")
    print(f"参数量: {total_params:,}")
    
    # 训练模型
    print("\n[3/4] 训练模型...")
    trainer = Trainer(model, train_loader, val_loader, test_loader)
    history = trainer.train()
    
    # 可视化结果
    print("\n[4/4] 可视化结果...")
    plot_training_history(history)
    
    # 可视化预测示例
    test_dataset = RAFDBDataset(
        root_dir=DATA_PATH,
        train=False,
        transform=get_test_transform()
    )
    visualize_predictions(model, test_dataset, num_samples=4)
    
    # 保存训练配置
    config_dict = {
        'model_type': MODEL_TYPE,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'gamma': GAMMA,
        'image_size': IMAGE_SIZE,
        'mean': MEAN,
        'std': STD,
        'class_names': CLASS_NAMES,
        'best_val_acc': trainer.best_val_acc,
        'test_acc': history.get('test_acc', 'N/A'),
        'model_save_path': MODEL_SAVE_PATH,
    }
    
    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳模型已保存到: {MODEL_SAVE_PATH}")
    print(f"训练配置已保存到: {CONFIG_SAVE_PATH}")
    print(f"最佳验证准确率: {trainer.best_val_acc:.2f}%")
    if 'test_acc' in history:
        print(f"测试准确率: {history['test_acc']:.2f}%")
    print("=" * 60)

# ==================== 单图像推理 ====================
def predict_single_image(image_path: str, model_path: str = 'fer_resnet18_best.pth'):
    """单张图像预测"""
    print(f"\n对单张图像进行预测: {image_path}")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 {image_path}")
        return None, None
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行训练程序或确保模型文件存在")
        return None, None
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 根据保存的配置确定模型类型
    if 'config' in checkpoint:
        model_type = checkpoint['config'].get('model_type', 'resnet18')
        num_classes = checkpoint['config'].get('num_classes', NUM_CLASSES)
        print(f"检测到模型类型: {model_type}")
    else:
        model_type = 'resnet18'
        num_classes = NUM_CLASSES
    
    # 创建对应模型
    if 'resnet18' in model_type:
        model = create_resnet18(pretrained=False, num_classes=num_classes)
    elif 'vit' in model_type:
        model = create_vit(num_classes=num_classes, pretrained=False)
    else:
        print(f"警告: 未知模型类型 {model_type}，使用ResNet-18")
        model = create_resnet18(pretrained=False, num_classes=num_classes)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax().item()
            confidence = probabilities[0, predicted_class].item()
        
        # 显示结果
        print("\n预测结果:")
        print(f"  类别: {CLASS_NAMES[predicted_class]}")
        print(f"  置信度: {confidence:.2%}")
        
        # 显示所有类别概率
        print("\n所有类别概率:")
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities[0])):
            print(f"  {class_name:>10}: {prob.item():.2%}")
        
        # 可视化
        plt.figure(figsize=(10, 4))
        
        # 显示图像
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"输入图像: {os.path.basename(image_path)}")
        plt.axis('off')
        
        # 显示概率条形图
        plt.subplot(1, 2, 2)
        colors = ['red' if i == predicted_class else 'blue' for i in range(len(CLASS_NAMES))]
        bars = plt.barh(CLASS_NAMES, probabilities[0].cpu().numpy(), color=colors, alpha=0.7)
        plt.xlabel('概率')
        plt.title(f'类别概率分布 (预测: {CLASS_NAMES[predicted_class]})')
        plt.xlim([0, 1])
        
        # 在条形上添加概率值
        for bar, prob in zip(bars, probabilities[0].cpu().numpy()):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.2%}', va='center', fontsize=9)
        
        # 自动创建目录（如果不存在）
        os.makedirs(VISUALIZE_PATH, exist_ok=True)  # exist_ok=True 表示目录已存在时不报错

        # 完整的文件路径
        save_path = os.path.join(VISUALIZE_PATH, 'pred.png')

        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存到: {save_path}")
        plt.close() # 关闭图形，释放内存 
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None

if __name__ == "__main__":
    # 显示配置信息
    print("当前配置:")
    print(f"  模型类型: {MODEL_TYPE}")
    print(f"  预训练: {PRETRAINED}")
    print(f"  Epoch数: {NUM_EPOCHS}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    
    # 运行主训练程序
    main()
    
    # 示例：对单张图像进行预测
    # 确保有 test_img 目录和测试图像
    test_img_path = os.path.abspath(os.path.join(CURRENT_DIR, '../../data/emotion/test_img/test_disgust.png'))
    if os.path.exists(test_img_path):
        predict_single_image(test_img_path, MODEL_SAVE_PATH)
    else:
        print(f"\n注意: 测试图像路径不存在 {test_img_path}")
        print("单图像预测功能需要测试图像文件")