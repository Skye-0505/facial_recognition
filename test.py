# test_config.py
import sys
sys.path.append('.')  # 确保可以导入

from config import cfg

def test_config():
    """测试配置是否正常工作"""
    print("测试项目配置...")
    
    # 打印基本配置
    cfg.print_config()
    
    # 检查关键路径
    print("\n检查关键路径:")
    print(f"数据集路径: {cfg.DATASET_DIR}")
    print(f"图标路径: {cfg.ICON_DIR}")
    print(f"模型路径: {cfg.MODEL_DIR}")
    
    # 检查模型文件是否存在
    print("\n检查模型文件:")
    for name, path in cfg.MODEL_PATHS.items():
        import os
        exists = os.path.exists(path)
        status = "✓ 存在" if exists else "✗ 缺失"
        print(f"  {name:25} {status:10} {path}")
    
    # 测试数据变换
    print("\n测试数据变换...")
    train_transform = cfg.get_train_transform()
    test_transform = cfg.get_test_transform()
    print(f"训练变换: {train_transform}")
    print(f"测试变换: {test_transform}")
    
    print("\n✅ 配置测试完成!")

if __name__ == "__main__":
    test_config()