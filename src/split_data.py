import os
import shutil
import random
from tqdm import tqdm

def solve_val_issue_and_split(src_root, target_root, max_per_class=300):
    # 1. 初始化路径
    splits = ['train', 'val', 'test']
    
    # 获取 raw 下的所有子文件夹
    all_classes = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    # 强制只取前 20 个类，确保符合实验要求
    selected_classes = sorted(all_classes)[:20]
    
    print(f"🚀 启动分类...")
    print(f"目标类别数: {len(selected_classes)}")
    print(f"每类最大样本: {max_per_class}")

    # 2. 彻底清理旧数据，防止 val 报错
    for s in splits:
        path = os.path.join(target_root, s)
        if os.path.exists(path):
            shutil.rmtree(path) # 强制删除整个目录
        os.makedirs(path)

    # 3. 开始分配
    for cls in tqdm(selected_classes, desc="同步类别数据"):
        # 创建 train/val/test 下的类别文件夹
        for s in splits:
            os.makedirs(os.path.join(target_root, s, cls), exist_ok=True)

        # 获取原始图片
        cls_path = os.path.join(src_root, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 过滤掉无效或太小的类别
        if len(images) < 10:
            print(f"⚠️ 警告: 类别 {cls} 图片太少 ({len(images)}张)，已跳过")
            continue

        # 打乱并截断
        random.seed(42)
        random.shuffle(images)
        if len(images) > max_per_class:
            images = images[:max_per_class]

        # 重新计算比例 (确保即便只有 10 张图也能分到 val)
        total = len(images)
        train_num = int(total * 0.7)
        val_num = int(total * 0.2)
        if val_num == 0: val_num = 1 # 强制至少 1 张

        # 分配文件
        for i, img in enumerate(images):
            src_file = os.path.join(cls_path, img)
            if i < train_num:
                dst = os.path.join(target_root, 'train', cls, img)
            elif i < train_num + val_num:
                dst = os.path.join(target_root, 'val', cls, img)
            else:
                dst = os.path.join(target_root, 'test', cls, img)
            
            try:
                shutil.copy(src_file, dst)
            except Exception as e:
                continue

    # 4. 最终对齐检查
    train_classes = len(os.listdir(os.path.join(target_root, 'train')))
    val_classes = len(os.listdir(os.path.join(target_root, 'val')))
    print(f"\n✅ 同步检查完成:")
    print(f"训练集类别: {train_classes}, 验证集类别: {val_classes}")
    if train_classes != val_classes:
        print("❌ 严重错误: 类别数量仍不对齐，请手动删除 data/train 和 data/val 后重试")
    else:
        print("🎉 类别完美对齐！")

if __name__ == "__main__":
    # 使用绝对路径，确保 100% 正确
    RAW = r"E:\GarbageClassification\data\raw"
    BASE = r"E:\GarbageClassification\data"
    solve_val_issue_and_split(RAW, BASE)