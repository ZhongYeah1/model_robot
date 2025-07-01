import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.models as models
from torchvision.models import resnet18
import os
import pandas as pd
from PIL import Image
import numpy as np
import wandb

WANDB_DIR = "wandb_logs_19_doub"
BEST_MODEL_DIR = "best_model_19_doub.pth"
VIDEO_ROOT = ""
LABEL_ROOT = ""
WANDBNAME = "train_19_doub"


device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 
                      'cpu')

# 自定义数据集类（支持新标签格式）
class RobotStateDataset(Dataset):
    def __init__(self, video_root, label_root, transform=None):
        self.video_root = video_root
        self.label_root = label_root
        self.transform = transform
        self.samples = []  # 存储(image_path, state_label, task_label)
        
        # 遍历所有视频文件夹
        video_dirs = sorted([d for d in os.listdir(video_root) if d.startswith('video_')])
        for video_dir in video_dirs:
            idx = int(video_dir.split('_')[1])
            label_path = os.path.join(label_root, f'label_{idx}.csv')
            
            # 读取标签文件
            labels = pd.read_csv(label_path, header=None).values.astype(np.float32)
            
            # 遍历视频中的每一帧
            video_path = os.path.join(video_root, video_dir)
            for frame_idx in range(len(labels)):
                img_path = os.path.join(video_path, f'img_{frame_idx}.png')
                if os.path.exists(img_path):
                    # 拆分标签
                    state_label = np.concatenate([labels[frame_idx][:7], labels[frame_idx][10:19]])
                    task_label = labels[frame_idx][7:10]
                    self.samples.append((img_path, state_label, task_label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, state_label, task_label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 转换为Tensor
        state_label = torch.tensor(state_label, dtype=torch.float32)
        task_label = torch.tensor(task_label, dtype=torch.float32)
        return image, state_label, task_label

# 优化后的模型架构
class RobotStatePredictor(nn.Module):
    def __init__(self):
        super(RobotStatePredictor, self).__init__()
        # 使用ResNet18作为骨干网络
        self.cnn = resnet18(pretrained=True)
        # 移除原始分类头
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # 状态预测头
        self.state_head = nn.Sequential(
            nn.Linear(512, 128),  # 直接从512维特征开始
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        # 任务分类头
        self.task_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        # CNN特征提取
        features = self.cnn(x)
        features = features.view(features.size(0), -1)  # 展平特征
        
        # 并行预测
        state_vector = self.state_head(features)
        task_vector = self.task_head(features)
        return state_vector, task_vector

# 组合损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, state_weight=3.0, task_weight=1.0):
        super().__init__()
        self.state_criterion = nn.MSELoss()
        self.task_criterion = nn.BCELoss()
        self.state_weight = state_weight
        self.task_weight = task_weight
    
    def forward(self, state_pred, task_pred, state_true, task_true):
        state_loss = self.state_criterion(state_pred, state_true)
        task_loss = self.task_criterion(task_pred, task_true)
        return (self.state_weight * state_loss + 
                self.task_weight * task_loss,
                state_loss,
                task_loss)

# 训练函数（支持多任务）
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_state_loss = 0.0
    running_task_loss = 0.0
    running_total_loss = 0.0
    task_correct = 0
    total_samples = 0
    
    for images, state_labels, task_labels in loader:
        images = images.to(device)
        state_labels = state_labels.to(device)
        task_labels = task_labels.to(device)
        
        # 前向传播
        state_pred, task_pred = model(images)
        
        # 计算损失
        total_loss, state_loss, task_loss = criterion(
            state_pred, task_pred, state_labels, task_labels
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 累计损失
        batch_size = images.size(0)
        running_total_loss += total_loss.item() * batch_size
        running_state_loss += state_loss.item() * batch_size
        running_task_loss += task_loss.item() * batch_size
        
        # 计算任务分类准确率
        task_pred_binary = (task_pred > 0.5).float()
        task_correct += (task_pred_binary == task_labels).all(dim=1).sum().item()
        total_samples += batch_size
    
    # 计算epoch指标
    epoch_total_loss = running_total_loss / len(loader.dataset)
    epoch_state_loss = running_state_loss / len(loader.dataset)
    epoch_task_loss = running_task_loss / len(loader.dataset)
    task_accuracy = task_correct / total_samples * 100.0
    
    return epoch_total_loss, epoch_state_loss, epoch_task_loss, task_accuracy

# 评估函数（支持多任务）
def evaluate(model, loader, criterion, device):
    model.eval()
    running_state_loss = 0.0
    running_task_loss = 0.0
    running_total_loss = 0.0
    task_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, state_labels, task_labels in loader:
            images = images.to(device)
            state_labels = state_labels.to(device)
            task_labels = task_labels.to(device)
            
            # 前向传播
            state_pred, task_pred = model(images)
            
            # 计算损失
            total_loss, state_loss, task_loss = criterion(
                state_pred, task_pred, state_labels, task_labels
            )
            
            # 累计损失
            batch_size = images.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_state_loss += state_loss.item() * batch_size
            running_task_loss += task_loss.item() * batch_size
            
            # 计算任务分类准确率
            task_pred_binary = (task_pred > 0.5).float()
            task_correct += (task_pred_binary == task_labels).all(dim=1).sum().item()
            total_samples += batch_size
    
    # 计算epoch指标
    epoch_total_loss = running_total_loss / len(loader.dataset)
    epoch_state_loss = running_state_loss / len(loader.dataset)
    epoch_task_loss = running_task_loss / len(loader.dataset)
    task_accuracy = task_correct / total_samples * 100.0
    
    return epoch_total_loss, epoch_state_loss, epoch_task_loss, task_accuracy

def split_by_video_ids(dataset, train_ratio=0.8, seed=42):
    """
    按视频ID划分数据集，保证同一个视频的所有帧都在同一个集合中
    """
    # 提取每个样本的视频ID
    sample_to_video = {}
    for i, (img_path, _, _) in enumerate(dataset.samples):
        # 从图像路径中提取视频ID：.../video_X/img_Y.png
        video_dir = os.path.dirname(img_path)
        video_id = int(os.path.basename(video_dir).split('_')[1])
        sample_to_video[i] = video_id
    
    # 获取唯一的视频ID列表
    unique_video_ids = list(set(sample_to_video.values()))
    
    # 随机打乱视频ID
    np.random.seed(seed)
    np.random.shuffle(unique_video_ids)
    
    # 按比例划分视频ID
    train_video_count = int(len(unique_video_ids) * train_ratio)
    train_video_ids = set(unique_video_ids[:train_video_count])
    test_video_ids = set(unique_video_ids[train_video_count:])
    
    # 根据视频ID分配样本索引
    train_indices = [i for i, vid in sample_to_video.items() if vid in train_video_ids]
    test_indices = [i for i, vid in sample_to_video.items() if vid in test_video_ids]
    
    # 创建子集
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"按视频ID划分: {len(train_video_ids)}个视频({len(train_indices)}帧)用于训练, "
          f"{len(test_video_ids)}个视频({len(test_indices)}帧)用于测试")
    return train_dataset, test_dataset

def main():
    # 初始化wandb
    wandb.init(
        project="robot-state-prediction",
        name=WANDBNAME,
        config={
            "architecture": "ResNet18-MultiTask",
            "dataset": "robot_state",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "state_weight": 3.0,
            "task_weight": 1.0,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau"
        },
        mode="offline",  # 添加此行，设置为离线模式
        dir=WANDB_DIR
    )
    config = wandb.config

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    video_root = VIDEO_ROOT  # 替换为实际路径
    label_root = LABEL_ROOT  # 替换为实际路径
    dataset = RobotStateDataset(video_root, label_root, transform=transform)

    train_dataset, test_dataset = split_by_video_ids(dataset, train_ratio=0.8, seed=42)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=2
    )

    # 记录数据集大小
    wandb.log({
        "train_size": len(train_dataset),
        "test_size": len(test_dataset)
    })

    # 初始化模型
    model = RobotStatePredictor().to(device)
    wandb.watch(model, log="all")

    # 损失函数和优化器
    criterion = MultiTaskLoss(
        state_weight=config.state_weight,
        task_weight=config.task_weight
    )
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=5, verbose=True
    )

    # 训练循环
    num_epochs = config.epochs
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        train_total, train_state, train_task, train_acc = train(
            model, train_loader, criterion, optimizer, device
        )
        
        # 评估阶段
        val_total, val_state, val_task, val_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # 学习率调整
        scheduler.step(val_total)
        
        # 记录指标到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/total_loss": train_total,
            "train/state_loss": train_state,
            "train/task_loss": train_task,
            "train/task_acc": train_acc,
            "val/total_loss": val_total,
            "val/state_loss": val_state,
            "val/task_loss": val_task,
            "val/task_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Total: {train_total:.4f} | State: {train_state:.4f} | Task: {train_task:.4f} | Acc: {train_acc:.2f}%')
        print(f'Val   Total: {val_total:.4f} | State: {val_state:.4f} | Task: {val_task:.4f} | Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_total < best_val_loss:
            best_val_loss = val_total
            model_path = BEST_MODEL_DIR
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, model_path)
            wandb.save(model_path)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_acc"] = val_acc
            print(f'Best model saved with val loss: {val_total:.4f}, acc: {val_acc:.2f}%')

    print('Training completed!')

    # 加载最佳模型进行最终评估
    checkpoint = torch.load(BEST_MODEL_DIR)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_total, final_state, final_task, final_acc = evaluate(
        model, test_loader, criterion, device
    )
    print(f'Final Validation: Total {final_total:.4f} | State {final_state:.4f} | Task {final_task:.4f} | Acc {final_acc:.2f}%')
    
    # 记录最终结果
    wandb.run.summary["final_val_loss"] = final_total
    wandb.run.summary["final_val_acc"] = final_acc
    wandb.finish()

if __name__ == '__main__':
    main()