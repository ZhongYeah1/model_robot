import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18
import pandas as pd
import os
from PIL import Image
import numpy as np
import wandb

BEST_MODEL_DIR = "best_model_10_one.pth"
WANDB_DIR = "wandb_logs_10_one"
VIDEO_ROOT = ""
LABEL_ROOT = ""

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 
                      'cpu')

# 自定义数据集类
class RobotStateDataset(Dataset):
    def __init__(self, video_root, label_root, transform=None):
        self.video_root = video_root
        self.label_root = label_root
        self.transform = transform
        self.samples = []
        
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
                    self.samples.append((img_path, labels[frame_idx]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 只使用前10个值作为标签（7维机器人状态 + 3维任务类型one-hot编码）
        label = torch.tensor(label[:10], dtype=torch.float32)
        return image, label

# 模型架构
class RobotStatePredictor(nn.Module):
    def __init__(self):
        super(RobotStatePredictor, self).__init__()
        # 使用ResNet18
        self.cnn = resnet18(pretrained=True)
        # 移除原始分类头
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # MLP头部
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # CNN特征提取
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        # MLP回归
        state_vector = self.mlp(features)
        return state_vector


# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

# 测试函数
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def main():
    
    # 初始化wandb
    wandb.init(
        project="robot-state-prediction",  # 项目名称
        name="train_10_one",        # 实验名称
        config={
            "architecture": "ResNet18",
            "dataset": "robot_state",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau"
        },
        mode="offline",  # 添加此行，设置为离线模式
        dir=WANDB_DIR
    )
    

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet需要的最小尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    video_root = VIDEO_ROOT  # 替换为实际路径
    label_root = LABEL_ROOT  # 替换为实际路径
    dataset = RobotStateDataset(video_root, label_root, transform=transform)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2
    )

    # 记录数据集大小
    wandb.log({"train_size": len(train_dataset), "test_size": len(test_dataset)})

    # 初始化模型
    model = RobotStatePredictor().to(device)
    # 记录模型到wandb
    wandb.watch(model, log="all")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    num_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # 记录指标到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | '
            f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = BEST_MODEL_DIR
            torch.save(model.state_dict(), model_path)
            # 保存模型到wandb
            wandb.save(model_path)
            wandb.run.summary["best_val_loss"] = best_val_loss
            print(f'Best model saved with val loss: {val_loss:.6f}')

    print('Training completed!')

    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(BEST_MODEL_DIR))
    final_val_loss = evaluate(model, test_loader, criterion, device)
    print(f'Final Validation Loss: {final_val_loss:.6f}')
    
    # 记录最终结果
    wandb.run.summary["final_val_loss"] = final_val_loss
    
    # 关闭wandb
    wandb.finish()
if __name__ == '__main__':
    main()