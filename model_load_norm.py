import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import math

# 定义与训练时相同的模型架构
class RobotStatePredictor(nn.Module):
    def __init__(self):
        super(RobotStatePredictor, self).__init__()
        # 使用ResNet18作为骨干网络
        self.cnn = resnet18(pretrained=True)
        # 移除原始分类头
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # 状态预测头
        self.state_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, x):
        # CNN特征提取
        features = self.cnn(x)
        features = features.view(features.size(0), -1)  # 展平特征
        
        # 状态预测
        state_vector = self.state_head(features)
        return state_vector

class ModelInference:
    def __init__(self, model_path, device=None):
        """
        初始化模型推理类
        
        Args:
            model_path: 模型文件路径
            device: 推理设备，默认自动选择
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      'mps' if torch.backends.mps.is_available() else 
                                      'cpu')
        else:
            self.device = device
            
        # 加载模型
        self.model, self.xyz_ranges, self.angle_indices = self.load_model(model_path)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型实例
        model = RobotStatePredictor()
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 移动到指定设备
        model = model.to(self.device)
        
        # 设置为评估模式（重要：禁用dropout和batch normalization的训练行为）
        model.eval()
        
        # 冻结模型参数（确保参数不会被意外修改）
        for param in model.parameters():
            param.requires_grad = False
            
        # 获取标准化参数
        xyz_ranges = checkpoint.get('xyz_ranges', {})
        angle_indices = checkpoint.get('angle_indices', [])
        
        print(f"模型已加载到设备: {self.device}")
        print(f"模型参数已冻结，共有 {sum(p.numel() for p in model.parameters())} 个参数")
        
        return model, xyz_ranges, angle_indices
    
    def denormalize_state_labels(self, normalized_labels):
        """
        反标准化模型输出（将标准化的预测值转换回原始范围）
        """
        denormalized_labels = normalized_labels.clone()
        
        # 反标准化XYZ坐标
        for idx, (min_val, max_val) in self.xyz_ranges.items():
            # 从[-1, 1]反标准化到原始范围
            denormalized_labels[:, idx] = (normalized_labels[:, idx] + 1) / 2 * (max_val - min_val) + min_val
        
        # 反标准化角度
        for idx in self.angle_indices:
            # 从[-1, 1]反标准化到[-pi, pi]
            denormalized_labels[:, idx] = normalized_labels[:, idx] * math.pi
        
        return denormalized_labels
    
    def predict_from_image_path(self, image_path):
        """
        从图像路径预测机器人状态
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            numpy array: 16维状态向量（反标准化后的原始值）
        """
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():  # 确保不计算梯度
            normalized_prediction = self.model(image_tensor)
            
            # 反标准化到原始范围
            original_prediction = self.denormalize_state_labels(normalized_prediction)
            
        # 转换为numpy数组并移除batch维度
        return original_prediction.cpu().numpy().squeeze()
    
    def predict_from_image_tensor(self, image_tensor):
        """
        从图像张量预测机器人状态
        
        Args:
            image_tensor: 预处理后的图像张量 (C, H, W) 或 (B, C, H, W)
            
        Returns:
            numpy array: 状态向量
        """
        # 确保有batch维度
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            normalized_prediction = self.model(image_tensor)
            original_prediction = self.denormalize_state_labels(normalized_prediction)
            
        return original_prediction.cpu().numpy().squeeze()
    
    def batch_predict(self, image_paths):
        """
        批量预测多个图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            numpy array: (N, 16) 形状的状态向量数组
        """
        predictions = []
        
        for image_path in image_paths:
            pred = self.predict_from_image_path(image_path)
            predictions.append(pred)
            
        return np.array(predictions)

# 使用示例
def main():
    # 模型路径
    model_path = "best_model_noonehot_lr5_all_normalized.pth"
    
    # 创建推理实例
    inferencer = ModelInference(model_path)
    
    # 单张图像推理
    image_path = "path/to/your/image.png"
    try:
        prediction = inferencer.predict_from_image_path(image_path)
        
        print("预测的机器人状态:")
        print(f"形状: {prediction.shape}")
        print(f"值: {prediction}")
        
        # 解释预测结果
        print("\n详细状态:")
        print(f"机器人位置 (pos_x, pos_y, pos_z): {prediction[:3]}")
        print(f"机器人姿态 (rot_x, rot_y, rot_z): {prediction[3:6]}")
        print(f"夹持器角度 (jaw_angle): {prediction[6]}")
        print(f"目标物体位置 (obj_pos_x, obj_pos_y, obj_pos_z): {prediction[7:10]}")
        print(f"目标物体姿态 (obj_rot_x, obj_rot_y, obj_rot_z): {prediction[10:13]}")
        print(f"目标点位置 (goal_pos_x, goal_pos_y, goal_pos_z): {prediction[13:16]}")
        
    except Exception as e:
        print(f"推理出错: {e}")
    
    # 批量推理示例
    # image_paths = ["image1.png", "image2.png", "image3.png"]
    # batch_predictions = inferencer.batch_predict(image_paths)
    # print(f"批量预测结果形状: {batch_predictions.shape}")

if __name__ == "__main__":
    main()