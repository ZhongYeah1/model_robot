import os
import pandas as pd
import numpy as np

# 定义标签数据根目录
LABEL_ROOT = "/cluster/home/ZhongYeah/Vision/DEX-main/SurRoL/surrol/data/label0701"

# 需要分析的列索引和对应名称 (所有与x,y,z坐标相关的列)
columns_to_analyze = {
    0: "pos_x",         # 机器人末端X坐标
    1: "pos_y",         # 机器人末端Y坐标
    2: "pos_z",         # 机器人末端Z坐标
    10: "obj_pos_x",    # 目标物体X坐标
    11: "obj_pos_y",    # 目标物体Y坐标
    12: "obj_pos_z",    # 目标物体Z坐标
    16: "goal_pos_x",   # 目标点X坐标
    17: "goal_pos_y",   # 目标点Y坐标
    18: "goal_pos_z"    # 目标点Z坐标
}

# 初始化存储最大值和最小值的字典
max_values = {col_name: float('-inf') for col_name in columns_to_analyze.values()}
min_values = {col_name: float('inf') for col_name in columns_to_analyze.values()}

# 获取所有CSV文件
csv_files = [f for f in os.listdir(LABEL_ROOT) if f.startswith('label_') and f.endswith('.csv')]
print(f"发现 {len(csv_files)} 个CSV文件")

# 处理计数
processed_files = 0
error_files = 0

# 遍历所有CSV文件
for i, csv_file in enumerate(csv_files):
        
    try:
        file_path = os.path.join(LABEL_ROOT, csv_file)
        
        # 读取CSV文件
        df = pd.read_csv(file_path, header=None)
        
        # 确保文件格式正确
        if len(df.columns) < 19:
            print(f"警告: {csv_file} 的列数 ({len(df.columns)}) 小于预期的 19 列")
            error_files += 1
            continue
        
        # 分析每一列
        for col_idx, col_name in columns_to_analyze.items():
            # 计算当前文件中该列的最大值和最小值
            current_max = df[col_idx].max()
            current_min = df[col_idx].min()
            
            # 更新全局最大值和最小值
            if current_max > max_values[col_name]:
                max_values[col_name] = current_max
                
            if current_min < min_values[col_name]:
                min_values[col_name] = current_min
        
        processed_files += 1
    
    except Exception as e:
        print(f"处理 {csv_file} 时出错: {str(e)}")
        error_files += 1

# 打印结果
print("\n处理完成!")
print(f"成功处理: {processed_files} 个文件")
print(f"处理失败: {error_files} 个文件")

print("\n各坐标列的数值范围:")
print(f"{'坐标名称':<12} {'最小值':<15} {'最大值':<15} {'范围':<15}")
print("-" * 60)
for col_name in columns_to_analyze.values():
    min_val = min_values[col_name]
    max_val = max_values[col_name]
    range_val = max_val - min_val
    print(f"{col_name:<12} {min_val:<15.6f} {max_val:<15.6f} {range_val:<15.6f}")

# 将结果保存到文件
result_file = "coordinates_range_analysis.txt"
with open(result_file, 'w') as f:
    f.write("坐标列的数值范围分析结果\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'坐标名称':<12} {'最小值':<15} {'最大值':<15} {'范围':<15}\n")
    f.write("-" * 60 + "\n")
    for col_name in columns_to_analyze.values():
        min_val = min_values[col_name]
        max_val = max_values[col_name]
        range_val = max_val - min_val
        f.write(f"{col_name:<12} {min_val:<15.6f} {max_val:<15.6f} {range_val:<15.6f}\n")

print(f"\n结果已保存到 {result_file}")