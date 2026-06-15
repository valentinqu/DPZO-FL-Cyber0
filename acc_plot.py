import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区 =================
# 1. 文件夹路径
FOLDER_PATH = r"D:\Visual Studio Code\TDLGroupProject\output\SF\20clients"


# 3. 滑动平均窗口大小
SMOOTHING_WINDOW = 5 

# ==========================================

def get_smooth_curve(points, window):
    """计算滑动平均"""
    if window <= 1:
        return points
    smoothed = np.convolve(points, np.ones(window)/window, mode='valid')
    # 填充前面的空白，使其长度与原数组一致
    pad = np.full(window - 1, smoothed[0]) 
    return np.concatenate((pad, smoothed))

def plot_all_csvs():
    # 获取该文件夹下所有的 CSV 文件
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    if not csv_files:
        print(f"在 {FOLDER_PATH} 下没有找到 CSV 文件！")
        return

    # 设置绘图风格
    plt.figure(figsize=(10, 6), dpi=300)
    
    # 获取一个好看的颜色图谱
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    color_idx = 0

    for file in csv_files:
        filename = os.path.basename(file)
            
        print(f"正在绘制: {filename}")
        
        # 读取数据 (假设第一列是 Loss，第二列是 Accuracy)
        try:
            df = pd.read_csv(file, header=None, names=["Loss", "Accuracy"])
            
            # 把 Accuracy 转换为百分比 (如果是 0.85 这种格式的话)
            if df['Accuracy'].max() <= 1.0:
                acc_data = df['Accuracy'] * 100
            else:
                acc_data = df['Accuracy']
                
            steps = np.arange(len(acc_data))
            
            # 从文件名中提取标签，去掉 .csv 后缀
            label_name = filename.replace('.csv', '')
            
            # 获取平滑后的曲线
            smoothed_acc = get_smooth_curve(acc_data.values, SMOOTHING_WINDOW)
            
            # 绘制真实的浅色震荡背景 (可选，增加高级感)
            plt.plot(steps, acc_data, color=colors[color_idx], alpha=0.2, linewidth=1)
            
            # 绘制平滑后的主曲线
            plt.plot(steps, smoothed_acc, label=label_name, color=colors[color_idx], linewidth=2.5)
            
            color_idx = (color_idx + 1) % 10 # 循环使用颜色
            
        except Exception as e:
            print(f"读取 {filename} 失败: {e}")

    # ===== 图表美化 =====
    plt.title('Accuracy over Rounds (SF Attack, Num Attacker=4)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Communication Rounds', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 把图例放到右下角，如果挡住了曲线可以改成 'best' 或者 bbox_to_anchor
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9, edgecolor='black')
    
    # 动态调整 Y 轴范围
    plt.ylim(50, 100) # 根据你的实际数据可以改成 75 到 95 等
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(FOLDER_PATH, "combined_SF_comparison.png")
    plt.savefig(save_path)
    print(f"\n绘图成功！图片已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_all_csvs()