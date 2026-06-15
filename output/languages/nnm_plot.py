import pandas as pd
import matplotlib.pyplot as plt
import os


file1 = r"D:\Visual Studio Code\TDLGroupProject\output\languages\llm_sst2_pure_baseline.csv"

file2 = r"D:\Visual Studio Code\TDLGroupProject\output\languages\llm_sst2_iid_attack_trimmed_mean.csv"

# 论文图例中显示的专业名称
label1 = "OPT+SST2 pure"
label2 = "FOE Attack + Trimmed Mean (iid)"

# ==========================================
# 2. 读取数据 (指定表头为 Loss 和 Accuracy)
# ==========================================
if not os.path.exists(file1) or not os.path.exists(file2):
    print("❌ 找不到 CSV 文件，请确保文件位于当前运行目录下！")
    exit()

df1 = pd.read_csv(file1, header=None, names=["Loss", "Accuracy"])
df2 = pd.read_csv(file2, header=None, names=["Loss", "Accuracy"])

# ==========================================
# 3. 开始绘图 (1行2列的画布，左 Loss，右 Acc)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# -----------------
#  左侧：Loss 对比
# -----------------
ax1.plot(df1.index, df1["Loss"], label=label1, color='red', linestyle='--', alpha=0.8)
ax1.plot(df2.index, df2["Loss"], label=label2, color='green', linewidth=2)

ax1.set_title("Training Loss Comparison (CNN Non-IID)", fontsize=14, pad=15)
ax1.set_xlabel("Communication Rounds", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.7)

# -----------------
#  右侧：Accuracy 对比
# -----------------
# 注意：将 Accuracy 乘以 100 转换为百分比格式
ax2.plot(df1.index, df1["Accuracy"] * 100, label=label1, color='red', linestyle='--', alpha=0.8)
ax2.plot(df2.index, df2["Accuracy"] * 100, label=label2, color='green', linewidth=2)

ax2.set_title("Test Accuracy Comparison (%)", fontsize=14, pad=15)
ax2.set_xlabel("Communication Rounds", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.7)

# ==========================================
# 4. 渲染与保存
# ==========================================
plt.tight_layout()

# 自动保存的高清图片名称
output_image = "llm_comparision.png"
plt.savefig(output_image, dpi=300)
print(f"绘图成功！图片已保存至: {output_image}")

# 弹出窗口显示
plt.show()