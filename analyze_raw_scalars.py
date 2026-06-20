import numpy as np
import pandas as pd


CSV_PATH = "output/languages/scalar_stats/sst2_opt125m_lora_raw_scalars.csv"

df = pd.read_csv(CSV_PATH)

x = df["abs_value"].astype(float).values

print("============ Raw Scalar Global Stats ============")
print(f"num_values = {len(x)}")
print(f"mean       = {x.mean():.6f}")
print(f"std        = {x.std():.6f}")
print(f"max        = {x.max():.6f}")
print()

for q in [50, 75, 90, 95, 97.5, 99, 99.5]:
    print(f"p{q:<5} = {np.percentile(x, q):.6f}")

print("\n============ Candidate C Clip Ratios ============")

candidate_cs = [
    np.percentile(x, 90),
    np.percentile(x, 95),
    np.percentile(x, 97.5),
    np.percentile(x, 99),
    50,
    100,
    150,
    200,
    300,
    500,
]

candidate_cs = sorted(set(round(float(c), 4) for c in candidate_cs))

for c in candidate_cs:
    clip_ratio = np.mean(x > c)
    print(f"C={c:<10.4f} clip_ratio={clip_ratio * 100:>6.2f}%")

print("\n============ Recommended C ============")

p95 = np.percentile(x, 95)
p975 = np.percentile(x, 97.5)
p99 = np.percentile(x, 99)

# 简单规则：
# 如果 p99 远大于 p95，说明有长尾 outliers，用 p95 或 p97.5。
# 如果差距不大，p97.5 更稳。
if p99 > 2.5 * p95:
    rec = p95
    reason = "p99 is much larger than p95; using p95 to avoid outlier-driven C."
else:
    rec = p975
    reason = "tail is moderate; using p97.5 for utility-preserving clipping."

# 为了实验配置好看，向上取整到 5 或 10 的倍数
rounded_5 = np.ceil(rec / 5) * 5
rounded_10 = np.ceil(rec / 10) * 10

print(f"raw recommended C = {rec:.6f}")
print(f"rounded to 5      = {rounded_5:.2f}")
print(f"rounded to 10     = {rounded_10:.2f}")
print(f"reason            = {reason}")