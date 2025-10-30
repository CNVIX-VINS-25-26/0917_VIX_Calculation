import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("CNVIX_vs_realized.csv")
x = df['CNVIX'].dropna().values
y = df['underlyinghisvol_30d'].dropna().values

# === Step 1: 用相同的区间做直方图估计 ===
bins = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
p, _ = np.histogram(x, bins=bins, density=True)
q, _ = np.histogram(y, bins=bins, density=True)

# 避免零概率
p += 1e-12
q += 1e-12

# === Step 2: 计算 KL 散度 ===
D_pq = entropy(p, q)  # D_KL(P||Q)
D_qp = entropy(q, p)  # D_KL(Q||P)

print(f"D_KL(CNVIX || Realized) = {D_pq:.4f}")
print(f"D_KL(Realized || CNVIX) = {D_qp:.4f}")

# 可视化两个分布
plt.figure(figsize=(7,4))
plt.hist(x, bins=bins, alpha=0.5, label='CNVIX', density=True)
plt.hist(y, bins=bins, alpha=0.5, label='Realized 30d HistVol', density=True)
plt.legend()
plt.title('Distribution Comparison: CNVIX vs Realized 30d Vol')
plt.xlabel('Volatility (%)')
plt.ylabel('Density')
plt.tight_layout()

output_png = "CNVIX_distribution.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"📈 分布图已保存为: {output_png}")
plt.show()
