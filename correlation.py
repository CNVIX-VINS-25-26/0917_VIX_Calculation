import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# === 读取数据 ===
df = pd.read_csv("CNVIX_vs_realized.csv")  # 或者替换为你的文件名
df = df.dropna(subset=['CNVIX', 'underlyinghisvol_30d'])

# === 计算相关系数 ===
corr, pval = pearsonr(df['CNVIX'], df['underlyinghisvol_30d'])
print(f"📊 Pearson correlation = {corr:.4f},  p-value = {pval:.4e}")

# === 绘制相关性散点图 ===
plt.figure(figsize=(7, 6))
sns.regplot(
    x='CNVIX', 
    y='underlyinghisvol_30d', 
    data=df, 
    scatter_kws={'alpha':0.6}, 
    line_kws={'color':'red', 'lw':2}
)
plt.title(f'CNVIX vs Realized 30d Historical Volatility\nPearson r = {corr:.3f}', fontsize=13)
plt.xlabel('CNVIX (Predicted Vol, %)')
plt.ylabel('Realized 30d Historical Volatility (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

output_png = "CNVIX_correlation.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"📈 相关性图已保存为: {output_png}")
plt.show()
