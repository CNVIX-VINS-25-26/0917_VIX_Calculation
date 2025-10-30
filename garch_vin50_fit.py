#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GARCH(1,1) 拟合脚本：VIN_50 vs 实现波动率
=====================================================
功能：
1. 从 VIN_50.csv 读取隐含波动率 (VIX)；
2. 从 CNVIX_vs_realized.csv 读取实现波动率 (underlyinghisvol_30d_shifted)；
3. 按日期对齐两列；
4. 对两列数据分别进行 GARCH(1,1) 拟合；
5. 输出参数、模型摘要；
6. 绘制拟合波动率曲线并保存为 GARCH_VIN50_fit.png。

作者：IrinaCtf / 2025-10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# === Step 1: 读取数据 ===
vin_file = "VIN_50.csv"
realized_file = "CNVIX_vs_realized.csv"

print("📘 正在读取数据文件...")
vin = pd.read_csv(vin_file)
realized = pd.read_csv(realized_file)

# 日期转换与排序
vin['date'] = pd.to_datetime(vin['date'])
realized['date'] = pd.to_datetime(realized['date'])
vin = vin.sort_values('date').reset_index(drop=True)
realized = realized.sort_values('date').reset_index(drop=True)

# === Step 2: 合并数据集 ===
merged = pd.merge(vin, realized[['date', 'underlyinghisvol_30d_shifted']], on='date', how='inner')
merged = merged.dropna(subset=['VIX', 'underlyinghisvol_30d_shifted']).copy()

print(f"✅ 数据对齐完成，共 {len(merged)} 条记录。")
print(merged.head())

# === Step 3: 对 VIX 和 实现波动率取对数变化 ===
merged['VIX_return'] = np.log(merged['VIX'] / merged['VIX'].shift(1)) * 100
merged['Realized_return'] = np.log(merged['underlyinghisvol_30d_shifted'] / merged['underlyinghisvol_30d_shifted'].shift(1)) * 100
merged = merged.dropna(subset=['VIX_return', 'Realized_return'])

# === Step 4: 定义GARCH拟合函数 ===
def fit_garch(series, name):
    """对单个序列进行GARCH(1,1)拟合"""
    model = arch_model(series, vol='Garch', p=1, q=1, dist='normal')
    fitted = model.fit(disp='off')
    print(f"\n📈 GARCH(1,1) 模型拟合完成 —— {name}")
    print(fitted.summary())
    return fitted.conditional_volatility, fitted.params

# === Step 5: 拟合两个序列 ===
vix_fit, vix_params = fit_garch(merged['VIX_return'], 'VIX (Implied Volatility)')
realized_fit, realized_params = fit_garch(merged['Realized_return'], 'Underlying Realized Volatility')

# === Step 6: 绘图对比 ===
plt.figure(figsize=(12, 6))
plt.plot(vix_fit.values, label='GARCH(1,1) - Fitted VIX Volatility', color='royalblue', linewidth=1.8)
plt.plot(realized_fit.values, label='GARCH(1,1) - Fitted Realized Volatility', color='orange', linestyle='--', linewidth=1.5)
plt.title('GARCH(1,1) Fitted Conditional Volatility: VIN_50 vs Realized')
plt.xlabel('Time Index')
plt.ylabel('Conditional Volatility (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_png = "GARCH_VIN50_fit.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"📊 图像已保存为: {output_png}")

# === Step 7: 输出参数 ===
print("\n🔍 GARCH(1,1) 参数对比")
print("VIX 模型参数:")
print(vix_params)
print("\nRealized 模型参数:")
print(realized_params)
print("\n🎯 拟合完成。")
