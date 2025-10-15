#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNVIX (上证50ETF波动率指数) 计算脚本
参考文档: 《上证50ETF波动率指数编制方案》
输入: option_50ETF_all.csv
输出: CNVIX_daily.csv
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime

# === 参数设置 ===
INPUT_FILE = "option_50ETF_all.csv"
OUTPUT_FILE = "CNVIX_daily.csv"
RISK_FREE_RATE = 0.03  # 无风险利率近似（年化）

# === Step 1: 读取与预处理 ===
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df['exe_enddate'] = pd.to_datetime(df['exe_enddate'])
df = df[df['exe_mode'].isin(['call', 'put'])]  # 确保数据干净
df = df.dropna(subset=['exe_price', 'close', 'ptmday'])

# 转换类型
df['exe_price'] = df['exe_price'].astype(float)
df['close'] = df['close'].astype(float)
df['ptmday'] = df['ptmday'].astype(float)

# === Step 2: 定义计算函数 ===
def calc_cnvix_for_date(df_day):
    """对单个交易日计算CNVIX"""
    # 找出各到期日剩余天数
    maturity_days = df_day.groupby('exe_enddate')['ptmday'].mean().sort_values()
    if len(maturity_days) < 2:
        return np.nan  # 少于两组到期日无法计算

    # 选择最接近30天的两个到期日
    idx = (maturity_days - 30).abs().argsort()[:2]
    maturity_list = maturity_days.index[idx].sort_values()

    results = []
    for expiry in maturity_list:
        sub = df_day[df_day['exe_enddate'] == expiry]
        if sub['exe_mode'].nunique() < 2:
            continue

        calls = sub[sub['exe_mode'] == 'call'][['exe_price', 'close']]
        puts = sub[sub['exe_mode'] == 'put'][['exe_price', 'close']]
        merged = pd.merge(calls, puts, on='exe_price', suffixes=('_call', '_put')).dropna()
        if merged.empty:
            continue

        T = sub['ptmday'].iloc[0] / 365
        r = RISK_FREE_RATE

        # 计算远期价格 F
        merged['F_temp'] = merged['exe_price'] + np.exp(r*T) * (merged['close_call'] - merged['close_put'])
        F = merged.loc[(merged['close_call'] - merged['close_put']).abs().idxmin(), 'F_temp']
        K0 = merged.loc[merged['exe_price'] <= F, 'exe_price'].max()

        # 中间价 Q(K)
        merged['Q'] = np.where(
            merged['exe_price'] < K0, merged['close_put'],
            np.where(merged['exe_price'] > K0, merged['close_call'],
                     0.5 * (merged['close_call'] + merged['close_put']))
        )

        merged = merged.sort_values('exe_price').reset_index(drop=True)
        merged['ΔK'] = merged['exe_price'].diff().bfill()

        # 按公式计算方差
        sigma2 = (2 * np.exp(r*T) / T) * np.sum(merged['ΔK'] / merged['exe_price']**2 * merged['Q']) \
                 - (1 / T) * ((F / K0 - 1)**2)

        results.append((T, sigma2))

    if len(results) < 2:
        return np.nan

    # 线性插值至30天
    (T1, s1), (T2, s2) = sorted(results, key=lambda x: x[0])
    T_target = 30 / 365
    sigma2_30 = (T1 * s1 * (T2 - T_target) + T2 * s2 * (T_target - T1)) / (T2 - T1)

    CNVIX = 100 * math.sqrt(sigma2_30 * 365 / 30)
    return CNVIX

# === Step 3: 按日期批量计算 ===
vix_list = []
for date, df_day in df.groupby('date'):
    try:
        vix_value = calc_cnvix_for_date(df_day)
        vix_list.append({'date': date, 'CNVIX': vix_value})
    except Exception as e:
        print(f"Error at {date}: {e}")
        continue

result = pd.DataFrame(vix_list).dropna().sort_values('date')
result.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')

print(f"✅ CNVIX计算完成，共 {len(result)} 个交易日。结果已保存至: {OUTPUT_FILE}")
print(result.head())

# === 可选: 绘图展示 ===
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    # plt.plot(result['date'], result['CNVIX'], label='CNVIX (calculated)')
    plt.plot(result['date'].values, result['CNVIX'].values, label='CNVIX (calculated)')
    plt.title('China 50ETF Implied Volatility Index (CNVIX)')
    plt.xlabel('Date')
    plt.ylabel('Volatility Index (annualized, %)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像文件
    output_png = "CNVIX_plot.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"📈 图像已保存为: {output_png}")

    # 可选：是否显示
    # plt.show()

except ImportError:
    print("Matplotlib 未安装，跳过绘图。")
