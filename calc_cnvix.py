#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNVIX (上证50ETF波动率指数) 全流程计算 + 实现波动率对比
============================================================
功能：
1️⃣ 从 option_50ETF_all.csv 读取数据；
2️⃣ 按交易日计算 CNVIX；
3️⃣ 输出 CNVIX_daily.csv；
4️⃣ 绘制 CNVIX 时间序列图；
5️⃣ 自动提取 underlyinghisvol_30d（标的30日历史波动率），与 CNVIX 对齐比较；
6️⃣ 输出 CNVIX_vs_realized.csv；
7️⃣ 绘制 CNVIX 与 30日实现波动率对比图。

参考文档：《上证50ETF波动率指数编制方案》

作者：IrinaCtf / 2025-10
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

# === Step 0: 参数设置 ===
INPUT_FILE = "option_50ETF_all.csv"
OUTPUT_FILE = "CNVIX_daily.csv"
OUTPUT_PLOT = "CNVIX_plot.png"
OUTPUT_FILE_COMPARE = "CNVIX_vs_realized.csv"
OUTPUT_PLOT_COMPARE = "CNVIX_vs_realized.png"

RISK_FREE_RATE = 0.03
TRADING_DAYS_PER_YEAR = 252
TARGET_TRADING_DAYS = 30

# === Step 1: 读取与预处理数据 ===
print("📘 正在读取原始数据...")
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df['exe_enddate'] = pd.to_datetime(df['exe_enddate'])
df = df[df['exe_mode'].isin(['call', 'put'])]  # 仅保留看涨和看跌期权
df = df.dropna(subset=['exe_price', 'close', 'ptmday'])

# 检查是否包含实现波动率列
if 'underlyinghisvol_30d' not in df.columns:
    raise ValueError("❌ 数据文件中缺少列 'underlyinghisvol_30d'，请确认输入文件正确。")

# ----------------------------------------------------------
# 交易日索引化：严格按“交易日序号”而非自然日
# ----------------------------------------------------------
trading_days = np.sort(df['date'].unique())
trading_day_to_idx = {day: idx for idx, day in enumerate(trading_days)}
df['trade_day_idx'] = df['date'].map(trading_day_to_idx)

expiry_idx = df['exe_enddate'].map(trading_day_to_idx)
missing_expiry = expiry_idx.isna()
if missing_expiry.any():
    insertion_pos = pd.Series(
        np.searchsorted(trading_days, df.loc[missing_expiry, 'exe_enddate'].values),
        index=expiry_idx.index[missing_expiry],
        dtype=float
    )
    insertion_pos[insertion_pos >= len(trading_days)] = np.nan
    expiry_idx.loc[missing_expiry] = insertion_pos

df['expiry_trade_idx'] = expiry_idx
df = df.dropna(subset=['expiry_trade_idx'])
df['expiry_trade_idx'] = df['expiry_trade_idx'].astype(int)
df['tdm_trading'] = df['expiry_trade_idx'] - df['trade_day_idx']
df = df[df['tdm_trading'] > 0]

# 类型转换
df['exe_price'] = df['exe_price'].astype(float)
df['close'] = df['close'].astype(float)
df['ptmday'] = df['ptmday'].astype(float)

print(f"✅ 数据预处理完成，共 {len(df)} 条有效记录。")

# === Step 2: 定义单日 CNVIX 计算函数 ===
def calc_cnvix_for_date(df_day):
    """计算单个交易日的 CNVIX"""
    maturity_days = df_day.groupby('exe_enddate')['tdm_trading'].mean().sort_values()
    if len(maturity_days) < 2:
        return np.nan

    idx = (maturity_days - TARGET_TRADING_DAYS).abs().argsort()[:2]
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

        T = sub['tdm_trading'].iloc[0] / TRADING_DAYS_PER_YEAR
        r = RISK_FREE_RATE

        merged['F_temp'] = merged['exe_price'] + np.exp(r*T) * (merged['close_call'] - merged['close_put'])
        F = merged.loc[(merged['close_call'] - merged['close_put']).abs().idxmin(), 'F_temp']
        K0 = merged.loc[merged['exe_price'] <= F, 'exe_price'].max()

        merged['Q'] = np.where(
            merged['exe_price'] < K0, merged['close_put'],
            np.where(merged['exe_price'] > K0, merged['close_call'],
                     0.5 * (merged['close_call'] + merged['close_put']))
        )
        merged = merged.sort_values('exe_price').reset_index(drop=True)
        merged['ΔK'] = merged['exe_price'].diff().bfill()

        sigma2 = (2 * np.exp(r*T) / T) * np.sum(merged['ΔK'] / merged['exe_price']**2 * merged['Q']) \
                 - (1 / T) * ((F / K0 - 1)**2)

        results.append((T, sigma2))

    if len(results) < 2:
        return np.nan

    (T1, s1), (T2, s2) = sorted(results, key=lambda x: x[0])
    T_target = TARGET_TRADING_DAYS / TRADING_DAYS_PER_YEAR
    sigma2_30 = (T1 * s1 * (T2 - T_target) + T2 * s2 * (T_target - T1)) / (T2 - T1)
    CNVIX = 100 * math.sqrt(sigma2_30 * TRADING_DAYS_PER_YEAR / TARGET_TRADING_DAYS)
    return CNVIX

# === Step 3: 按日期批量计算 CNVIX ===
print("⚙️ 开始计算 CNVIX ...")
vix_list = []
for date, df_day in df.groupby('date'):
    try:
        vix_value = calc_cnvix_for_date(df_day)
        # 提取实现波动率
        realized = df_day['underlyinghisvol_30d'].mean()
        vix_list.append({'date': date, 'CNVIX': vix_value, 'underlyinghisvol_30d': realized})
    except Exception as e:
        print(f"❌ Error at {date}: {e}")
        continue

result = pd.DataFrame(vix_list).dropna().sort_values('date')
result.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')
print(f"✅ CNVIX计算完成，共 {len(result)} 个交易日。结果已保存至: {OUTPUT_FILE}")

# === Step 4: 绘制 CNVIX 时间序列图 ===
print("📈 绘制 CNVIX 趋势图 ...")
plt.figure(figsize=(10, 5))
plt.plot(result['date'].values, result['CNVIX'].values, label='CNVIX (Implied Volatility)', color='royalblue')
plt.title('China 50ETF Implied Volatility Index (CNVIX)')
plt.xlabel('Date')
plt.ylabel('Volatility Index (annualized, %)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"🖼️ 已保存 CNVIX_plot.png")

# === Step 5: CNVIX vs Realized Volatility 对比 ===
print("🔍 生成 CNVIX 与 30日实现波动率 对比数据 ...")

# 实现波动率右移30个交易日（使其与CNVIX的“预期”窗口对齐）
result['underlyinghisvol_30d_shifted'] = result['underlyinghisvol_30d'].shift(-30)

compare = result.dropna(subset=['CNVIX', 'underlyinghisvol_30d_shifted']).copy()
compare.to_csv(OUTPUT_FILE_COMPARE, index=False, float_format='%.4f')
print(f"✅ 已生成对齐文件: {OUTPUT_FILE_COMPARE}")

# === Step 6: 绘制对比图 ===
print("📊 绘制 CNVIX vs Realized Volatility 图像 ...")
plt.figure(figsize=(10, 5))
plt.plot(compare['date'].values, compare['CNVIX'].values, label='CNVIX (Implied)', linewidth=1.8, color='steelblue')
plt.plot(compare['date'].values, compare['underlyinghisvol_30d_shifted'].values,
         label='Underlying 30d Realized Volatility (+30 days shifted)',
         linestyle='--', linewidth=1.3, color='orange')


plt.title('CNVIX vs 30-day Shifted Historical Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility Index (annualized, %)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_COMPARE, dpi=300, bbox_inches='tight')
print(f"📈 图像已保存为: {OUTPUT_PLOT_COMPARE}")

# === Step 7: 结束 ===
print("🎯 全流程完成。生成文件：")
print(f"  - {OUTPUT_FILE}")
print(f"  - {OUTPUT_PLOT}")
print(f"  - {OUTPUT_FILE_COMPARE}")
print(f"  - {OUTPUT_PLOT_COMPARE}")
