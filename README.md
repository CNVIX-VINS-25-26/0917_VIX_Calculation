# Star this repo if found useful, please! Thank you for supporting!
# Intro

This repo provides tools to calculate the (CN)VIX index out of given data.

## CNVIX (上证50ETF波动率指数) 计算原理

CNVIX 的核心计算公式来源于《上证50ETF波动率指数编制方案》：

![CNVIX公式](https://latex.codecogs.com/svg.image?\sigma^2=\frac{2e^{rT}}{T}\sum_i\frac{\Delta K_i}{K_i^2}Q(K_i)-\frac{1}{T}\left(\frac{F}{K_0}-1\right)^2)

其中：

- \( Q(K_i) \)：行权价 \(K_i\) 对应期权平均价；
- \( \Delta K_i \)：相邻行权价差；
- \( K_0 \)：平值期权的执行价；
- \( F = K_i + e^{rT}(C_i - P_i) \)：远期价格；
- 最后通过 30 天插值得到年化波动率：

![插值公式](https://latex.codecogs.com/svg.image?\sigma_{30}^2=\frac{T_1\sigma_1^2(T_2-30)+T_2\sigma_2^2(30-T_1)}{T_2-T_1})

## Confidentiality
Modification of the  `.gitignore` file must be careful. It prevents the commits of confidential data.

## Usage
1. Set up the repo: 
```
git clone https://github.com/IrinaCtf/VINS_VIX_Calculation.git
```

2. Put the data file into the same folder as `calc_cnvix.py`(e.g.: `option_50ETF_all.csv`). If using a different file name, add it in `.gitignore` for security reason.

3. Install necessary python packages and run the calculation program
```
pip install -r requirements.txt
python3 calc_cnvix.py
```

4. Read the output:
* `CNVIX_daily.csv`
* `CNVIC_plot.png`

## Useful Information
* `code`：合约代码（Security code）。
* `date`：行情日期（Quote date，YYYY-MM-DD）。
* `sec_name`：合约简称（Security name），如“50ETF购2015年3月2.20”。
* `exe_mode`：期权类型（Option type）— call=认购、put=认沽。
* `exe_price`：执行价/行权价（Strike price），单位：元。
* `exe_enddate`：到期日/最后到期日（Expiration date）。
* `close`：收盘价（Close price），单位：元/份。
* `volume`：当日成交量（Trading volume），单位：张或份（视数据源定义，一般张）。
* `ptmtradeday`：距离到期的剩余交易日（Days to maturity – trading days）。
* `ptmday`：距离到期的剩余自然日（Days to maturity – calendar days）。
* `theoryvalue`：理论价格（Theoretical price），通常用 Black-Scholes 等模型按给定波动率计算的期权公允价，单位：元。
* `delta`：Δ，期权价格对标的价格的一阶敏感度（per 1 unit move in underlying）。
* `gamma`：Γ，Δ 对标的价格的敏感度（二阶，per 1 unit underlying）；常以“每元”的量纲表示。
* `vega`：Vega，期权价格对波动率的敏感度（per 1% vol change，一般表示为每 1 个波动率点变化时价格变动的元数）。
* `theta`：Θ，期权价格对时间（一天流逝）的敏感度（per day，通常为每日时间价值损耗，认购多为负）。
* `rho`：Rho，期权价格对无风险利率的敏感度（per 1% interest rate change）。
* `underlyinghisvol_30d`：标的 30 日历史波动率（Underlying 30-day historical volatility，年化，%）。
* `us_hisvol`：标的历史波动率（Underlying historical volatility，年化，%）。通常为数据提供方的基准口径（窗口或算法可能与 30/90 日不同）。
* `underlyinghisvol_90d`：标的 90 日历史波动率（Underlying 90-day historical volatility，年化，%）。
* `us_impliedvol`：该合约对应的隐含波动率（Implied volatility，年化，%），多以收盘价反推。
* `oi`：未平仓量（Open interest），单位：张（合约持仓数）。
* `startdate`：该期权系列的上市日期/起始日期（Listing/start date）。
### 小提示：
* `ptmtradeday` 与 `ptmday` 的差异在于是否只计交易日；在做到期加权或年化换算时要选对。
* Greeks 的单位在不同数据源可能有缩放（例如 vega/100、rho/100）；你在计算和校验时应结合 theoryvalue 与 us_impliedvol 做一次“反推”确认其量纲。
* 波动率列通常以百分数记录，但在 CSV 中多为百分比数值（例如 0.31 表示 31% 或 31 表示 31%，需结合上下文核对）。
