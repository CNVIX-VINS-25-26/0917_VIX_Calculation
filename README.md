# Star this repo if found useful, please! Thank you for supporting!
# Intro

This repo provides tools to calculate the (CN)VIX index out of given data.

## CNVIX 指数的原理

CNVIX 的编制思想与美国 CBOE VIX 类似：  
它反映未来 30 天的隐含波动率预期，是通过不同执行价期权的隐含波动率加权平均得到的“无方向波动率指标”。

### 主要逻辑
- 选定目标到期时间 $T^* = 30$ 天  
- 找到两组期权：
  - $T_1$: 剩余天数最接近但 < 30 天  
  - $T_2$: 剩余天数最接近但 > 30 天  
- 对每个到期日计算期权方差，再插值得到 30 天方差。

---

### 方差计算公式
![sigma2](https://math.vercel.app/?from=%5Cdisplaystyle%20%5Csigma%5E2%20%3D%20%5Cfrac%7B2e%5ErT%7D%7BT%7D%20%5Csum_i%20%5Cfrac%7B%5CDelta%20K_i%7D%7BK_i%5E2%7D%20Q(K_i)%20-%20%5Cfrac%7B1%7D%7BT%7D%5Cleft(%5Cfrac%7BF%7D%7BK_0%7D%20-%201%5Cright)%5E2)

其中：
- $Q(K_i)$：行权价 $K_i$ 对应期权平均价格  
- $\Delta K_i$：相邻行权价差  
- $K_0$：使行权价刚好低于远期价 $F$  
- $F$：由认购认沽平价关系得出  
  ![F](https://math.vercel.app/?from=%5Cdisplaystyle%20F%20%3D%20K_i%20%2B%20e%5ErT%20(C_i%20-%20P_i))
- $T$：年化到期时间  
  ![T](https://math.vercel.app/?from=%5Cdisplaystyle%20T%20%3D%20%5Cfrac%7BD%7D%7B365%7D)

---

### CNVIX 方差插值
将两个到期时间的方差线性插值得到 30 天：
![sigma30](https://math.vercel.app/?from=%5Cdisplaystyle%20%5Csigma_%7B30%7D%5E2%20%3D%20%5Cfrac%7BT_1%20%5Csigma_1%5E2%20(T_2-30)%20%2B%20T_2%20%5Csigma_2%5E2%20(30-T_1)%7D%7BT_2-T_1%7D)

---

### 最终指数公式
![CNVIX](https://math.vercel.app/?from=%5Cdisplaystyle%20%5Cmathrm%7BCNVIX%7D%20%3D%20100%20%5Ctimes%20%5Csqrt%7B%5Csigma_%7B30%7D%5E2%20%5Ctimes%20%5Cfrac%7B365%7D%7B30%7D%7D)

--- 

## Data Needed
| 字段名                      | 用法                           |
| ------------------------ | ---------------------------- |
| `exe_mode`               | 区分 call/put                  |
| `exe_price`              | 行权价 (K_i)                    |
| `close`                  | 当日收盘价，作为 ( Q(K_i) ) 或用于远期价估计 |
| `ptmday` 或 `ptmtradeday` | 剩余天数，用于确定 T1, T2             |
| `us_impliedvol`          | 可以辅助验证或回算隐含波动率               |
| `date` / `exe_enddate`   | 确定到期结构                       |
| （需额外）无风险利率 (r)           | 可用当日 Shibor 或固定年化 3% 近似      |

## 三、计算步骤概要（算法）

### 1. 读取 CSV 并清洗  
- 按 `date` 分组  
- 对每个日期选出最近两组到期日的期权  
- 按 `exe_price` 匹配 call / put 成对数据  

---

### 2. 计算远期价格 \( F \)
由认购认沽平价关系得到：

![F](https://math.vercel.app/?from=%5Cdisplaystyle%20F%20%3D%20K%20%2B%20e%5ErT%20(C%20-%20P))

其中：
- \( K \)：行权价  
- \( C, P \)：相同执行价下的认购、认沽期权收盘价  
- \( r \)：无风险利率  
- \( T \)：年化剩余到期时间  

选取使得 \(|C-P|\) 最小的 \(K_0\)。

---

### 3. 计算每个执行价的中间价 \( Q(K) \)
平均价定义为：

![Q](https://math.vercel.app/?from=%5Cdisplaystyle%20Q(K)%3D%5Cfrac%7BC(K)%2BP(K)%7D%7B2%7D)

或按规则：
- 当 \(K < K_0\) 时取认沽价格  
- 当 \(K > K_0\) 时取认购价格  
- 当 \(K = K_0\) 时取二者均值  

---

### 4. 计算方差 \(\sigma^2\)

核心公式：

![sigma2](https://math.vercel.app/?from=%5Cdisplaystyle%20%5Csigma%5E2%20%3D%20%5Cfrac%7B2e%5ErT%7D%7BT%7D%20%5Csum_i%20%5Cfrac%7B%5CDelta%20K_i%7D%7BK_i%5E2%7D%20Q(K_i)%20-%20%5Cfrac%7B1%7D%7BT%7D%20%5Cleft(%5Cfrac%7BF%7D%7BK_0%7D-1%5Cright)%5E2)

其中：
- \(\Delta K_i\)：相邻行权价差  
- \(Q(K_i)\)：对应中间价  
- \(F\)：远期价格  
- \(K_0\)：平值行权价  

---

### 5. 插值得到 30 天方差
对两组期限 \(T_1, T_2\) 的方差 \(\sigma_1^2, \sigma_2^2\) 线性插值：

![sigma30](https://math.vercel.app/?from=%5Cdisplaystyle%20%5Csigma_%7B30%7D%5E2%20%3D%20%5Cfrac%7BT_1%5Csigma_1%5E2(T_2-30)%2BT_2%5Csigma_2%5E2(30-T_1)%7D%7BT_2-T_1%7D)

---

### 6. 计算最终 CNVIX 指数
将方差年化并取平方根：

![CNVIX](https://math.vercel.app/?from=%5Cdisplaystyle%20%5Cmathrm%7BCNVIX%7D%20%3D%20100%20%5Ctimes%20%5Csqrt%7B%5Csigma_%7B30%7D%5E2%20%5Ctimes%20%5Cfrac%7B365%7D%7B30%7D%7D)

---

以上步骤即可从期权 CSV 数据中逐日计算得到 CNVIX 指数。


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
