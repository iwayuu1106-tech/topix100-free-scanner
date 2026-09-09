# FROZEN-2 PRETEST 1.0

- Spec SHA256: `f12bc8d4271bb43856c6066a47aa7df95b8a244a76b01727606fb7a114b92914`
- Rules were frozen before this backtest.
- Universe proxy: current monitored TOPIX100 (99/99 usable), not historical all-TSE.
- Data: Yahoo Finance/yfinance actual Japanese OHLCV; split handling reconstructed to historical raw yen/share.

## Base cost 0.10% round trip

- Completed trades: **40**
- Win rate: **25.00%**
- Avg gain: **¥8349 / 3.81R**
- Avg loss: **¥-2622 / -1.01R**
- Avg win/loss R ratio: **3.77x**
- Expectancy: **¥121 / 0.19R per trade**
- Profit factor: **1.0614465608927806**
- Max drawdown: **-11.22%**
- Max losing streak: **6**
- Avg / median / max holding sessions: **11.62 / 7.00 / 60**
- Largest winner: **¥26229 / 11.69R**
- Largest loser: **¥-4854 / -1.79R**
- Ending equity: **¥304833**

## Stress cost 0.20%
- Expectancy R: **0.16R**
- Profit factor: **1.030533031377082**
- Ending equity: **¥302441**

## Temporal split
- 2020-2022 expectancy R: **0.27R**, PF **1.2132471330209336**
- 2023-2025 expectancy R: **0.11R**, PF **0.936818700698862**

## Winner concentration
- Top1 share of gross profits: **31.42%**
- Top5 share of gross profits: **84.67%**
- Top10 share of gross profits: **100.00%**
- Remove top1: **{"removed": 1, "remaining_n": 39, "total_pnl_jpy": -21395.594583129914, "expectancy_jpy": -548.6049893110235, "expectancy_R": -0.10088979409354938, "profit_factor": 0.7279804198394771}**
- Remove top3: **{"removed": 3, "remaining_n": 37, "total_pnl_jpy": -54601.40087073194, "expectancy_jpy": -1475.713537046809, "expectancy_R": -0.47362372556629495, "profit_factor": 0.3058080212108719}**
- Remove top5: **{"removed": 5, "remaining_n": 35, "total_pnl_jpy": -65856.6042934963, "expectancy_jpy": -1881.6172655284659, "expectancy_R": -0.6735302026218262, "profit_factor": 0.16271147403215086}**

## Verdict
- Numeric verdict: **保留**
- Final verdict after evidence-quality cap: **保留**

The evidence-quality cap is pre-registered because this run is not historical all-TSE J-Quants.