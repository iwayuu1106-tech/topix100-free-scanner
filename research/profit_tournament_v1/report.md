# Profit Tournament V1

- Tested: **72** daily-rule configurations
- Discovery: 2020-2022 selects one exit per entry family
- Holdout: 2023-2025 ranks the 12 family champions
- Capital: ¥300,000; risk: 1% current equity; one position; 100-share lots; cost 0.10% base / 0.20% stress

## Holdout winner
- Entry: **golden25_75**
- Exit: **sma20_break**
- Trades: **22**
- Ending equity: **¥391512**
- Net P/L: **¥91512**
- Win rate: **50.00%**
- Expectancy: **1.537R/trade**
- PF: **3.655687783338533**
- Max DD: **-8.28%**
- Stress ending equity: **¥388036**

## Top 5 holdout
1. golden25_75 + sma20_break: ¥391512 (P/L ¥91512, n=22, PF=3.655687783338533)
2. breakout55 + sma20_break: ¥329315 (P/L ¥29315, n=25, PF=1.5017782011100973)
3. ma75_reclaim + sma75_break: ¥319916 (P/L ¥19916, n=18, PF=1.6848261547205696)
4. near52w_high + trail_4ATR: ¥319295 (P/L ¥19295, n=10, PF=2.4597258163999647)
5. breakout20_vol15 + sma75_break: ¥318109 (P/L ¥18109, n=18, PF=1.4161486424450935)

## Descriptive full-sample oracle (NOT selection winner)
- ma75_reclaim + sma75_break: ¥404497 (P/L ¥104497)

## Limitations
- Current 2026 TOPIX100 constituents are applied historically; survivorship/constituent bias remains.
- Yahoo/yfinance daily data is used; this is not the official J-Quants all-TSE test.
- ORB and PEAD are not included in this daily-price tournament because the required six-year intraday/fundamental event dataset is unavailable here.