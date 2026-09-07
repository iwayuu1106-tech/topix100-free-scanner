# v1.2 実売買期待値バックテスト監査

- v1.2変更: **なし**
- 固定SHA256: `c2aef8f85dfdf125285623b7b6556b78a43509124ee09985b047b365f929d3a6`
- 結論: **正式な実現売買期待値は、固定v1.2のままでは日足OHLCだけから一意に算出できない。**
- 5/10/20/40日リターンはイベント研究の参考値としてのみ保存。実売買P/Lとは混ぜない。

## 理由
- `UP_COUNT_UNRESOLVED_CAN_BLOCK_POSITION_SIZE`: 100-share quantity under the 10-20% / 40-50% / 30-40% allocation bands cannot be fixed when wave/up-count is not unique; portfolio capital usage and annual return are therefore not deterministic.
- `SELL_INTEGRATION_NOT_FULLY_MECHANICAL`: MACD/BB/candlestick/high-retest signals are contextual and may conflict; frozen v1.2 requires SELL_DECISION=UNRESOLVED rather than inventing a universal exit, so realized holding-period P/L cannot be completed for all trades.
- `BUY_GAP_FILL_UNKNOWN`: If next-day open is above a buy stop, trigger can be known but exact fill is UNKNOWN from daily OHLC; exact P/L must be excluded.
- `SELL_STOP_GAP_FILL_UNKNOWN`: If open gaps below an active sell stop, trigger can be known but exact fill is UNKNOWN from daily OHLC; exact P/L must be excluded.
- `INTRADAY_SEQUENCE_UNKNOWN`: If multiple relevant levels are touched in one daily bar, event order is not recoverable from daily OHLC and must remain UNKNOWN.

## 4戦略の参考（validation、イベント研究）
| strategy | candidates | 20d mean | 20d win | 40d mean | 40d win |
|---|---:|---:|---:|---:|---:|
| S1_V2_FOUNDATION | 339 | 2.94% | 63.64% | 5.94% | 65.92% |
| S2_V2_PLUS_SMA75_SLOPE_GE_0_75PCT | 91 | 3.84% | 71.59% | 9.50% | 74.36% |
| S3_V2_PLUS_RS20_GE_3PCT | 21 | 5.33% | 71.43% | 7.97% | 80.00% |
| S4_V2_PLUS_BOTH | 5 | 8.27% | 80.00% | 9.09% | 75.00% |

> この表は実際のv1.2約定・損切り・売却・30万円ポートフォリオ成績ではない。
