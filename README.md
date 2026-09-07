# Free TOPIX100 runner

無料データで TOPIX100 系監視銘柄の日足を同一方法で取得して、
v1.2 の前段（欠損検査・SMA25/SMA75/MACD/BB・75MA上抜けイベント候補化）を行います。

- 市場データ: Yahoo Finance / yfinance
- 実行: GitHub Actions 無料枠
- 平日 19:15 JST に自動実行
- 1銘柄でもデータ不足/日付不一致/異常なら STOP
- 別サイトで黙って穴埋めしません
- `candidate_packet.json` は BUY 指示ではなく、正式 v1.2 AIレビューに渡す候補パケットです
- v1.2 の曖昧部分（B候補完全式、波境界など）は勝手に数式化しません

出力:
- `output/audit.json`
- `output/topix100_scan.csv`
- `output/candidate_packet.json`
