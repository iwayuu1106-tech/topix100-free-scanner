#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)
MIN_ROWS = 120
CAPITAL = 300000
LOT = 100


def extract_ticker_frame(big: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(big.columns, pd.MultiIndex):
        return big.copy()
    if ticker in big.columns.get_level_values(0):
        return big[ticker].copy()
    if ticker in big.columns.get_level_values(1):
        return big.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    z = g.copy().sort_index()
    px = pd.to_numeric(z["Adj Close"], errors="coerce")
    z["SMA25"] = px.rolling(25).mean()
    z["SMA75"] = px.rolling(75).mean()
    z["SMA75_5"] = z["SMA75"].shift(5)
    z["Fresh75CrossUp"] = (px.shift(1) <= z["SMA75"].shift(1)) & (px > z["SMA75"])
    return z


def main():
    universe = json.loads(Path("universe.json").read_text(encoding="utf-8"))
    codes = [str(x).zfill(4) for x in universe["codes_4digit"]]
    tickers = [f"{c}.T" for c in codes]
    raw = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
        timeout=60,
    )
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    required = ["Open","High","Low","Close","Adj Close","Volume"]
    rows = []
    gaps = []
    for code, ticker in zip(codes, tickers):
        g = extract_ticker_frame(raw, ticker)
        if g.empty:
            gaps.append({"code":code,"problem":"no_frame"}); continue
        miss = [c for c in required if c not in g.columns]
        if miss:
            gaps.append({"code":code,"problem":"missing_columns","columns":miss}); continue
        g = g[required].dropna(how="all").copy()
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = g.dropna(subset=required)
        if len(g) < MIN_ROWS:
            gaps.append({"code":code,"problem":"history_lt_120","rows":len(g)}); continue
        z = add_indicators(g)
        last = z.iloc[-1]
        prev = z.iloc[-2]
        if any(pd.isna(last[x]) for x in ["SMA25","SMA75","SMA75_5"]):
            gaps.append({"code":code,"problem":"indicator_nan"}); continue
        asof = z.index[-1].date().isoformat()
        close = float(last["Adj Close"])
        sma25 = float(last["SMA25"])
        sma75 = float(last["SMA75"])
        sma75_5 = float(last["SMA75_5"])
        fresh = bool(last["Fresh75CrossUp"])
        foundation = bool(fresh and sma75 > sma75_5 and sma25 > sma75 and close <= sma25)
        lot_cost = float(last["Close"]) * LOT
        affordable = lot_cost <= CAPITAL
        rows.append({
            "code": code,
            "asof": asof,
            "snapshot_generated_jst": now_jst.isoformat(),
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "adj_close": close,
            "volume": int(last["Volume"]),
            "sma25": sma25,
            "sma75": sma75,
            "sma75_5": sma75_5,
            "prev_adj_close": float(prev["Adj Close"]),
            "fresh_75_cross_up": fresh,
            "sma75_rising_5d": bool(sma75 > sma75_5),
            "sma25_above_sma75": bool(sma25 > sma75),
            "price_le_sma25": bool(close <= sma25),
            "v2_foundation": foundation,
            "lot_cost_100_shares_jpy": lot_cost,
            "affordable_with_300k": affordable,
        })
    if gaps:
        status = "PARTIAL_WITH_GAPS"
    else:
        status = "PASS_INTRADAY_SNAPSHOT"
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"intraday_v2_scan.csv", index=False, encoding="utf-8-sig")
    strict = df[df["v2_foundation"]].sort_values(["affordable_with_300k","code"], ascending=[False,True]) if len(df) else df
    packet = {
        "status": status,
        "warning": "INTRADAY PROVISIONAL: current-day daily bar is incomplete and can change before the 15:30 close. This is not the official close-confirmed strategy signal.",
        "snapshot_generated_jst": now_jst.isoformat(),
        "universe_count": len(codes),
        "rows_ok": len(rows),
        "gap_count": len(gaps),
        "gaps": gaps,
        "latest_dates": sorted(df["asof"].unique().tolist()) if len(df) else [],
        "v2_foundation_candidate_count": int(len(strict)),
        "v2_foundation_candidates": strict.to_dict(orient="records") if len(strict) else [],
    }
    (OUT/"intraday_v2_candidates.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(packet, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
