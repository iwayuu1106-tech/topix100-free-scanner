#!/usr/bin/env python3
"""Runner shim for PRETEST 1.0.

This does not alter the frozen execution rules. It only makes the OHLC source
sanity check tolerant to floating-point/rounding noise and reports any genuine
price inconsistency before the realized backtest can run.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import research_v2_realized_backtest_pretest_1_0 as base


def download_frames(codes: list[str]):
    tickers = [f"{c}.T" for c in codes]
    try:
        raw = yf.download(
            tickers=tickers,
            period="6y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=True,
            repair=False,
            threads=True,
            progress=False,
            timeout=60,
        )
    except Exception as e:
        base.stop("YFINANCE_FETCH_EXCEPTION", {"error": repr(e)})
    if raw is None or raw.empty:
        base.stop("YFINANCE_FETCH_EMPTY")

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    required = ["Open", "High", "Low", "Close", "Volume"]
    frames = {}
    problems = []
    split_col_missing = []

    for code, ticker in zip(codes, tickers):
        g = base.extract_ticker_frame(raw, ticker)
        if g.empty:
            problems.append({"code": code, "problem": "no_frame"})
            continue
        miss = [x for x in required if x not in g.columns]
        if miss:
            problems.append({"code": code, "problem": "missing_columns", "columns": miss})
            continue
        use = required + (["Stock Splits"] if "Stock Splits" in g.columns else [])
        g = g[use].copy()
        if "Stock Splits" not in g.columns:
            g["Stock Splits"] = 0.0
            split_col_missing.append(code)
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = g.dropna(subset=required)
        if now_jst.time() < time(16, 0) and len(g) and g.index.max().date() == now_jst.date():
            g = g[g.index.date < now_jst.date()]
        if len(g) < 120:
            problems.append({"code": code, "problem": "under_120_rows", "rows": int(len(g))})
            continue

        row_scale = g[["Open", "High", "Low", "Close"]].abs().max(axis=1).clip(lower=1.0)
        tol = row_scale * 1e-6 + 1e-8
        row_max = g[["Open", "Close", "Low"]].max(axis=1)
        row_min = g[["Open", "Close", "High"]].min(axis=1)
        bad_mask = ((g["High"] + tol) < row_max) | ((g["Low"] - tol) > row_min) | (g["Volume"] < 0)
        bad = g.loc[bad_mask]
        if len(bad):
            sample = []
            for d, r in bad.head(5).iterrows():
                sample.append({
                    "date": d.date().isoformat(),
                    "open": float(r["Open"]), "high": float(r["High"]),
                    "low": float(r["Low"]), "close": float(r["Close"]),
                    "volume": float(r["Volume"]),
                    "split": float(r["Stock Splits"]),
                })
            problems.append({"code": code, "problem": "ohlcv_sanity", "rows": int(len(bad)), "sample": sample})
            continue
        g["ExecSMA75"] = base.split_only_sma75(g)
        frames[code] = g

    if problems:
        base.stop("PRICE_DATA_PROBLEM", {"count": len(problems), "items": problems[:100]})
    common_latest = min(g.index.max() for g in frames.values())
    frames = {c: g[g.index <= common_latest].copy() for c, g in frames.items()}
    return frames, common_latest, split_col_missing


base.download_frames = download_frames

if __name__ == "__main__":
    base.main()
