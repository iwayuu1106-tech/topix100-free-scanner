#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import yfinance as yf

CAPITAL = 300_000
MIN_ROWS = 120
LOT = 100
PROTOCOL_SHA = "c2aef8f85dfdf125285623b7b6556b78a43509124ee09985b047b365f929d3a6"

def fail(outdir: Path, reason: str, details=None):
    obj = {
        "status": "STOP",
        "reason": reason,
        "details": details or {},
        "protocol_sha256": PROTOCOL_SHA,
        "capital_jpy": CAPITAL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "audit.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    raise SystemExit(2)

def tick_size_topix500(price: float) -> float:
    if price <= 1_000: return 0.1
    if price <= 3_000: return 0.5
    if price <= 10_000: return 1.0
    if price <= 30_000: return 5.0
    if price <= 100_000: return 10.0
    if price <= 300_000: return 50.0
    if price <= 1_000_000: return 100.0
    if price <= 3_000_000: return 500.0
    if price <= 10_000_000: return 1_000.0
    if price <= 30_000_000: return 5_000.0
    return 10_000.0

def one_tick_above(x: float) -> float:
    t = tick_size_topix500(x)
    return math.floor(x / t + 1 + 1e-12) * t

def one_tick_below(x: float) -> float:
    t = tick_size_topix500(x)
    return math.ceil(x / t - 1 - 1e-12) * t

def extract_ticker_frame(big: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(big.columns, pd.MultiIndex):
        return big.copy()
    if ticker in big.columns.get_level_values(0):
        g = big[ticker].copy()
    elif ticker in big.columns.get_level_values(1):
        g = big.xs(ticker, axis=1, level=1).copy()
    else:
        return pd.DataFrame()
    return g

def indicators(g: pd.DataFrame) -> pd.DataFrame:
    z = g.copy().sort_index()
    adj = pd.to_numeric(z["Adj Close"], errors="coerce")
    z["SMA25"] = adj.rolling(25).mean()
    z["SMA75"] = adj.rolling(75).mean()
    z["SMA75_5"] = z["SMA75"].shift(5)
    ema12 = adj.ewm(span=12, adjust=False).mean()
    ema26 = adj.ewm(span=26, adjust=False).mean()
    z["MACD"] = ema12 - ema26
    z["Signal"] = z["MACD"].ewm(span=9, adjust=False).mean()
    sd = adj.rolling(25).std(ddof=0)
    z["BB+2"] = z["SMA25"] + 2*sd
    z["BB-2"] = z["SMA25"] - 2*sd
    z["Fresh75CrossUp"] = (adj.shift(1) <= z["SMA75"].shift(1)) & (adj > z["SMA75"])
    z["FreshMACDBull"] = (z["MACD"].shift(1) <= z["Signal"].shift(1)) & (z["MACD"] > z["Signal"])
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    u = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    codes = [str(x) for x in u["codes_4digit"]]
    tickers = [f"{c}.T" for c in codes]

    try:
        raw = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
            timeout=30,
        )
    except Exception as e:
        fail(outdir, "YFINANCE_BULK_FETCH_EXCEPTION", {"error": repr(e)})

    if raw is None or raw.empty:
        fail(outdir, "YFINANCE_RETURNED_EMPTY")

    frames = {}
    gaps = []
    latest_dates = {}
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    required = ["Open","High","Low","Close","Adj Close","Volume"]

    for code, ticker in zip(codes, tickers):
        g = extract_ticker_frame(raw, ticker)
        if g.empty:
            gaps.append({"code": code, "problem": "no_frame"})
            continue
        missing_cols = [c for c in required if c not in g.columns]
        if missing_cols:
            gaps.append({"code": code, "problem": "missing_columns", "columns": missing_cols})
            continue
        g = g[required].dropna(how="all").copy()
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = g.dropna(subset=["Open","High","Low","Close","Adj Close","Volume"])
        if now_jst.time() < time(16, 0) and not g.empty and g.index.max().date() == now_jst.date():
            g = g[g.index.date < now_jst.date()]
        if len(g) < MIN_ROWS:
            gaps.append({"code": code, "problem": "history_lt_120", "rows": len(g)})
            continue
        latest_dates[code] = g.index.max().date().isoformat()
        bad = g[
            (g["High"] < g[["Open","Close","Low"]].max(axis=1)) |
            (g["Low"] > g[["Open","Close","High"]].min(axis=1)) |
            (g["Volume"] < 0)
        ]
        if len(bad):
            gaps.append({"code": code, "problem": "ohlcv_sanity_failure", "rows": len(bad)})
            continue
        frames[code] = g

    if gaps:
        fail(outdir, "UNIVERSE_DATA_GAP", {"gap_count": len(gaps), "gaps": gaps[:100]})

    unique_latest = sorted(set(latest_dates.values()))
    if len(unique_latest) != 1:
        fail(outdir, "LATEST_DATE_MISMATCH", {"latest_dates": latest_dates})
    asof = unique_latest[0]

    rows = []
    candidates = []
    for code in codes:
        z = indicators(frames[code])
        last = z.iloc[-1]
        needed = ["SMA25","SMA75","SMA75_5","MACD","Signal","BB+2","BB-2"]
        if any(pd.isna(last[x]) for x in needed):
            fail(outdir, "INDICATOR_NAN", {"code": code, "asof": asof})
        item = {
            "code": code,
            "asof": asof,
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "adj_close": float(last["Adj Close"]),
            "volume": int(last["Volume"]),
            "sma25": float(last["SMA25"]),
            "sma75": float(last["SMA75"]),
            "sma75_5": float(last["SMA75_5"]),
            "macd": float(last["MACD"]),
            "signal": float(last["Signal"]),
            "bb_plus2": float(last["BB+2"]),
            "bb_minus2": float(last["BB-2"]),
            "fresh_75_cross_up": bool(last["Fresh75CrossUp"]),
            "fresh_macd_bull_cross": bool(last["FreshMACDBull"]),
            "sma75_rising_5d": bool(last["SMA75"] > last["SMA75_5"]),
        }
        rows.append(item)
        if item["fresh_75_cross_up"]:
            item2 = item.copy()
            item2.update({
                "v1_2_candidate_state": "UNRESOLVED_UNTIL_AI_REVIEW",
                "hypothetical_trigger_if_candidate_state_is_unique": one_tick_above(item["sma75"]),
                "hypothetical_initial_stop_if_filled": one_tick_below(item["sma75"]),
                "quantity": "UNRESOLVED_UNTIL_UP_COUNT_IS_UNIQUE",
            })
            candidates.append(item2)

    pd.DataFrame(rows).to_csv(outdir/"topix100_scan.csv", index=False, encoding="utf-8-sig")

    hist_rows = []
    candidate_codes = {c["code"] for c in candidates}
    for code in sorted(candidate_codes):
        hz = indicators(frames[code]).tail(MIN_ROWS).reset_index().rename(columns={"index":"Date"})
        hz.insert(0, "code", code)
        hist_rows.append(hz)
    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_csv(outdir/"candidate_history_120d.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["code","Date"]).to_csv(outdir/"candidate_history_120d.csv", index=False, encoding="utf-8-sig")

    candidates = sorted(candidates, key=lambda x: (not x["fresh_macd_bull_cross"], not x["sma75_rising_5d"], x["code"]))
    packet = {
        "status": "DATA_LAYER_PASS",
        "source": "Yahoo Finance via yfinance, same bulk method for every monitored stock",
        "asof": asof,
        "universe_count": len(codes),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "warning": "Fresh 75MA cross is only the mechanical shortlist. v1.2 final BUY/WAIT/UNRESOLVED still requires the source-faithful AI review; no invented B-candidate or wave-count formula.",
    }
    (outdir/"candidate_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "status": "PASS_DATA_LAYER",
        "asof": asof,
        "checks": {
            "01_same_free_source_all_names": True,
            "02_all_names_have_120_rows": True,
            "03_latest_date_matches_all_names": True,
            "04_ohlcv_sanity": True,
            "05_sma25_sma75_macd_bb": True,
            "06_candidate_packet": True,
            "07_no_silent_fallback": True,
            "08_intraday_bar_excluded_before_1600_jst": True,
            "09_candidate_history_120d_saved": True,
        },
        "next_step": "Feed candidate_packet.json to the frozen v1.2 AI review for fundamentals/news, Pattern A/B, wave count, final status, 300k sizing, and order card.",
    }
    (outdir/"audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
