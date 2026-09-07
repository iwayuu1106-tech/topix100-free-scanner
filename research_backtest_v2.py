#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

PROTOCOL_SHA = "c2aef8f85dfdf125285623b7b6556b78a43509124ee09985b047b365f929d3a6"
HORIZONS = [5, 10, 20, 40]
MIN_YEARS = 3.0
MIN_ROWS = 120
MIN_GROUP_N = 40
OUT = Path("research")
OUT.mkdir(parents=True, exist_ok=True)


def stop(reason: str, details=None):
    obj = {
        "status": "STOP",
        "reason": reason,
        "details": details or {},
        "protocol_sha256": PROTOCOL_SHA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    raise SystemExit(2)


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
    adj = pd.to_numeric(z["Adj Close"], errors="coerce")
    close = pd.to_numeric(z["Close"], errors="coerce")
    factor = adj / close.replace(0, np.nan)
    z["AdjHigh"] = pd.to_numeric(z["High"], errors="coerce") * factor
    z["AdjLow"] = pd.to_numeric(z["Low"], errors="coerce") * factor
    z["SMA25"] = adj.rolling(25).mean()
    z["SMA75"] = adj.rolling(75).mean()
    z["SMA75_5"] = z["SMA75"].shift(5)
    ema12 = adj.ewm(span=12, adjust=False).mean()
    ema26 = adj.ewm(span=26, adjust=False).mean()
    z["MACD"] = ema12 - ema26
    z["Signal"] = z["MACD"].ewm(span=9, adjust=False).mean()
    z["VolPrev20"] = pd.to_numeric(z["Volume"], errors="coerce").shift(1).rolling(20).mean()
    z["VolRatio20"] = pd.to_numeric(z["Volume"], errors="coerce") / z["VolPrev20"]
    z["Cross75Up"] = (adj.shift(1) <= z["SMA75"].shift(1)) & (adj > z["SMA75"])
    z["SMA75Slope5Pct"] = z["SMA75"] / z["SMA75_5"] - 1.0
    return z


def event_row(code: str, z: pd.DataFrame, i: int) -> dict:
    r = z.iloc[i]
    base = float(r["Adj Close"])
    d = {
        "code": code,
        "date": z.index[i].date().isoformat(),
        "event_adj_close": base,
        "sma25": float(r["SMA25"]),
        "sma75": float(r["SMA75"]),
        "sma75_5": float(r["SMA75_5"]),
        "sma75_slope5_pct": float(r["SMA75Slope5Pct"]),
        "macd": float(r["MACD"]),
        "signal": float(r["Signal"]),
        "volume": int(r["Volume"]),
        "volume_ratio20": float(r["VolRatio20"]) if pd.notna(r["VolRatio20"]) else np.nan,
        "sma75_rising": bool(r["SMA75"] > r["SMA75_5"]),
        "macd_above_signal": bool(r["MACD"] > r["Signal"]),
        "macd_above_zero": bool(r["MACD"] > 0),
        "price_above_sma25": bool(r["Adj Close"] > r["SMA25"]),
        "sma25_above_sma75": bool(r["SMA25"] > r["SMA75"]),
    }
    for h in HORIZONS:
        if i + h < len(z):
            future = z.iloc[i + 1:i + h + 1]
            endp = float(z.iloc[i + h]["Adj Close"])
            d[f"ret_{h}d"] = endp / base - 1.0
            d[f"max_up_{h}d"] = float(future["AdjHigh"].max() / base - 1.0)
            d[f"max_down_{h}d"] = float(future["AdjLow"].min() / base - 1.0)
        else:
            d[f"ret_{h}d"] = np.nan
            d[f"max_up_{h}d"] = np.nan
            d[f"max_down_{h}d"] = np.nan
    return d


def metric_block(df: pd.DataFrame, h: int) -> dict:
    s = pd.to_numeric(df[f"ret_{h}d"], errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    pos = s[s > 0]
    neg = s[s < 0]
    mup = pd.to_numeric(df.loc[s.index, f"max_up_{h}d"], errors="coerce").dropna()
    mdn = pd.to_numeric(df.loc[s.index, f"max_down_{h}d"], errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "win_rate": float((s > 0).mean()),
        "mean_return": float(s.mean()),
        "median_return": float(s.median()),
        "avg_gain": float(pos.mean()) if len(pos) else None,
        "avg_loss": float(neg.mean()) if len(neg) else None,
        "mean_max_up": float(mup.mean()) if len(mup) else None,
        "mean_max_down": float(mdn.mean()) if len(mdn) else None,
        "best_max_up": float(mup.max()) if len(mup) else None,
        "worst_max_down": float(mdn.min()) if len(mdn) else None,
    }


def summarize_group(df: pd.DataFrame, feature: str, group: str) -> dict:
    return {
        "feature": feature,
        "group": group,
        "event_count": int(len(df)),
        **{f"h{h}": metric_block(df, h) for h in HORIZONS},
    }


def candidate_score(group_summary: dict, baseline: dict) -> tuple[float, dict]:
    improvements = {}
    score = 0.0
    positive_horizons = 0
    win_positive_horizons = 0
    for h in HORIZONS:
        g = group_summary[f"h{h}"]
        b = baseline[f"h{h}"]
        if not g.get("n") or not b.get("n"):
            continue
        dr = g["mean_return"] - b["mean_return"]
        dw = g["win_rate"] - b["win_rate"]
        improvements[f"{h}d_mean_return_delta"] = dr
        improvements[f"{h}d_win_rate_delta"] = dw
        if dr > 0:
            positive_horizons += 1
        if dw > 0:
            win_positive_horizons += 1
        score += dr * (1.0 if h in [20, 40] else 0.5) + dw * 0.15
    n40 = group_summary["h40"].get("n", 0)
    if n40 < MIN_GROUP_N:
        score -= 0.25
    return score, {
        "positive_return_horizons": positive_horizons,
        "positive_winrate_horizons": win_positive_horizons,
        "n40": n40,
        "improvements_vs_all_events": improvements,
    }


def main():
    universe = json.loads(Path("universe.json").read_text(encoding="utf-8"))
    codes = [str(x) for x in universe["codes_4digit"]]
    tickers = [f"{c}.T" for c in codes]

    try:
        raw = yf.download(
            tickers=tickers,
            period="6y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=True,
            progress=False,
            timeout=60,
        )
    except Exception as e:
        stop("YFINANCE_6Y_FETCH_EXCEPTION", {"error": repr(e)})
    if raw is None or raw.empty:
        stop("YFINANCE_6Y_EMPTY")

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frames_raw = {}
    hard_missing = []
    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))

    for code, ticker in zip(codes, tickers):
        g = extract_ticker_frame(raw, ticker)
        if g.empty:
            hard_missing.append({"code": code, "problem": "no_frame"})
            continue
        miss_cols = [c for c in required if c not in g.columns]
        if miss_cols:
            hard_missing.append({"code": code, "problem": "missing_columns", "columns": miss_cols})
            continue
        g = g[required].dropna(how="all").copy()
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = g.dropna(subset=required)
        if now_jst.hour < 16 and len(g) and g.index.max().date() == now_jst.date():
            g = g[g.index.date < now_jst.date()]
        if len(g) < MIN_ROWS:
            hard_missing.append({"code": code, "problem": "under_120_rows", "rows": int(len(g))})
            continue
        frames_raw[code] = g

    if hard_missing:
        stop("UNIVERSE_HISTORY_HARD_GAP", {"count": len(hard_missing), "items": hard_missing})

    common_latest = min(g.index.max() for g in frames_raw.values())
    frames = {}
    coverage = []
    short_coverage = []
    strict_codes = []
    for code, g in frames_raw.items():
        g = g[g.index <= common_latest].copy()
        years = (g.index.max() - g.index.min()).days / 365.25
        item = {
            "code": code,
            "first_date": g.index.min().date().isoformat(),
            "last_date": g.index.max().date().isoformat(),
            "rows": int(len(g)),
            "years": float(years),
            "meets_3y": bool(years >= MIN_YEARS),
        }
        coverage.append(item)
        if years >= MIN_YEARS:
            strict_codes.append(code)
        else:
            short_coverage.append(item)
        frames[code] = add_indicators(g)

    event_cutoff = common_latest - pd.DateOffset(years=5)
    all_events = []
    for code, z in frames.items():
        idxs = np.where(z["Cross75Up"].fillna(False).values)[0]
        for i in idxs:
            if z.index[i] < event_cutoff:
                continue
            needed = ["SMA25", "SMA75", "SMA75_5", "MACD", "Signal", "VolRatio20", "AdjHigh", "AdjLow"]
            if any(pd.isna(z.iloc[i][x]) for x in needed):
                continue
            all_events.append(event_row(code, z, int(i)))

    if not all_events:
        stop("NO_75MA_CROSS_EVENTS_FOUND")

    ev_all = pd.DataFrame(all_events).sort_values(["date", "code"]).reset_index(drop=True)
    ev_strict = ev_all[ev_all["code"].isin(strict_codes)].copy().reset_index(drop=True)
    if ev_strict.empty:
        stop("NO_STRICT_3YPLUS_EVENTS")

    ev_all.to_csv(OUT / "events_all_available.csv", index=False, encoding="utf-8-sig")
    ev_strict.to_csv(OUT / "events_strict_3yplus.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False, encoding="utf-8-sig")

    baseline = {f"h{h}": metric_block(ev_strict, h) for h in HORIZONS}
    baseline_all_available = {f"h{h}": metric_block(ev_all, h) for h in HORIZONS}

    groups = []
    bool_features = [
        "sma75_rising",
        "macd_above_signal",
        "macd_above_zero",
        "price_above_sma25",
        "sma25_above_sma75",
    ]
    for f in bool_features:
        groups.append(summarize_group(ev_strict[ev_strict[f] == True], f, "TRUE"))
        groups.append(summarize_group(ev_strict[ev_strict[f] == False], f, "FALSE"))

    vol_groups = {
        "VOL_LT_1.0X": ev_strict[ev_strict["volume_ratio20"] < 1.0],
        "VOL_1.0_TO_1.5X": ev_strict[(ev_strict["volume_ratio20"] >= 1.0) & (ev_strict["volume_ratio20"] < 1.5)],
        "VOL_GE_1.5X": ev_strict[ev_strict["volume_ratio20"] >= 1.5],
    }
    for name, g in vol_groups.items():
        groups.append(summarize_group(g, "volume_ratio20", name))

    combined_masks = {
        "SMA75_RISING_AND_MACD_ABOVE_SIGNAL": ev_strict["sma75_rising"] & ev_strict["macd_above_signal"],
        "SMA75_RISING_AND_PRICE_ABOVE_SMA25": ev_strict["sma75_rising"] & ev_strict["price_above_sma25"],
        "SMA75_RISING_AND_SMA25_ABOVE_SMA75": ev_strict["sma75_rising"] & ev_strict["sma25_above_sma75"],
        "SMA75_RISING_AND_VOL_GE_1.0X": ev_strict["sma75_rising"] & (ev_strict["volume_ratio20"] >= 1.0),
        "SMA75_RISING_AND_MACD_ABOVE_SIGNAL_AND_VOL_GE_1.0X": ev_strict["sma75_rising"] & ev_strict["macd_above_signal"] & (ev_strict["volume_ratio20"] >= 1.0),
    }
    for name, mask in combined_masks.items():
        groups.append(summarize_group(ev_strict[mask], "combined", name))

    flat = []
    ranked = []
    for g in groups:
        row = {"feature": g["feature"], "group": g["group"], "event_count": g["event_count"]}
        for h in HORIZONS:
            for k, v in g[f"h{h}"].items():
                row[f"{h}d_{k}"] = v
        flat.append(row)
        score, meta = candidate_score(g, baseline)
        ranked.append({
            "feature": g["feature"],
            "group": g["group"],
            "event_count": g["event_count"],
            "score": score,
            **meta,
        })
    pd.DataFrame(flat).to_csv(OUT / "subgroups.csv", index=False, encoding="utf-8-sig")

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    v2_candidates = [
        x for x in ranked
        if x["n40"] >= MIN_GROUP_N
        and x["positive_return_horizons"] >= 3
        and x["positive_winrate_horizons"] >= 3
    ]

    summary = {
        "status": "PASS",
        "study_type": "75MA cross event study for v2 research; NOT exact v1.2 trading P/L",
        "protocol_v1_2_unchanged": True,
        "protocol_sha256": PROTOCOL_SHA,
        "source": "Yahoo Finance via yfinance only; no silent fallback",
        "latest_common_closed_date": common_latest.date().isoformat(),
        "event_window_start": event_cutoff.date().isoformat(),
        "requested_universe_count": len(codes),
        "strict_3yplus_universe_count": len(strict_codes),
        "short_coverage_names": short_coverage,
        "strict_event_count": int(len(ev_strict)),
        "all_available_event_count": int(len(ev_all)),
        "survivorship_bias_warning": "Current monitored names are applied historically, so constituent-selection/survivorship bias remains. This is v2 idea research, not clean historical TOPIX100 OOS proof.",
        "coverage_warning": "Names with under 3 years of history are excluded from the strict ranking but retained in all-available output. No predecessor/ticker substitution is used.",
        "definitions": {
            "event": "Adjusted Close crosses from <= SMA75 on prior session to > SMA75 on event session",
            "forward_return": "Adjusted close return from event close to h-th later trading-day adjusted close",
            "max_up": "Maximum adjusted intraday high return over the next h trading days versus event adjusted close",
            "max_down": "Minimum adjusted intraday low return over the next h trading days versus event adjusted close",
            "sma75_rising": "SMA75(event) > SMA75(5 trading days earlier)",
            "macd_above_signal": "MACD(12,26) > Signal(9), adjust=False, at event close",
            "price_above_sma25": "Adjusted close > SMA25 at event close",
            "volume_ratio20": "event volume / mean volume of prior 20 trading days",
        },
        "baseline_strict_3yplus": baseline,
        "baseline_all_available": baseline_all_available,
        "v2_candidate_screen_rule": f"strict >=3y names only; 40-day n>={MIN_GROUP_N}; mean-return improvement >0 on >=3/4 horizons and win-rate improvement >0 on >=3/4 horizons",
        "v2_candidates_ranked": v2_candidates[:10],
        "all_groups_ranked": ranked,
        "outputs": [
            "research/events_all_available.csv",
            "research/events_strict_3yplus.csv",
            "research/subgroups.csv",
            "research/coverage.csv",
            "research/summary.json"
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "strict_3yplus_universe_count": len(strict_codes),
        "short_coverage_names": short_coverage,
        "strict_event_count": int(len(ev_strict)),
        "all_available_event_count": int(len(ev_all)),
        "v2_candidate_count": len(v2_candidates),
        "top_v2_candidates": v2_candidates[:5],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
