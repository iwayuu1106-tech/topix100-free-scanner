#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

PROTOCOL_SHA = "c2aef8f85dfdf125285623b7b6556b78a43509124ee09985b047b365f929d3a6"
OUT = Path("research")
EVENTS_PATH = OUT / "events_strict_3yplus.csv"
HORIZONS = [5, 10, 20, 40]
DISCOVERY_END = pd.Timestamp("2024-07-08")
VALIDATION_START = pd.Timestamp("2024-09-09")
MIN_SINGLE_N40 = 80
MIN_PAIR_N40 = 60

# IMPORTANT: These bins/thresholds are declared in code before this research run.
# Do not optimize a continuous cut-point after seeing returns.
PREDECLARED_BINS = {
    "sma75_slope5_pct": [
        ("SLOPE_0_TO_0_25PCT", 0.0, 0.0025),
        ("SLOPE_0_25_TO_0_75PCT", 0.0025, 0.0075),
        ("SLOPE_GE_0_75PCT", 0.0075, math.inf),
    ],
    "pullback25_pct": [
        ("PULLBACK_0_TO_1PCT", 0.0, 0.01),
        ("PULLBACK_1_TO_3PCT", 0.01, 0.03),
        ("PULLBACK_GE_3PCT", 0.03, math.inf),
    ],
    "rs20_vs_topix": [
        ("RS20_LT_0", -math.inf, 0.0),
        ("RS20_0_TO_3PCT", 0.0, 0.03),
        ("RS20_GE_3PCT", 0.03, math.inf),
    ],
    "rs60_vs_topix": [
        ("RS60_LT_0", -math.inf, 0.0),
        ("RS60_0_TO_5PCT", 0.0, 0.05),
        ("RS60_GE_5PCT", 0.05, math.inf),
    ],
    "vol20_ann": [
        ("VOL20_LT_20PCT", 0.0, 0.20),
        ("VOL20_20_TO_35PCT", 0.20, 0.35),
        ("VOL20_GE_35PCT", 0.35, math.inf),
    ],
}

FUNDAMENTALS_STATUS = {
    "status": "NOT_EXECUTED_NO_POINT_IN_TIME_SOURCE",
    "requested_feature": "historical revenue growth + operating-income growth known as of each event date",
    "reason": (
        "The free runner currently has no verified point-in-time historical fundamentals feed with disclosure timestamps. "
        "Using today's Yahoo/yfinance financial-statement snapshot for past events could leak later revisions/future knowledge, "
        "so this factor is deliberately not scored."
    ),
    "requirement_to_enable": (
        "Add a historical fundamentals source that preserves report/disclosure dates and values as known at that time; "
        "then predeclare growth bins before rerunning."
    ),
}


def stop(reason: str, details=None):
    obj = {
        "status": "STOP",
        "reason": reason,
        "details": details or {},
        "protocol_v1_2_unchanged": True,
        "protocol_sha256": PROTOCOL_SHA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expectancy_filter_summary.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    raise SystemExit(2)


def extract_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy()
    l0 = raw.columns.get_level_values(0)
    l1 = raw.columns.get_level_values(1)
    if ticker in l0:
        return raw[ticker].copy()
    if ticker in l1:
        return raw.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


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


def all_metrics(df: pd.DataFrame) -> dict:
    return {f"h{h}": metric_block(df, h) for h in HORIZONS}


def compare_metrics(group_m: dict, base_m: dict) -> dict:
    out = {}
    for h in HORIZONS:
        g = group_m[f"h{h}"]
        b = base_m[f"h{h}"]
        if not g.get("n") or not b.get("n"):
            out[f"{h}d_mean_return_delta"] = None
            out[f"{h}d_win_rate_delta"] = None
            out[f"{h}d_mean_max_down_delta"] = None
            continue
        out[f"{h}d_mean_return_delta"] = float(g["mean_return"] - b["mean_return"])
        out[f"{h}d_win_rate_delta"] = float(g["win_rate"] - b["win_rate"])
        # Positive means drawdown improved (less negative).
        out[f"{h}d_mean_max_down_delta"] = float(g["mean_max_down"] - b["mean_max_down"])
    return out


def pass_rule(discovery_delta: dict, validation_delta: dict, discovery_n40: int, validation_n40: int, min_n40: int) -> tuple[bool, dict]:
    val_return_pos = sum((validation_delta.get(f"{h}d_mean_return_delta") or -999) > 0 for h in HORIZONS)
    val_win_pos = sum((validation_delta.get(f"{h}d_win_rate_delta") or -999) > 0 for h in HORIZONS)
    checks = {
        "discovery_n40_enough": discovery_n40 >= min_n40,
        "validation_n40_enough": validation_n40 >= min_n40,
        "discovery_20d_mean_return_improves": (discovery_delta.get("20d_mean_return_delta") or -999) > 0,
        "discovery_40d_mean_return_improves": (discovery_delta.get("40d_mean_return_delta") or -999) > 0,
        "validation_20d_mean_return_improves": (validation_delta.get("20d_mean_return_delta") or -999) > 0,
        "validation_20d_win_rate_improves": (validation_delta.get("20d_win_rate_delta") or -999) > 0,
        "validation_40d_mean_return_improves": (validation_delta.get("40d_mean_return_delta") or -999) > 0,
        "validation_40d_win_rate_improves": (validation_delta.get("40d_win_rate_delta") or -999) > 0,
        "validation_20d_drawdown_not_worse": (validation_delta.get("20d_mean_max_down_delta") or -999) >= 0,
        "validation_40d_drawdown_not_worse": (validation_delta.get("40d_mean_max_down_delta") or -999) >= 0,
        "validation_mean_return_improves_at_least_3_of_4": val_return_pos >= 3,
        "validation_win_rate_improves_at_least_3_of_4": val_win_pos >= 3,
    }
    return all(checks.values()), checks


def rank_score(delta: dict) -> float:
    # Research ranking only, not a trading rule. Emphasize 20d/40d return, then win-rate and drawdown.
    score = 0.0
    for h, weight in [(5, 0.25), (10, 0.5), (20, 1.0), (40, 1.0)]:
        dr = delta.get(f"{h}d_mean_return_delta") or 0.0
        dw = delta.get(f"{h}d_win_rate_delta") or 0.0
        dd = delta.get(f"{h}d_mean_max_down_delta") or 0.0
        score += weight * dr + 0.10 * weight * dw + 0.50 * weight * dd
    return float(score)


def interval_mask(s: pd.Series, low: float, high: float) -> pd.Series:
    if math.isinf(low) and low < 0:
        return s < high
    if math.isinf(high):
        return s >= low
    # Left-inclusive, right-exclusive except lower bound is exact and deterministic.
    return (s >= low) & (s < high)


def enrich_trailing_features(foundation: pd.DataFrame) -> pd.DataFrame:
    codes = sorted(foundation["code"].astype(str).unique())
    tickers = [f"{c}.T" for c in codes]
    market_ticker = "1306.T"
    all_tickers = tickers + [market_ticker]
    try:
        raw = yf.download(
            tickers=all_tickers,
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
        stop("TRAILING_FEATURE_DOWNLOAD_EXCEPTION", {"error": repr(e)})
    if raw is None or raw.empty:
        stop("TRAILING_FEATURE_DOWNLOAD_EMPTY")

    mg = extract_frame(raw, market_ticker)
    if mg.empty or "Adj Close" not in mg.columns:
        stop("MARKET_PROXY_MISSING", {"ticker": market_ticker})
    mg = mg[["Adj Close"]].dropna().copy()
    mg.index = pd.to_datetime(mg.index).tz_localize(None)
    mg = mg[~mg.index.duplicated(keep="last")].sort_index()
    market_adj = pd.to_numeric(mg["Adj Close"], errors="coerce")
    market_ret20 = market_adj.pct_change(20)
    market_ret60 = market_adj.pct_change(60)

    rows = []
    hard_missing = []
    for code, ticker in zip(codes, tickers):
        g = extract_frame(raw, ticker)
        if g.empty or "Adj Close" not in g.columns:
            hard_missing.append({"code": code, "problem": "no_adj_close"})
            continue
        g = g[["Adj Close"]].dropna().copy()
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        adj = pd.to_numeric(g["Adj Close"], errors="coerce")
        stock_ret20 = adj.pct_change(20)
        stock_ret60 = adj.pct_change(60)
        logret = np.log(adj / adj.shift(1))
        vol20 = logret.rolling(20).std(ddof=0) * np.sqrt(252.0)
        feat = pd.DataFrame({
            "date": g.index,
            "rs20_vs_topix": stock_ret20 - market_ret20.reindex(g.index),
            "rs60_vs_topix": stock_ret60 - market_ret60.reindex(g.index),
            "vol20_ann": vol20,
        })
        feat["code"] = code
        rows.append(feat.reset_index(drop=True))

    if hard_missing:
        stop("TRAILING_FEATURE_TICKER_GAP", {"count": len(hard_missing), "items": hard_missing})
    feat_all = pd.concat(rows, ignore_index=True)
    out = foundation.merge(feat_all, on=["code", "date"], how="left", validate="many_to_one")
    miss = out[["rs20_vs_topix", "rs60_vs_topix", "vol20_ann"]].isna().any(axis=1)
    if miss.any():
        ex = out.loc[miss, ["code", "date"]].head(20).astype(str).to_dict("records")
        stop("TRAILING_FEATURE_EVENT_GAP", {"count": int(miss.sum()), "examples": ex})
    return out


def flatten_row(prefix: dict, discovery_m: dict, validation_m: dict, discovery_delta: dict, validation_delta: dict, passed: bool, checks: dict, score: float) -> dict:
    row = dict(prefix)
    row["passed"] = bool(passed)
    row["validation_rank_score"] = score
    row.update({f"check_{k}": v for k, v in checks.items()})
    for phase, mm in [("discovery", discovery_m), ("validation", validation_m)]:
        for h in HORIZONS:
            for k, v in mm[f"h{h}"].items():
                row[f"{phase}_{h}d_{k}"] = v
    for k, v in discovery_delta.items():
        row[f"discovery_delta_{k}"] = v
    for k, v in validation_delta.items():
        row[f"validation_delta_{k}"] = v
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not EVENTS_PATH.exists():
        stop("EVENTS_FILE_MISSING", {"path": str(EVENTS_PATH)})

    ev = pd.read_csv(EVENTS_PATH)
    required = {
        "code", "date", "event_adj_close", "sma25", "sma75", "sma75_5",
        "sma75_rising", "sma25_above_sma75", "price_above_sma25",
        *[f"ret_{h}d" for h in HORIZONS],
        *[f"max_up_{h}d" for h in HORIZONS],
        *[f"max_down_{h}d" for h in HORIZONS],
    }
    missing = sorted(required - set(ev.columns))
    if missing:
        stop("EVENT_SCHEMA_MISSING", {"missing": missing})

    ev["code"] = ev["code"].astype(str).str.zfill(4)
    ev["date"] = pd.to_datetime(ev["date"])
    for col in ["sma75_rising", "sma25_above_sma75", "price_above_sma25"]:
        if ev[col].dtype != bool:
            ev[col] = ev[col].astype(str).str.lower().map({"true": True, "false": False})
    if ev[["sma75_rising", "sma25_above_sma75", "price_above_sma25"]].isna().any().any():
        stop("BOOLEAN_PARSE_FAILURE")

    # Frozen v2 foundation. The source file already contains only 75MA cross-up events.
    foundation = ev[
        ev["sma75_rising"]
        & ev["sma25_above_sma75"]
        & (~ev["price_above_sma25"])
    ].copy().reset_index(drop=True)
    if foundation.empty:
        stop("NO_V2_FOUNDATION_EVENTS")

    foundation["sma75_slope5_pct"] = pd.to_numeric(foundation["sma75"], errors="coerce") / pd.to_numeric(foundation["sma75_5"], errors="coerce") - 1.0
    foundation["pullback25_pct"] = (
        pd.to_numeric(foundation["sma25"], errors="coerce") - pd.to_numeric(foundation["event_adj_close"], errors="coerce")
    ) / pd.to_numeric(foundation["sma25"], errors="coerce")
    if foundation[["sma75_slope5_pct", "pullback25_pct"]].isna().any().any():
        stop("FOUNDATION_DERIVED_FEATURE_NA")

    foundation = enrich_trailing_features(foundation)
    foundation.to_csv(OUT / "expectancy_foundation_enriched.csv", index=False, encoding="utf-8-sig")

    discovery_base = foundation[foundation["date"] <= DISCOVERY_END].copy()
    validation_base = foundation[foundation["date"] >= VALIDATION_START].copy()
    if len(discovery_base) == 0 or len(validation_base) == 0:
        stop("TEMPORAL_SPLIT_EMPTY")
    discovery_base_m = all_metrics(discovery_base)
    validation_base_m = all_metrics(validation_base)

    single_rows = []
    single_objects = []
    passing_conditions = []
    for feature, bins in PREDECLARED_BINS.items():
        s = pd.to_numeric(foundation[feature], errors="coerce")
        if s.isna().any():
            stop("FEATURE_NA", {"feature": feature, "count": int(s.isna().sum())})
        for label, low, high in bins:
            mask = interval_mask(s, low, high)
            sub = foundation[mask].copy()
            dsub = sub[sub["date"] <= DISCOVERY_END].copy()
            vsub = sub[sub["date"] >= VALIDATION_START].copy()
            dm = all_metrics(dsub)
            vm = all_metrics(vsub)
            dd = compare_metrics(dm, discovery_base_m)
            vd = compare_metrics(vm, validation_base_m)
            passed, checks = pass_rule(dd, vd, dm["h40"].get("n", 0), vm["h40"].get("n", 0), MIN_SINGLE_N40)
            score = rank_score(vd)
            obj = {
                "feature": feature,
                "label": label,
                "low_inclusive": None if math.isinf(low) else low,
                "high_exclusive": None if math.isinf(high) else high,
                "event_count_total": int(len(sub)),
                "discovery_metrics": dm,
                "validation_metrics": vm,
                "discovery_delta_vs_v2_foundation": dd,
                "validation_delta_vs_v2_foundation": vd,
                "passed": passed,
                "checks": checks,
                "validation_rank_score": score,
            }
            single_objects.append(obj)
            single_rows.append(flatten_row(
                {"feature": feature, "label": label, "event_count_total": int(len(sub)), "low": low, "high": high},
                dm, vm, dd, vd, passed, checks, score
            ))
            if passed:
                passing_conditions.append({"feature": feature, "label": label, "low": low, "high": high})

    single_df = pd.DataFrame(single_rows).sort_values(["passed", "validation_rank_score"], ascending=[False, False])
    single_df.to_csv(OUT / "expectancy_single_filters.csv", index=False, encoding="utf-8-sig")

    # Pair only conditions that passed individually, and never combine two bins from the same feature.
    pair_rows = []
    pair_objects = []
    for a, b in itertools.combinations(passing_conditions, 2):
        if a["feature"] == b["feature"]:
            continue
        ma = interval_mask(pd.to_numeric(foundation[a["feature"]], errors="coerce"), a["low"], a["high"])
        mb = interval_mask(pd.to_numeric(foundation[b["feature"]], errors="coerce"), b["low"], b["high"])
        sub = foundation[ma & mb].copy()
        dsub = sub[sub["date"] <= DISCOVERY_END].copy()
        vsub = sub[sub["date"] >= VALIDATION_START].copy()
        dm = all_metrics(dsub)
        vm = all_metrics(vsub)
        dd = compare_metrics(dm, discovery_base_m)
        vd = compare_metrics(vm, validation_base_m)
        passed, checks = pass_rule(dd, vd, dm["h40"].get("n", 0), vm["h40"].get("n", 0), MIN_PAIR_N40)
        score = rank_score(vd)
        pair_label = f"{a['label']} + {b['label']}"
        obj = {
            "label": pair_label,
            "conditions": [a, b],
            "event_count_total": int(len(sub)),
            "discovery_metrics": dm,
            "validation_metrics": vm,
            "discovery_delta_vs_v2_foundation": dd,
            "validation_delta_vs_v2_foundation": vd,
            "passed": passed,
            "checks": checks,
            "validation_rank_score": score,
        }
        pair_objects.append(obj)
        pair_rows.append(flatten_row(
            {"label": pair_label, "event_count_total": int(len(sub)), "feature_a": a["feature"], "feature_b": b["feature"]},
            dm, vm, dd, vd, passed, checks, score
        ))

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["passed", "validation_rank_score"], ascending=[False, False])
    pair_df.to_csv(OUT / "expectancy_pair_filters.csv", index=False, encoding="utf-8-sig")

    passing_single = sorted([x for x in single_objects if x["passed"]], key=lambda x: x["validation_rank_score"], reverse=True)
    passing_pair = sorted([x for x in pair_objects if x["passed"]], key=lambda x: x["validation_rank_score"], reverse=True)

    rank_plan = {
        "C": {
            "definition": "Frozen v2 foundation only",
            "foundation": "75MA rising + SMA25>SMA75 + event close<=SMA25 + 75MA cross-up",
            "status": "BASELINE",
        },
        "B": {
            "definition": "v2 foundation + one individually validated expectancy filter",
            "eligible_filters": [x["label"] for x in passing_single],
            "status": "RESEARCH_CANDIDATE" if passing_single else "NO_FILTER_PASSED",
        },
        "A": {
            "definition": "v2 foundation + a maximum of two expectancy filters; both must pass singly and the pair must pass again",
            "eligible_pairs_ranked": [x["label"] for x in passing_pair],
            "top_pair": passing_pair[0]["label"] if passing_pair else None,
            "status": "RESEARCH_CANDIDATE" if passing_pair else "NO_PAIR_PASSED",
        },
        "adoption_note": "A/B/C is a candidate-ranking proposal only. It does not modify v1.2 execution, stops, exits, or capital rules and should be forward-tested before live weighting changes.",
    }

    summary = {
        "status": "PASS",
        "purpose": "Raise expected value without changing the frozen v2 foundation or v1.2 execution/risk-management rules.",
        "protocol_v1_2_unchanged": True,
        "protocol_sha256": PROTOCOL_SHA,
        "v2_foundation_unchanged": True,
        "v2_foundation": "75MA rising + SMA25>SMA75 + event adjusted close<=SMA25 + adjusted close crosses above SMA75",
        "study_window": {
            "first_event": foundation["date"].min().date().isoformat(),
            "last_event": foundation["date"].max().date().isoformat(),
            "foundation_event_count": int(len(foundation)),
            "discovery_end": DISCOVERY_END.date().isoformat(),
            "validation_start": VALIDATION_START.date().isoformat(),
            "discovery_count": int(len(discovery_base)),
            "validation_count": int(len(validation_base)),
        },
        "predeclared_bins": PREDECLARED_BINS,
        "pass_rule": {
            "single_min_n40_each_phase": MIN_SINGLE_N40,
            "pair_min_n40_each_phase": MIN_PAIR_N40,
            "requires": [
                "discovery 20d and 40d mean return > v2 foundation",
                "validation 20d and 40d mean return and win rate > v2 foundation",
                "validation 20d and 40d mean max-down no worse than v2 foundation",
                "validation mean-return improvement in >=3/4 horizons",
                "validation win-rate improvement in >=3/4 horizons",
            ],
        },
        "fundamentals": FUNDAMENTALS_STATUS,
        "single_filter_count": int(len(single_objects)),
        "single_filters_passed": int(len(passing_single)),
        "single_filters_ranked": passing_single,
        "pair_filter_count": int(len(pair_objects)),
        "pairs_passed": int(len(passing_pair)),
        "pairs_ranked": passing_pair,
        "rank_plan": rank_plan,
        "warnings": [
            "Current monitored constituents are applied historically, so constituent/survivorship bias remains.",
            "This is event-filter research, not exact full v1.2 realized trade P/L with all fills/exits.",
            "The v2 foundation and candidate filter families were motivated by prior historical research; validation here is a robustness check, not pristine untouched OOS.",
            "Do not add the fundamentals factor until a point-in-time historical disclosure source is connected.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (OUT / "expectancy_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "expectancy_fundamentals_status.json").write_text(
        json.dumps(FUNDAMENTALS_STATUS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "foundation_events": len(foundation),
        "single_filters_passed": len(passing_single),
        "pairs_passed": len(passing_pair),
        "top_single": passing_single[0]["label"] if passing_single else None,
        "top_pair": passing_pair[0]["label"] if passing_pair else None,
        "fundamentals": FUNDAMENTALS_STATUS["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
