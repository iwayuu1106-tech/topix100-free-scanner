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
OUT = Path("research")
EVENTS_PATH = OUT / "events_strict_3yplus.csv"

# Keep the same temporal split used by the prior research holdout.
DISCOVERY_END = pd.Timestamp("2024-07-08")
VALIDATION_START = pd.Timestamp("2024-09-09")

# Pre-declared decision thresholds for this run. These are NOT changes to v1.2.
MIN_TOTAL_EVENTS = 200
MIN_VALIDATION_N40 = 80
MIN_YEAR_N = 20
MIN_REGIME_N = 30
MAX_TOP1_EVENT_SHARE = 0.10
MAX_TOP10_EVENT_SHARE = 0.50

HYPOTHESIS_LABEL = (
    "75MA rising + SMA25>SMA75 + event adjusted close<=SMA25 + adjusted close crosses above SMA75"
)


def stop(reason: str, details=None):
    obj = {
        "status": "STOP",
        "reason": reason,
        "details": details or {},
        "protocol_v1_2_unchanged": True,
        "protocol_sha256": PROTOCOL_SHA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "formal_hypothesis_summary.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    raise SystemExit(2)


def metric_block(df: pd.DataFrame, h: int) -> dict:
    ret_col = f"ret_{h}d"
    up_col = f"max_up_{h}d"
    down_col = f"max_down_{h}d"
    s = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    pos = s[s > 0]
    neg = s[s < 0]
    mup = pd.to_numeric(df.loc[s.index, up_col], errors="coerce").dropna()
    mdn = pd.to_numeric(df.loc[s.index, down_col], errors="coerce").dropna()
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


def deltas(group: dict, baseline: dict) -> dict:
    out = {}
    for h in HORIZONS:
        g = group[f"h{h}"]
        b = baseline[f"h{h}"]
        if g.get("n", 0) and b.get("n", 0):
            out[f"{h}d_mean_return_delta"] = float(g["mean_return"] - b["mean_return"])
            out[f"{h}d_win_rate_delta"] = float(g["win_rate"] - b["win_rate"])
        else:
            out[f"{h}d_mean_return_delta"] = None
            out[f"{h}d_win_rate_delta"] = None
    return out


def get_single_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy()
    if ticker in raw.columns.get_level_values(0):
        return raw[ticker].copy()
    if ticker in raw.columns.get_level_values(1):
        return raw.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def load_market_regime(last_event_date: pd.Timestamp) -> pd.DataFrame:
    # 1306.T is used only as a free TOPIX-market proxy for regime classification.
    # It is not used to change v1.2 or to generate entry events.
    ticker = "1306.T"
    try:
        raw = yf.download(
            ticker,
            period="6y",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            timeout=60,
        )
    except Exception as e:
        stop("TOPIX_PROXY_FETCH_EXCEPTION", {"ticker": ticker, "error": repr(e)})
    g = get_single_frame(raw, ticker)
    if g.empty or "Adj Close" not in g.columns:
        stop("TOPIX_PROXY_EMPTY_OR_SCHEMA", {"ticker": ticker})
    g = g[["Adj Close"]].dropna().copy()
    g.index = pd.to_datetime(g.index).tz_localize(None)
    g = g[~g.index.duplicated(keep="last")].sort_index()

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    if now_jst.hour < 16 and len(g) and g.index.max().date() == now_jst.date():
        g = g[g.index.date < now_jst.date()]
    g = g[g.index <= last_event_date]
    if len(g) < 220:
        stop("TOPIX_PROXY_INSUFFICIENT_HISTORY", {"ticker": ticker, "rows": int(len(g))})

    g["market_sma200"] = pd.to_numeric(g["Adj Close"], errors="coerce").rolling(200).mean()
    g["market_regime"] = np.where(
        g["Adj Close"] >= g["market_sma200"], "BULL_ABOVE_SMA200", "BEAR_BELOW_SMA200"
    )
    g = g.dropna(subset=["market_sma200"])
    out = g.reset_index().rename(columns={g.index.name or "index": "market_date", "Adj Close": "market_adj_close"})
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "market_date"})
    out["market_date"] = pd.to_datetime(out["market_date"])
    return out[["market_date", "market_adj_close", "market_sma200", "market_regime"]]


def flatten_group_row(label_cols: dict, metrics: dict) -> dict:
    row = dict(label_cols)
    for h in HORIZONS:
        for k, v in metrics[f"h{h}"].items():
            row[f"{h}d_{k}"] = v
    return row


def main():
    if not EVENTS_PATH.exists():
        stop("BASE_RESEARCH_EVENTS_MISSING", {"path": str(EVENTS_PATH)})

    ev = pd.read_csv(EVENTS_PATH)
    required = {
        "code", "date", "sma75_rising", "price_above_sma25", "sma25_above_sma75",
        *[f"ret_{h}d" for h in HORIZONS],
        *[f"max_up_{h}d" for h in HORIZONS],
        *[f"max_down_{h}d" for h in HORIZONS],
    }
    missing = sorted(required - set(ev.columns))
    if missing:
        stop("BASE_RESEARCH_SCHEMA_MISSING", {"missing": missing})

    ev["date"] = pd.to_datetime(ev["date"])
    for col in ["sma75_rising", "price_above_sma25", "sma25_above_sma75"]:
        if ev[col].dtype != bool:
            ev[col] = ev[col].astype(str).str.lower().map({"true": True, "false": False})
    if ev[["sma75_rising", "price_above_sma25", "sma25_above_sma75"]].isna().any().any():
        stop("BOOLEAN_PARSE_FAILURE")

    baseline = ev.copy()
    mask = (
        ev["sma75_rising"]
        & ev["sma25_above_sma75"]
        & (~ev["price_above_sma25"])
    )
    hyp = ev[mask].copy().reset_index(drop=True)
    if hyp.empty:
        stop("NO_HYPOTHESIS_EVENTS")

    # Regime classification uses the last available TOPIX proxy observation at or before each event date.
    market = load_market_regime(ev["date"].max())
    hyp_reg = pd.merge_asof(
        hyp.sort_values("date"), market.sort_values("market_date"),
        left_on="date", right_on="market_date", direction="backward",
        tolerance=pd.Timedelta("7D")
    )
    if hyp_reg["market_regime"].isna().any():
        bad = hyp_reg.loc[hyp_reg["market_regime"].isna(), ["code", "date"]].head(20).to_dict("records")
        stop("MARKET_REGIME_JOIN_GAP", {"examples": bad})
    hyp = hyp_reg.drop(columns=["market_date"]).copy()

    # Save exact matched events.
    hyp.to_csv(OUT / "formal_hypothesis_events.csv", index=False, encoding="utf-8-sig")

    overall_base = all_metrics(baseline)
    overall_hyp = all_metrics(hyp)
    overall_delta = deltas(overall_hyp, overall_base)

    # Retrospective temporal split. Important: this is a robustness check, not pristine OOS,
    # because this combined hypothesis was formed after earlier component research had already inspected history.
    discovery_base = baseline[baseline["date"] <= DISCOVERY_END].copy()
    discovery_hyp = hyp[hyp["date"] <= DISCOVERY_END].copy()
    validation_base = baseline[baseline["date"] >= VALIDATION_START].copy()
    validation_hyp = hyp[hyp["date"] >= VALIDATION_START].copy()

    disc_base_m = all_metrics(discovery_base)
    disc_hyp_m = all_metrics(discovery_hyp)
    val_base_m = all_metrics(validation_base)
    val_hyp_m = all_metrics(validation_hyp)

    # Year stability, with same-year baseline comparison.
    year_rows = []
    years = sorted(hyp["date"].dt.year.unique())
    for y in years:
        hy = hyp[hyp["date"].dt.year == y]
        by = baseline[baseline["date"].dt.year == y]
        hm, bm = all_metrics(hy), all_metrics(by)
        row = flatten_group_row({"year": int(y), "event_count": int(len(hy))}, hm)
        dd = deltas(hm, bm)
        row.update({f"delta_{k}": v for k, v in dd.items()})
        row["year_sample_adequate"] = bool(hm["h40"].get("n", 0) >= MIN_YEAR_N)
        year_rows.append(row)
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(OUT / "formal_hypothesis_yearly.csv", index=False, encoding="utf-8-sig")

    # Ticker concentration/stability.
    ticker_rows = []
    for code, g in hyp.groupby("code", sort=True):
        m = all_metrics(g)
        row = flatten_group_row({"code": str(code), "event_count": int(len(g))}, m)
        ticker_rows.append(row)
    ticker_df = pd.DataFrame(ticker_rows).sort_values("event_count", ascending=False)
    ticker_df.to_csv(OUT / "formal_hypothesis_tickers.csv", index=False, encoding="utf-8-sig")

    total_events = len(hyp)
    top1_share = float(ticker_df.iloc[0]["event_count"] / total_events) if total_events else 1.0
    top10_share = float(ticker_df.head(10)["event_count"].sum() / total_events) if total_events else 1.0

    # Market-regime stability.
    regime_rows = []
    for regime, g in hyp.groupby("market_regime", sort=True):
        m = all_metrics(g)
        row = flatten_group_row({"market_regime": regime, "event_count": int(len(g))}, m)
        row["regime_sample_adequate"] = bool(m["h40"].get("n", 0) >= MIN_REGIME_N)
        regime_rows.append(row)
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / "formal_hypothesis_regimes.csv", index=False, encoding="utf-8-sig")

    eligible_years = year_df[year_df["40d_n"] >= MIN_YEAR_N].copy() if not year_df.empty else year_df
    positive_year_fraction_20 = (
        float((eligible_years["20d_mean_return"] > 0).mean()) if len(eligible_years) else None
    )
    positive_year_fraction_40 = (
        float((eligible_years["40d_mean_return"] > 0).mean()) if len(eligible_years) else None
    )

    adequate_regimes = regime_df[regime_df["40d_n"] >= MIN_REGIME_N].copy() if not regime_df.empty else regime_df
    regimes_positive_20 = bool((adequate_regimes["20d_mean_return"] > 0).all()) if len(adequate_regimes) else False
    regimes_positive_40 = bool((adequate_regimes["40d_mean_return"] > 0).all()) if len(adequate_regimes) else False

    val_n40 = int(val_hyp_m["h40"].get("n", 0))
    val_delta = deltas(val_hyp_m, val_base_m)
    core_validation_pass = all([
        val_n40 >= MIN_VALIDATION_N40,
        (val_delta.get("20d_mean_return_delta") or 0) > 0,
        (val_delta.get("20d_win_rate_delta") or 0) > 0,
        (val_delta.get("40d_mean_return_delta") or 0) > 0,
        (val_delta.get("40d_win_rate_delta") or 0) > 0,
    ])
    sample_pass = total_events >= MIN_TOTAL_EVENTS
    concentration_pass = top1_share <= MAX_TOP1_EVENT_SHARE and top10_share <= MAX_TOP10_EVENT_SHARE
    year_pass = (
        positive_year_fraction_20 is not None
        and positive_year_fraction_40 is not None
        and positive_year_fraction_20 >= 0.70
        and positive_year_fraction_40 >= 0.70
    )
    regime_pass = regimes_positive_20 and regimes_positive_40

    if not core_validation_pass:
        verdict = "却下"
        verdict_reason = "時間分割の後半で20日・40日の平均リターンと勝率が基準を同時に上回る条件を満たさなかった。"
    elif sample_pass and concentration_pass and year_pass and regime_pass:
        verdict = "v2正式候補"
        verdict_reason = "時間分割・年別・銘柄集中・相場環境の事前基準をすべて通過した。"
    else:
        verdict = "保留"
        verdict_reason = "時間分割の主要成績は通過したが、年別・銘柄集中・相場環境・サンプル数のいずれかが事前基準未達。"

    summary = {
        "status": "PASS",
        "hypothesis": HYPOTHESIS_LABEL,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "protocol_v1_2_unchanged": True,
        "protocol_sha256": PROTOCOL_SHA,
        "study_note": "This validates a 75MA-cross event filter, not exact full v1.2 trading P/L with all entry/exit rules.",
        "survivorship_bias_warning": "Current monitored constituents are applied historically; this is not a clean historical TOPIX100 constituent backtest.",
        "holdout_integrity_warning": "The combined hypothesis was formed after earlier historical component research, so this temporal split is a retrospective robustness check, not pristine untouched OOS. Future pre-registered live observations are still required before adopting v2.",
        "market_regime_definition": "1306.T (TOPIX ETF proxy) adjusted close >= its SMA200 = BULL; below = BEAR. Last observation at or before event date, no future data.",
        "decision_thresholds": {
            "min_total_events": MIN_TOTAL_EVENTS,
            "min_validation_n40": MIN_VALIDATION_N40,
            "min_year_n40": MIN_YEAR_N,
            "min_regime_n40": MIN_REGIME_N,
            "max_top1_event_share": MAX_TOP1_EVENT_SHARE,
            "max_top10_event_share": MAX_TOP10_EVENT_SHARE,
            "formal_candidate_requires": "validation 20d/40d mean-return and win-rate deltas >0; total sample adequate; concentration limits pass; >=70% adequate years positive at 20d and 40d; all adequate market regimes positive at 20d and 40d",
        },
        "sample": {
            "baseline_event_count": int(len(baseline)),
            "hypothesis_event_count": int(len(hyp)),
            "first_date": hyp["date"].min().date().isoformat(),
            "last_date": hyp["date"].max().date().isoformat(),
            "discovery_hypothesis_count": int(len(discovery_hyp)),
            "validation_hypothesis_count": int(len(validation_hyp)),
            "validation_n40": val_n40,
        },
        "overall_baseline": overall_base,
        "overall_hypothesis": overall_hyp,
        "overall_deltas": overall_delta,
        "temporal_robustness": {
            "discovery_end": DISCOVERY_END.date().isoformat(),
            "validation_start": VALIDATION_START.date().isoformat(),
            "discovery_baseline": disc_base_m,
            "discovery_hypothesis": disc_hyp_m,
            "discovery_deltas": deltas(disc_hyp_m, disc_base_m),
            "validation_baseline": val_base_m,
            "validation_hypothesis": val_hyp_m,
            "validation_deltas": val_delta,
            "core_validation_pass": core_validation_pass,
        },
        "year_stability": {
            "adequate_year_count": int(len(eligible_years)),
            "positive_year_fraction_20d": positive_year_fraction_20,
            "positive_year_fraction_40d": positive_year_fraction_40,
            "pass": year_pass,
        },
        "ticker_concentration": {
            "unique_tickers": int(hyp["code"].nunique()),
            "top1_event_share": top1_share,
            "top10_event_share": top10_share,
            "pass": concentration_pass,
            "top10": ticker_df.head(10)[["code", "event_count", "20d_n", "20d_win_rate", "20d_mean_return", "40d_n", "40d_win_rate", "40d_mean_return"]].to_dict("records"),
        },
        "market_regime_stability": {
            "adequate_regime_count": int(len(adequate_regimes)),
            "all_adequate_regimes_positive_20d": regimes_positive_20,
            "all_adequate_regimes_positive_40d": regimes_positive_40,
            "pass": regime_pass,
            "rows": regime_df.to_dict("records"),
        },
        "criteria_pass": {
            "sample": sample_pass,
            "temporal_validation": core_validation_pass,
            "year_stability": year_pass,
            "ticker_concentration": concentration_pass,
            "market_regime": regime_pass,
        },
        "next_step": "Do not modify v1.2. If verdict is v2正式候補, freeze this filter as a v2 candidate and pre-register a forward/live comparison against v1.2 before adoption.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (OUT / "formal_hypothesis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
