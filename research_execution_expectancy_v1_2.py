#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL_SHA = "c2aef8f85dfdf125285623b7b6556b78a43509124ee09985b047b365f929d3a6"
OUT = Path("research")
SRC = OUT / "expectancy_foundation_enriched.csv"
DISCOVERY_END = pd.Timestamp("2024-07-08")
VALIDATION_START = pd.Timestamp("2024-09-09")
HORIZONS = [5, 10, 20, 40]

STRATEGIES = {
    "S1_V2_FOUNDATION": "v2 foundation only",
    "S2_V2_PLUS_SMA75_SLOPE_GE_0_75PCT": "v2 foundation + 75MA 5-session rise >= 0.75%",
    "S3_V2_PLUS_RS20_GE_3PCT": "v2 foundation + 20-session relative strength vs TOPIX >= +3%",
    "S4_V2_PLUS_BOTH": "v2 foundation + both filters",
}

# Frozen v1.2 blockers. These are not new rules; they are the reasons exact realized
# P/L cannot be manufactured from daily bars without changing the protocol.
FORMAL_BLOCKERS = [
    {
        "code": "UP_COUNT_UNRESOLVED_CAN_BLOCK_POSITION_SIZE",
        "effect": "100-share quantity under the 10-20% / 40-50% / 30-40% allocation bands cannot be fixed when wave/up-count is not unique; portfolio capital usage and annual return are therefore not deterministic.",
    },
    {
        "code": "SELL_INTEGRATION_NOT_FULLY_MECHANICAL",
        "effect": "MACD/BB/candlestick/high-retest signals are contextual and may conflict; frozen v1.2 requires SELL_DECISION=UNRESOLVED rather than inventing a universal exit, so realized holding-period P/L cannot be completed for all trades.",
    },
    {
        "code": "BUY_GAP_FILL_UNKNOWN",
        "effect": "If next-day open is above a buy stop, trigger can be known but exact fill is UNKNOWN from daily OHLC; exact P/L must be excluded.",
    },
    {
        "code": "SELL_STOP_GAP_FILL_UNKNOWN",
        "effect": "If open gaps below an active sell stop, trigger can be known but exact fill is UNKNOWN from daily OHLC; exact P/L must be excluded.",
    },
    {
        "code": "INTRADAY_SEQUENCE_UNKNOWN",
        "effect": "If multiple relevant levels are touched in one daily bar, event order is not recoverable from daily OHLC and must remain UNKNOWN.",
    },
]


def clean_float(x):
    if x is None or pd.isna(x) or np.isinf(x):
        return None
    return float(x)


def metric_block(df: pd.DataFrame, h: int) -> dict:
    rc = f"ret_{h}d"
    uc = f"max_up_{h}d"
    dc = f"max_down_{h}d"
    s = pd.to_numeric(df[rc], errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    pos = s[s > 0]
    neg = s[s < 0]
    avg_gain = clean_float(pos.mean()) if len(pos) else None
    avg_loss = clean_float(neg.mean()) if len(neg) else None
    payoff = None
    if avg_gain is not None and avg_loss not in (None, 0):
        payoff = avg_gain / abs(avg_loss)
    return {
        "n": int(len(s)),
        "win_rate": clean_float((s > 0).mean()),
        "mean_return": clean_float(s.mean()),
        "median_return": clean_float(s.median()),
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "payoff_ratio": clean_float(payoff),
        "mean_max_up": clean_float(pd.to_numeric(df.loc[s.index, uc], errors="coerce").mean()),
        "mean_max_down": clean_float(pd.to_numeric(df.loc[s.index, dc], errors="coerce").mean()),
    }


def period_metrics(df: pd.DataFrame) -> dict:
    return {f"h{h}": metric_block(df, h) for h in HORIZONS}


def strategy_mask(df: pd.DataFrame, key: str) -> pd.Series:
    base = pd.Series(True, index=df.index)
    if key == "S1_V2_FOUNDATION":
        return base
    if key == "S2_V2_PLUS_SMA75_SLOPE_GE_0_75PCT":
        return base & (pd.to_numeric(df["sma75_slope5_pct"], errors="coerce") >= 0.0075)
    if key == "S3_V2_PLUS_RS20_GE_3PCT":
        return base & (pd.to_numeric(df["rs20_vs_topix"], errors="coerce") >= 0.03)
    if key == "S4_V2_PLUS_BOTH":
        return (
            base
            & (pd.to_numeric(df["sma75_slope5_pct"], errors="coerce") >= 0.0075)
            & (pd.to_numeric(df["rs20_vs_topix"], errors="coerce") >= 0.03)
        )
    raise KeyError(key)


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    df = pd.read_csv(SRC)
    df["date"] = pd.to_datetime(df["date"])

    required = {
        "date", "code", "sma75_slope5_pct", "rs20_vs_topix",
        *[f"ret_{h}d" for h in HORIZONS],
        *[f"max_up_{h}d" for h in HORIZONS],
        *[f"max_down_{h}d" for h in HORIZONS],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    periods = {
        "all": df,
        "discovery": df[df["date"] <= DISCOVERY_END],
        "validation": df[df["date"] >= VALIDATION_START],
    }

    result = {}
    rows = []
    for key, desc in STRATEGIES.items():
        mask = strategy_mask(df, key)
        sdf = df[mask].copy()
        blocks = {}
        for pname, pdf in periods.items():
            psub = sdf.loc[sdf.index.intersection(pdf.index)]
            blocks[pname] = {
                "candidate_event_count": int(len(psub)),
                "event_study_proxy": period_metrics(psub),
            }
            for h in HORIZONS:
                m = blocks[pname]["event_study_proxy"][f"h{h}"]
                rows.append({
                    "strategy": key,
                    "description": desc,
                    "period": pname,
                    "horizon_days": h,
                    "candidate_event_count": int(len(psub)),
                    **m,
                })
        result[key] = {
            "description": desc,
            "periods": blocks,
            "formal_realized_trade_metrics": {
                "status": "UNSCORABLE_UNDER_FROZEN_V1_2_WITH_DAILY_OHLC",
                "win_rate": None,
                "mean_pnl_per_trade": None,
                "avg_gain": None,
                "avg_loss": None,
                "payoff_ratio": None,
                "max_drawdown": None,
                "annual_return": None,
                "capital_efficiency": None,
                "official_trade_count": None,
            },
        }

    # Proxy leaders are kept explicitly separate from formal realized-trade conclusions.
    proxy_leaders = {}
    for h in [20, 40]:
        candidates = []
        for key in STRATEGIES:
            m = result[key]["periods"]["validation"]["event_study_proxy"][f"h{h}"]
            if m.get("n", 0) > 0 and m.get("mean_return") is not None:
                candidates.append((m["mean_return"], m["n"], key))
        candidates.sort(reverse=True)
        if candidates:
            proxy_leaders[f"validation_{h}d_mean_return"] = {
                "strategy": candidates[0][2],
                "mean_return": candidates[0][0],
                "n": candidates[0][1],
                "warning": "Event-study proxy only; not realized v1.2 trade P/L.",
            }

    summary = {
        "status": "PASS_AUDIT_FORMAL_REALIZED_EXPECTANCY_UNSCORABLE",
        "protocol_v1_2_unchanged": True,
        "protocol_sha256": PROTOCOL_SHA,
        "candidate_selection_assumption": "The previously frozen v2 foundation is used only to select candidate events; this does not resolve frozen v1.2 exit integration, fill ambiguity, or wave-based allocation ambiguity.",
        "strategies": result,
        "formal_comparison_verdict": {
            "highest_expected_value": "UNRESOLVED",
            "most_stable": "UNRESOLVED",
            "lowest_risk": "UNRESOLVED",
            "reason": "Exact realized trade paths and/or capital sizing are not deterministic under frozen v1.2 for the full sample. Ranking them numerically would require adding rules that v1.2 explicitly forbids inventing.",
        },
        "event_study_proxy_leaders": proxy_leaders,
        "formal_blockers": FORMAL_BLOCKERS,
        "what_is_still_valid": "The existing 5/10/20/40-day event-study return comparisons remain valid as observational research, but they are not the actual entry-stop-exit-capital realized expectancy requested here.",
        "next_research_path_without_changing_v1_2": [
            "Collect forward/live v1.2 decisions with explicit AI sell decisions and actual/observable fills, then score only deterministic completed trades.",
            "Or create a separately versioned v2 execution specification that predefines candidate persistence, wave-count/sizing, and a universal deterministic exit hierarchy before viewing its future results; do not relabel that as v1.2.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "execution_expectancy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(OUT / "execution_expectancy_event_proxy.csv", index=False, encoding="utf-8-sig")

    md = []
    md.append("# v1.2 実売買期待値バックテスト監査\n")
    md.append("- v1.2変更: **なし**")
    md.append(f"- 固定SHA256: `{PROTOCOL_SHA}`")
    md.append("- 結論: **正式な実現売買期待値は、固定v1.2のままでは日足OHLCだけから一意に算出できない。**")
    md.append("- 5/10/20/40日リターンはイベント研究の参考値としてのみ保存。実売買P/Lとは混ぜない。\n")
    md.append("## 理由")
    for b in FORMAL_BLOCKERS:
        md.append(f"- `{b['code']}`: {b['effect']}")
    md.append("\n## 4戦略の参考（validation、イベント研究）")
    md.append("| strategy | candidates | 20d mean | 20d win | 40d mean | 40d win |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for key in STRATEGIES:
        p = result[key]["periods"]["validation"]
        h20 = p["event_study_proxy"]["h20"]
        h40 = p["event_study_proxy"]["h40"]
        def pct(v):
            return "NA" if v is None else f"{100*v:.2f}%"
        md.append(
            f"| {key} | {p['candidate_event_count']} | {pct(h20.get('mean_return'))} | {pct(h20.get('win_rate'))} | {pct(h40.get('mean_return'))} | {pct(h40.get('win_rate'))} |"
        )
    md.append("\n> この表は実際のv1.2約定・損切り・売却・30万円ポートフォリオ成績ではない。")
    (OUT / "execution_expectancy_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "formal_verdict": summary["formal_comparison_verdict"],
        "proxy_leaders": proxy_leaders,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
