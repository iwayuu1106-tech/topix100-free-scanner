#!/usr/bin/env python3
"""Corrected PRETEST 1.0 realized backtest runner.

IMPORTANT:
- v1.2 is unchanged.
- v2_execution_spec_pretest_1_0.json is unchanged.
- The 4 strategies, thresholds, sizing, entries, exits, same-day priority and
  winner rules are unchanged.
- This runner fixes only the data-unit implementation so the fixed A02 RAW-OHLC
  rule and A12 split handling are not applied twice.

Yahoo/yfinance historical OHLC is split-adjusted across history even with
`auto_adjust=False`. To execute in historical raw yen/share units, this runner
multiplies each OHLC row by the product of split ratios that become effective
strictly after that row. Stock Splits is retained unchanged, so A12 then changes
shares/stops once, on the effective date.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import research_v2_realized_backtest_pretest_1_0 as base

QUALITY_PATH = Path("v2_execution_data_quality_addendum_pretest_1_0.json")
QUALITY_SHA = "a1c8e2bbf515e7625474f9eb66757f64126f168ec721cb7ce6348e2742bc873e"
CORRECTION_PATH = Path("v2_execution_data_representation_correction_pretest_1_0.json")
CORRECTION_SHA = "53618707957dca6e48e418ee326aac62972f0e5d0fec6367a1c3a7b064bc636f"

ORIGINAL_OUT = Path("research")
CORRECTED_OUT = ORIGINAL_OUT / "corrected_pretest_1_0"
CORRECTED_OUT.mkdir(parents=True, exist_ok=True)
base.OUT = CORRECTED_OUT

OHLC_REPAIRS: list[dict] = []
SPLIT_EVENTS: list[dict] = []
RAW_RECON_AUDIT: list[dict] = []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_inputs():
    base.verify_spec()
    if not QUALITY_PATH.exists() or sha256_file(QUALITY_PATH) != QUALITY_SHA:
        base.stop("DATA_QUALITY_ADDENDUM_SHA_MISMATCH", {
            "actual": sha256_file(QUALITY_PATH) if QUALITY_PATH.exists() else None,
            "expected": QUALITY_SHA,
        })
    if not CORRECTION_PATH.exists() or sha256_file(CORRECTION_PATH) != CORRECTION_SHA:
        base.stop("DATA_REPRESENTATION_CORRECTION_SHA_MISMATCH", {
            "actual": sha256_file(CORRECTION_PATH) if CORRECTION_PATH.exists() else None,
            "expected": CORRECTION_SHA,
        })
    corr = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    if corr.get("parent_execution_spec_sha256") != base.EXPECTED_SPEC_SHA:
        base.stop("DATA_REPRESENTATION_PARENT_SPEC_MISMATCH")
    if corr.get("v1_2_sha256") != base.V1_2_SHA or corr.get("v1_2_modified") is not False:
        base.stop("DATA_REPRESENTATION_V1_2_MISMATCH")
    if corr.get("strategy_rules_modified") is not False:
        base.stop("DATA_REPRESENTATION_STRATEGY_CHANGE_FORBIDDEN")


def future_split_factor(splits: pd.Series) -> pd.Series:
    """Product of split ratios strictly AFTER each row."""
    sf = pd.to_numeric(splits, errors="coerce").fillna(0.0).astype(float)
    sf = sf.where(sf > 0.0, 1.0)
    inclusive = sf.iloc[::-1].cumprod().iloc[::-1]
    return inclusive / sf


def apply_ohlc_envelope_repair(code: str, g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for d in g.index:
        op = float(g.at[d, "Open"])
        hi = float(g.at[d, "High"])
        lo = float(g.at[d, "Low"])
        cl = float(g.at[d, "Close"])
        new_hi = max(op, hi, lo, cl)
        new_lo = min(op, hi, lo, cl)
        if new_hi != hi or new_lo != lo:
            OHLC_REPAIRS.append({
                "code": code,
                "date": d.date().isoformat(),
                "original_open": op,
                "original_high": hi,
                "original_low": lo,
                "original_close": cl,
                "repaired_high": new_hi,
                "repaired_low": new_lo,
            })
            g.at[d, "High"] = new_hi
            g.at[d, "Low"] = new_lo
    return g


def reconstruct_historical_raw_ohlc(code: str, g: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo split-adjusted OHLC to historical raw yen/share units."""
    g = g.copy()
    factor = future_split_factor(g["Stock Splits"])

    split_rows = g[pd.to_numeric(g["Stock Splits"], errors="coerce").fillna(0.0) > 0.0]
    for d, row in split_rows.iterrows():
        ratio = float(row["Stock Splits"])
        if abs(ratio - 1.0) <= 1e-12:
            continue
        idx = g.index.get_loc(d)
        prev_date = g.index[idx - 1] if isinstance(idx, (int, np.integer)) and idx > 0 else None
        prev_factor = float(factor.loc[prev_date]) if prev_date is not None else None
        day_factor = float(factor.loc[d])
        factor_jump = (prev_factor / day_factor) if (prev_factor is not None and day_factor != 0) else None
        ok = factor_jump is None or abs(factor_jump - ratio) <= max(1e-9, abs(ratio) * 1e-9)
        SPLIT_EVENTS.append({
            "code": code,
            "effective_date": d.date().isoformat(),
            "ratio": ratio,
            "previous_trading_date": prev_date.date().isoformat() if prev_date is not None else None,
            "future_factor_prev": prev_factor,
            "future_factor_on_effective_date": day_factor,
            "factor_jump": factor_jump,
            "factor_jump_matches_ratio": bool(ok),
        })
        if not ok:
            base.stop("RAW_RECON_SPLIT_FACTOR_INVARIANT_FAILED", SPLIT_EVENTS[-1])

    changed = factor.ne(1.0)
    max_factor = float(factor.max()) if len(factor) else 1.0
    min_factor = float(factor.min()) if len(factor) else 1.0

    for col in ["Open", "High", "Low", "Close"]:
        g[col] = pd.to_numeric(g[col], errors="coerce").astype(float) * factor

    RAW_RECON_AUDIT.append({
        "code": code,
        "rows": int(len(g)),
        "rows_reconstructed": int(changed.sum()),
        "max_future_split_factor": max_factor,
        "min_future_split_factor": min_factor,
        "split_event_count": int(sum(1 for x in SPLIT_EVENTS if x["code"] == code)),
    })
    return g


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
    frames: dict[str, pd.DataFrame] = {}
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

        # Existing pre-registered universal data-quality repair.
        g = apply_ohlc_envelope_repair(code, g)

        # Frozen data-representation correction: reconstruct historical RAW OHLC.
        g = reconstruct_historical_raw_ohlc(code, g)

        bad = g[
            (g["High"] < g[["Open", "Close", "Low"]].max(axis=1))
            | (g["Low"] > g[["Open", "Close", "High"]].min(axis=1))
            | (g["Volume"] < 0)
        ]
        if len(bad):
            problems.append({"code": code, "problem": "post_raw_reconstruction_ohlcv_sanity", "rows": int(len(bad))})
            continue

        g["ExecSMA75"] = base.split_only_sma75(g)
        frames[code] = g

    if problems:
        base.stop("PRICE_DATA_PROBLEM_AFTER_RAW_RECONSTRUCTION", {
            "count": len(problems), "items": problems[:100]
        })
    if not frames:
        base.stop("NO_PRICE_FRAMES_AFTER_RAW_RECONSTRUCTION")

    common_latest = min(g.index.max() for g in frames.values())
    frames = {c: g[g.index <= common_latest].copy() for c, g in frames.items()}
    return frames, common_latest, split_col_missing


def append_audit_and_publish():
    summary_path = CORRECTED_OUT / "v2_realized_summary_pretest_1_0.json"
    report_path = CORRECTED_OUT / "v2_realized_report_pretest_1_0.md"
    trades_path = CORRECTED_OUT / "v2_realized_trades_pretest_1_0.csv"
    daily_path = CORRECTED_OUT / "v2_realized_daily_equity_pretest_1_0.csv"
    if not summary_path.exists():
        base.stop("CORRECTED_SUMMARY_MISSING")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "PASS_REALIZED_BACKTEST_PRETEST_1_0_CORRECTED_DATA_UNITS"
    summary["data_source"] = (
        "Yahoo Finance via yfinance auto_adjust=False/actions=True; OHLC reconstructed "
        "to historical raw yen/share using future split factors before applying frozen A12"
    )
    summary["data_quality_addendum_file"] = str(QUALITY_PATH)
    summary["data_quality_addendum_sha256"] = QUALITY_SHA
    summary["data_representation_correction_file"] = str(CORRECTION_PATH)
    summary["data_representation_correction_sha256"] = CORRECTION_SHA
    summary["strategy_rules_modified_by_correction"] = False
    summary["invalid_prior_result_status"] = "INVALID_IMPLEMENTATION_DOUBLE_COUNTED_SPLITS"
    summary["ohlc_envelope_repair_count"] = len(OHLC_REPAIRS)
    summary["split_event_count"] = len(SPLIT_EVENTS)
    summary["raw_reconstructed_row_count"] = int(sum(x["rows_reconstructed"] for x in RAW_RECON_AUDIT))
    summary["raw_reconstruction_audit"] = RAW_RECON_AUDIT
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(OHLC_REPAIRS).to_csv(
        CORRECTED_OUT / "v2_realized_ohlc_repairs_pretest_1_0.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(SPLIT_EVENTS).to_csv(
        CORRECTED_OUT / "v2_realized_split_events_pretest_1_0.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(RAW_RECON_AUDIT).to_csv(
        CORRECTED_OUT / "v2_realized_raw_reconstruction_audit_pretest_1_0.csv", index=False, encoding="utf-8-sig"
    )

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n## データ単位修正監査\n\n")
        f.write(f"- 固定v2実行仕様SHA256: `{base.EXPECTED_SPEC_SHA}`（変更なし）\n")
        f.write(f"- データ品質アドオンSHA256: `{QUALITY_SHA}`\n")
        f.write(f"- データ表現修正SHA256: `{CORRECTION_SHA}`\n")
        f.write("- 売買ルール・4戦略・閾値・30万円・100株・勝者判定式の変更: **なし**\n")
        f.write(f"- OHLC包絡修復行数: **{len(OHLC_REPAIRS)}**\n")
        f.write(f"- 検出した非1株式分割/併合イベント: **{len(SPLIT_EVENTS)}**\n")
        f.write(f"- RAW株価単位へ復元した銘柄日行数: **{sum(x['rows_reconstructed'] for x in RAW_RECON_AUDIT)}**\n")
        f.write("- 旧バックテストは株式分割を二重計上したため **INVALID_IMPLEMENTATION**。本レポートの成績だけを採用する。\n")

    # Publish stable top-level corrected filenames requested by the project.
    mapping = {
        summary_path: ORIGINAL_OUT / "v2_realized_summary_pretest_1_0_corrected.json",
        report_path: ORIGINAL_OUT / "v2_realized_report_pretest_1_0_corrected.md",
        trades_path: ORIGINAL_OUT / "v2_realized_trades_pretest_1_0_corrected.csv",
        daily_path: ORIGINAL_OUT / "v2_realized_daily_equity_pretest_1_0_corrected.csv",
        CORRECTED_OUT / "v2_realized_ohlc_repairs_pretest_1_0.csv": ORIGINAL_OUT / "v2_realized_ohlc_repairs_pretest_1_0_corrected.csv",
        CORRECTED_OUT / "v2_realized_split_events_pretest_1_0.csv": ORIGINAL_OUT / "v2_realized_split_events_pretest_1_0_corrected.csv",
        CORRECTED_OUT / "v2_realized_raw_reconstruction_audit_pretest_1_0.csv": ORIGINAL_OUT / "v2_realized_raw_reconstruction_audit_pretest_1_0_corrected.csv",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)

    print(json.dumps({
        "status": summary["status"],
        "execution_spec_sha256": base.EXPECTED_SPEC_SHA,
        "data_representation_correction_sha256": CORRECTION_SHA,
        "winner_analysis": summary["winner_analysis"],
        "validation_ending_equity": {
            s: summary["results"][s]["validation"]["ending_equity_jpy"]
            for s in base.STRATEGIES
        },
        "all_ending_equity": {
            s: summary["results"][s]["all"]["ending_equity_jpy"]
            for s in base.STRATEGIES
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify_frozen_inputs()
    base.download_frames = download_frames
    base.main()
    append_audit_and_publish()
