#!/usr/bin/env python3
"""PRETEST 1.0 runner with pre-registered universal OHLC-envelope repair.

Trading/execution rules remain frozen in v2_execution_spec_pretest_1_0.json.
This runner only applies the separately frozen data-quality addendum before the
base simulator runs, and audits every repaired row.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import research_v2_realized_backtest_pretest_1_0 as base

ADDENDUM_PATH = Path("v2_execution_data_quality_addendum_pretest_1_0.json")
EXPECTED_ADDENDUM_SHA = "a1c8e2bbf515e7625474f9eb66757f64126f168ec721cb7ce6348e2742bc873e"
REPAIRS: list[dict] = []


def verify_addendum():
    if not ADDENDUM_PATH.exists():
        base.stop("DATA_QUALITY_ADDENDUM_MISSING")
    actual = hashlib.sha256(ADDENDUM_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_ADDENDUM_SHA:
        base.stop("DATA_QUALITY_ADDENDUM_SHA_MISMATCH", {"actual": actual})
    obj = json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))
    if obj.get("parent_execution_spec_sha256") != base.EXPECTED_SPEC_SHA:
        base.stop("DATA_QUALITY_ADDENDUM_PARENT_MISMATCH")
    if obj.get("v1_2_sha256") != base.V1_2_SHA or obj.get("v1_2_modified") is not False:
        base.stop("DATA_QUALITY_ADDENDUM_V1_2_MISMATCH")


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

        # Frozen universal data-quality rule: expand only H/L enough to contain
        # all reported OHLC prints; Open/Close/Volume/Splits remain untouched.
        for d in g.index:
            op = float(g.at[d, "Open"])
            hi = float(g.at[d, "High"])
            lo = float(g.at[d, "Low"])
            cl = float(g.at[d, "Close"])
            new_hi = max(op, hi, lo, cl)
            new_lo = min(op, hi, lo, cl)
            if new_hi != hi or new_lo != lo:
                REPAIRS.append({
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

        bad = g[
            (g["High"] < g[["Open", "Close", "Low"]].max(axis=1))
            | (g["Low"] > g[["Open", "Close", "High"]].min(axis=1))
            | (g["Volume"] < 0)
        ]
        if len(bad):
            problems.append({"code": code, "problem": "post_repair_ohlcv_sanity", "rows": int(len(bad))})
            continue

        g["ExecSMA75"] = base.split_only_sma75(g)
        frames[code] = g

    if problems:
        base.stop("PRICE_DATA_PROBLEM_AFTER_FROZEN_REPAIR", {"count": len(problems), "items": problems[:100]})
    common_latest = min(g.index.max() for g in frames.values())
    frames = {c: g[g.index <= common_latest].copy() for c, g in frames.items()}
    return frames, common_latest, split_col_missing


def append_repair_audit():
    summary_path = base.OUT / "v2_realized_summary_pretest_1_0.json"
    report_path = base.OUT / "v2_realized_report_pretest_1_0.md"
    if not summary_path.exists():
        base.stop("REALIZED_SUMMARY_MISSING_AFTER_BASE_RUN")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["data_quality_addendum_file"] = str(ADDENDUM_PATH)
    summary["data_quality_addendum_sha256"] = EXPECTED_ADDENDUM_SHA
    summary["ohlc_envelope_repair_count"] = len(REPAIRS)
    summary["ohlc_envelope_repairs"] = REPAIRS
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_df = pd.DataFrame(REPAIRS)
    audit_df.to_csv(base.OUT / "v2_realized_ohlc_repairs_pretest_1_0.csv", index=False, encoding="utf-8-sig")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n## データ品質監査\n\n")
        f.write(f"- 事前固定データ品質アドオンSHA256: `{EXPECTED_ADDENDUM_SHA}`\n")
        f.write(f"- OHLC包絡修復行数: **{len(REPAIRS)}**\n")
        f.write("- 修復は全期間・全銘柄へ同じ規則を適用し、Open/Close/Volume/Stock Splitsは変更していない。\n")
        f.write("- 修復全件は `research/v2_realized_ohlc_repairs_pretest_1_0.csv` に保存。\n")


if __name__ == "__main__":
    verify_addendum()
    base.download_frames = download_frames
    base.main()
    append_repair_audit()
