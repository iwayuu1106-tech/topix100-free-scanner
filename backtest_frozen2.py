#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SPEC_PATH = Path("frozen2_pretest_1_0.json")
EXPECTED_SPEC_SHA = "f12bc8d4271bb43856c6066a47aa7df95b8a244a76b01727606fb7a114b92914"
UNIVERSE_PATH = Path("universe.json")
OUT = Path("research/frozen2_pretest_1_0")
OUT.mkdir(parents=True, exist_ok=True)

START_FETCH = "2019-01-01"
END_FETCH_EXCLUSIVE = "2026-09-09"
SIGNAL_START = pd.Timestamp("2020-01-01")
SIGNAL_END = pd.Timestamp("2025-12-30")
DEV_END = pd.Timestamp("2022-12-30")
HOLDOUT_START = pd.Timestamp("2023-01-01")
CAPITAL0 = 300_000.0
LOT = 100
RISK_FRACTION = 0.01
MIN_TURNOVER = 500_000_000.0
BENCHMARK = "1306"

def fail(reason, details=None):
    obj = {
        "status": "STOP",
        "reason": reason,
        "details": details or {},
        "spec_sha256": EXPECTED_SPEC_SHA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    raise SystemExit(2)

def verify_spec():
    if not SPEC_PATH.exists():
        fail("SPEC_MISSING")
    actual = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_SPEC_SHA:
        fail("SPEC_SHA_MISMATCH", {"expected": EXPECTED_SPEC_SHA, "actual": actual})
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))

def extract_frame(big: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(big.columns, pd.MultiIndex):
        return big.copy()
    if ticker in big.columns.get_level_values(0):
        return big[ticker].copy()
    if ticker in big.columns.get_level_values(1):
        return big.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()

def future_split_factor(splits: pd.Series) -> pd.Series:
    sf = pd.to_numeric(splits, errors="coerce").fillna(0.0).astype(float)
    sf = sf.where(sf > 0, 1.0)
    inclusive = sf.iloc[::-1].cumprod().iloc[::-1]
    return inclusive / sf

def reconstruct_raw(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    fac = future_split_factor(g["Stock Splits"])
    for c in ["Open", "High", "Low", "Close"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").astype(float) * fac
    return g

def add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Split-only indicators in each date's then-current share units; dividends ignored."""
    g = g.copy().sort_index()
    split = pd.to_numeric(g["Stock Splits"], errors="coerce").fillna(0.0).astype(float)
    split_mult = split.where(split > 0, 1.0)
    cum = split_mult.cumprod()
    g["CumSplit"] = cum

    close = pd.to_numeric(g["Close"], errors="coerce").astype(float)
    high = pd.to_numeric(g["High"], errors="coerce").astype(float)
    low = pd.to_numeric(g["Low"], errors="coerce").astype(float)

    scaled_close = close * cum
    scaled_high = high * cum
    scaled_low = low * cum

    g["SMA25"] = scaled_close.rolling(25).mean() / cum
    g["SMA75"] = scaled_close.rolling(75).mean() / cum
    g["SMA75_5"] = g["SMA75"].shift(5)

    prev_scaled_close = scaled_close.shift(1)
    tr_scaled = pd.concat([
        scaled_high - scaled_low,
        (scaled_high - prev_scaled_close).abs(),
        (scaled_low - prev_scaled_close).abs(),
    ], axis=1).max(axis=1)
    g["ATR14"] = tr_scaled.rolling(14).mean() / cum

    g["Turnover"] = close * pd.to_numeric(g["Volume"], errors="coerce").astype(float)
    g["Turnover20Med"] = g["Turnover"].rolling(20).median()
    g["HistoryRows"] = np.arange(1, len(g) + 1)

    g["Signal"] = (
        (g["SMA75"] > g["SMA75_5"])
        & (g["SMA25"] > g["SMA75"])
        & (close <= g["SMA25"])
        & (close.shift(1) <= g["SMA75"].shift(1))
        & (close > g["SMA75"])
        & (g["Turnover20Med"] >= MIN_TURNOVER)
        & (g["HistoryRows"] >= 120)
        & g["ATR14"].notna()
    )
    return g

def download_all(codes):
    all_codes = list(dict.fromkeys(codes + [BENCHMARK]))
    tickers = [f"{c}.T" for c in all_codes]
    try:
        raw = yf.download(
            tickers=tickers,
            start=START_FETCH,
            end=END_FETCH_EXCLUSIVE,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            actions=True,
            repair=False,
            threads=True,
            progress=False,
            timeout=90,
        )
    except Exception as e:
        fail("YFINANCE_FETCH_EXCEPTION", {"error": repr(e)})
    if raw is None or raw.empty:
        fail("YFINANCE_FETCH_EMPTY")

    required = ["Open", "High", "Low", "Close", "Volume"]
    frames = {}
    issues = []
    for code in all_codes:
        ticker = f"{code}.T"
        g = extract_frame(raw, ticker)
        if g.empty:
            issues.append({"code": code, "problem": "no_frame"})
            continue
        miss = [c for c in required if c not in g.columns]
        if miss:
            issues.append({"code": code, "problem": "missing_columns", "columns": miss})
            continue
        cols = required + (["Stock Splits"] if "Stock Splits" in g.columns else [])
        g = g[cols].copy()
        if "Stock Splits" not in g.columns:
            g["Stock Splits"] = 0.0
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = g.dropna(subset=required)
        if len(g) < 120:
            issues.append({"code": code, "problem": "under_120_rows", "rows": int(len(g))})
            continue

        op = pd.to_numeric(g["Open"], errors="coerce")
        hi = pd.to_numeric(g["High"], errors="coerce")
        lo = pd.to_numeric(g["Low"], errors="coerce")
        cl = pd.to_numeric(g["Close"], errors="coerce")
        g["High"] = pd.concat([op, hi, lo, cl], axis=1).max(axis=1)
        g["Low"] = pd.concat([op, hi, lo, cl], axis=1).min(axis=1)

        g = reconstruct_raw(g)
        g = add_indicators(g)
        frames[code] = g

    if BENCHMARK not in frames:
        fail("BENCHMARK_MISSING", {"issues": issues})
    usable_codes = [c for c in codes if c in frames]
    if len(usable_codes) < max(30, int(len(codes) * 0.90)):
        fail("TOO_MANY_UNIVERSE_DATA_GAPS", {"requested": len(codes), "usable": len(usable_codes), "issues": issues})
    return frames, usable_codes, issues

def make_regime(bench: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(bench["Close"], errors="coerce")
    sma = bench["SMA75"]
    prev20 = sma.shift(20)
    r = pd.Series("RANGE", index=bench.index, dtype=object)
    r[(c > sma) & (sma > prev20)] = "UP"
    r[(c < sma) & (sma < prev20)] = "DOWN"
    return r

def max_losing_streak(pnls):
    best = cur = 0
    for x in pnls:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def metrics_from_trades(df: pd.DataFrame, max_dd=None):
    if df is None or len(df) == 0:
        return {
            "n": 0, "win_rate": None, "avg_gain_jpy": None, "avg_loss_jpy": None,
            "avg_gain_R": None, "avg_loss_R": None, "avg_win_loss_R_ratio": None,
            "expectancy_jpy": None, "expectancy_R": None, "profit_factor": None,
            "max_drawdown": max_dd, "max_losing_streak": 0,
            "avg_holding_sessions": None, "median_holding_sessions": None, "max_holding_sessions": None,
            "largest_winner_jpy": None, "largest_winner_R": None,
            "largest_loser_jpy": None, "largest_loser_R": None,
            "top1_gross_profit_share": None, "top5_gross_profit_share": None, "top10_gross_profit_share": None,
        }
    d = df.copy()
    p = pd.to_numeric(d["pnl_jpy"], errors="coerce")
    r = pd.to_numeric(d["R_multiple"], errors="coerce")
    wins = d[p > 0]
    losses = d[p < 0]
    gross_win = float(wins["pnl_jpy"].sum()) if len(wins) else 0.0
    gross_loss = float(-losses["pnl_jpy"].sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else None)
    avg_gain_r = float(wins["R_multiple"].mean()) if len(wins) else None
    avg_loss_r = float(losses["R_multiple"].mean()) if len(losses) else None
    ratio = (avg_gain_r / abs(avg_loss_r)) if (avg_gain_r is not None and avg_loss_r not in (None, 0)) else None

    win_sorted = wins.sort_values("pnl_jpy", ascending=False)
    def share(k):
        if gross_win <= 0:
            return None
        return float(win_sorted.head(k)["pnl_jpy"].sum() / gross_win)

    return {
        "n": int(len(d)),
        "win_rate": float((p > 0).mean()),
        "avg_gain_jpy": float(wins["pnl_jpy"].mean()) if len(wins) else None,
        "avg_loss_jpy": float(losses["pnl_jpy"].mean()) if len(losses) else None,
        "avg_gain_R": avg_gain_r,
        "avg_loss_R": avg_loss_r,
        "avg_win_loss_R_ratio": ratio,
        "expectancy_jpy": float(p.mean()),
        "expectancy_R": float(r.mean()),
        "profit_factor": float(pf) if pf is not None and np.isfinite(pf) else ("INF" if pf == float("inf") else None),
        "max_drawdown": max_dd,
        "max_losing_streak": int(max_losing_streak(p.tolist())),
        "avg_holding_sessions": float(d["holding_sessions"].mean()),
        "median_holding_sessions": float(d["holding_sessions"].median()),
        "max_holding_sessions": int(d["holding_sessions"].max()),
        "largest_winner_jpy": float(p.max()),
        "largest_winner_R": float(r.loc[p.idxmax()]),
        "largest_loser_jpy": float(p.min()),
        "largest_loser_R": float(r.loc[p.idxmin()]),
        "top1_gross_profit_share": share(1),
        "top5_gross_profit_share": share(5),
        "top10_gross_profit_share": share(10),
    }

def drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    roll = equity.cummax()
    dd = equity / roll - 1.0
    return float(dd.min())

def winner_removal(df, n):
    if len(df) == 0:
        return {"removed": n, "remaining_n": 0}
    winners = df[df["pnl_jpy"] > 0].sort_values("pnl_jpy", ascending=False)
    remove_idx = winners.head(n).index
    rem = df.drop(index=remove_idx)
    m = metrics_from_trades(rem)
    return {
        "removed": int(n),
        "remaining_n": int(len(rem)),
        "total_pnl_jpy": float(rem["pnl_jpy"].sum()) if len(rem) else 0.0,
        "expectancy_jpy": m["expectancy_jpy"],
        "expectancy_R": m["expectancy_R"],
        "profit_factor": m["profit_factor"],
    }

def simulate(frames, codes, regime, rt_cost):
    one_way = rt_cost / 2.0
    calendar = frames[BENCHMARK].index
    calendar = calendar[(calendar >= SIGNAL_START) & (calendar <= pd.Timestamp(END_FETCH_EXCLUSIVE) - pd.Timedelta(days=1))]
    if len(calendar) == 0:
        fail("EMPTY_CALENDAR")

    cash = CAPITAL0
    pos = None
    scheduled = None
    trades = []
    daily = []
    skipped = defaultdict(int)
    last_close = {}

    sig_by_date = defaultdict(list)
    for code in codes:
        g = frames[code]
        q = g[(g.index >= SIGNAL_START) & (g.index <= SIGNAL_END) & g["Signal"]]
        for d, row in q.iterrows():
            sig_by_date[pd.Timestamp(d)].append({
                "code": code,
                "turnover20": float(row["Turnover20Med"]),
                "atr": float(row["ATR14"]),
                "cum_split": float(row["CumSplit"]),
            })

    for day in calendar:
        for code in codes:
            g = frames[code]
            if day in g.index:
                last_close[code] = float(g.at[day, "Close"])

        if pos is not None:
            g = frames[pos["code"]]
            if day in g.index:
                ratio = float(g.at[day, "Stock Splits"]) if pd.notna(g.at[day, "Stock Splits"]) else 0.0
                if ratio > 0 and abs(ratio - 1.0) > 1e-12 and day > pos["entry_date"]:
                    pos["shares"] *= ratio
                    pos["stop"] /= ratio
                    pos["high_water"] /= ratio

        if pos is None and scheduled is not None and scheduled["entry_date"] == day:
            code = scheduled["code"]
            g = frames[code]
            if day not in g.index:
                skipped["entry_missing_or_halted"] += 1
                scheduled = None
            else:
                entry = float(g.at[day, "Open"])
                entry_cum = float(g.at[day, "CumSplit"])
                risk_per_share = float(scheduled["atr"]) * float(scheduled["cum_split"]) / entry_cum
                risk_budget = cash * RISK_FRACTION
                q_risk = math.floor((risk_budget / risk_per_share) / LOT) * LOT if risk_per_share > 0 else 0
                q_cash = math.floor((cash / (entry * (1.0 + one_way))) / LOT) * LOT if entry > 0 else 0
                shares = int(min(q_risk, q_cash))
                if shares < LOT:
                    skipped["sizing_under_100"] += 1
                    scheduled = None
                else:
                    entry_notional = entry * shares
                    entry_fee = entry_notional * one_way
                    cash -= entry_notional + entry_fee
                    initial_stop = entry - risk_per_share
                    initial_risk = risk_per_share * shares
                    pos = {
                        "code": code,
                        "signal_date": scheduled["signal_date"],
                        "entry_date": day,
                        "entry_price": entry,
                        "entry_shares": shares,
                        "shares": float(shares),
                        "entry_notional": entry_notional,
                        "entry_fee": entry_fee,
                        "initial_risk_jpy": initial_risk,
                        "stop": initial_stop,
                        "high_water": float(g.at[day, "High"]),
                        "holding_sessions": 1,
                        "regime": scheduled["regime"],
                    }
                    scheduled = None

                    if float(g.at[day, "Low"]) <= pos["stop"]:
                        exit_price = float(pos["stop"])
                        proceeds = exit_price * pos["shares"]
                        exit_fee = proceeds * one_way
                        pnl = proceeds - exit_fee - pos["entry_notional"] - pos["entry_fee"]
                        cash += proceeds - exit_fee
                        trades.append({
                            "code": code, "signal_date": pos["signal_date"].date().isoformat(),
                            "entry_date": day.date().isoformat(), "exit_date": day.date().isoformat(),
                            "entry_price": pos["entry_price"], "exit_price": exit_price,
                            "entry_shares": pos["entry_shares"], "exit_shares": pos["shares"],
                            "initial_risk_jpy": pos["initial_risk_jpy"],
                            "pnl_jpy": pnl, "R_multiple": pnl / pos["initial_risk_jpy"],
                            "holding_sessions": pos["holding_sessions"], "exit_reason": "INITIAL_STOP_INTRADAY",
                            "regime": pos["regime"], "entry_year": int(day.year),
                        })
                        pos = None
                    else:
                        atr_day = float(g.at[day, "ATR14"])
                        candidate = pos["high_water"] - 3.0 * atr_day
                        pos["stop"] = max(pos["stop"], candidate)

        if pos is not None and day > pos["entry_date"]:
            code = pos["code"]
            g = frames[code]
            if day in g.index:
                pos["holding_sessions"] += 1
                op = float(g.at[day, "Open"])
                lo = float(g.at[day, "Low"])
                exit_price = None
                reason = None
                if op <= pos["stop"]:
                    exit_price = op
                    reason = "TRAIL_GAP_OPEN"
                elif lo <= pos["stop"]:
                    exit_price = float(pos["stop"])
                    reason = "TRAIL_STOP"
                if exit_price is not None:
                    proceeds = exit_price * pos["shares"]
                    exit_fee = proceeds * one_way
                    pnl = proceeds - exit_fee - pos["entry_notional"] - pos["entry_fee"]
                    cash += proceeds - exit_fee
                    trades.append({
                        "code": code, "signal_date": pos["signal_date"].date().isoformat(),
                        "entry_date": pos["entry_date"].date().isoformat(), "exit_date": day.date().isoformat(),
                        "entry_price": pos["entry_price"], "exit_price": exit_price,
                        "entry_shares": pos["entry_shares"], "exit_shares": pos["shares"],
                        "initial_risk_jpy": pos["initial_risk_jpy"],
                        "pnl_jpy": pnl, "R_multiple": pnl / pos["initial_risk_jpy"],
                        "holding_sessions": pos["holding_sessions"], "exit_reason": reason,
                        "regime": pos["regime"], "entry_year": int(pos["entry_date"].year),
                    })
                    pos = None
                else:
                    pos["high_water"] = max(pos["high_water"], float(g.at[day, "High"]))
                    atr_day = float(g.at[day, "ATR14"])
                    candidate = pos["high_water"] - 3.0 * atr_day
                    pos["stop"] = max(pos["stop"], candidate)

        if pos is None and scheduled is None and day <= SIGNAL_END:
            cands = sig_by_date.get(day, [])
            if cands:
                chosen = sorted(cands, key=lambda x: (-x["turnover20"], x["code"]))[0]
                idx = calendar.searchsorted(day, side="right")
                if idx < len(calendar):
                    ed = calendar[idx]
                    scheduled = {
                        **chosen,
                        "signal_date": day,
                        "entry_date": ed,
                        "regime": regime.get(day, "RANGE"),
                    }
                skipped["candidate_signals_total"] += len(cands)
                skipped["candidate_signals_not_selected"] += max(0, len(cands) - 1)

        equity = cash
        if pos is not None:
            code = pos["code"]
            px = last_close.get(code)
            if px is not None:
                equity += pos["shares"] * px
        daily.append({
            "date": day.date().isoformat(),
            "equity": float(equity),
            "cash": float(cash),
            "holding_code": pos["code"] if pos else None,
            "scheduled_code": scheduled["code"] if scheduled else None,
        })

    tdf = pd.DataFrame(trades)
    ddf = pd.DataFrame(daily)
    if len(ddf):
        ddf["date"] = pd.to_datetime(ddf["date"])
    open_position = None
    if pos is not None:
        px = last_close.get(pos["code"])
        mtm = (pos["shares"] * px - pos["entry_notional"] - pos["entry_fee"]) if px is not None else None
        open_position = {
            "code": pos["code"], "entry_date": pos["entry_date"].date().isoformat(),
            "shares": pos["shares"], "active_stop": pos["stop"], "mark_price": px,
            "unrealized_before_exit_cost_jpy": mtm,
        }
    return tdf, ddf, dict(skipped), open_position

def phase_metrics(tdf, phase):
    if len(tdf) == 0:
        return metrics_from_trades(tdf)
    ed = pd.to_datetime(tdf["entry_date"])
    if phase == "development":
        sub = tdf[ed <= DEV_END]
    else:
        sub = tdf[(ed >= HOLDOUT_START) & (ed <= SIGNAL_END)]
    return metrics_from_trades(sub)

def by_groups(tdf, field):
    out = {}
    if len(tdf) == 0:
        return out
    for k, sub in tdf.groupby(field):
        out[str(k)] = metrics_from_trades(sub)
    return out

def annual_equity_returns(ddf):
    if len(ddf) == 0:
        return {}
    q = ddf.set_index("date")["equity"].sort_index()
    year_end = q.groupby(q.index.year).last()
    out = {}
    prev = CAPITAL0
    for y, v in year_end.items():
        out[str(int(y))] = {
            "year_end_equity_jpy": float(v),
            "return": float(v / prev - 1.0),
        }
        prev = float(v)
    return out

def verdict(base_m, stress_m, dev_m, hold_m):
    n = base_m["n"]
    er = base_m["expectancy_R"]
    pf = base_m["profit_factor"]
    ratio = base_m["avg_win_loss_R_ratio"]
    dd = abs(base_m["max_drawdown"] or 0.0)
    ser = stress_m["expectancy_R"]
    if er is None or er <= 0 or (isinstance(pf, (int, float)) and pf <= 1.0) or dd > 0.25 or (ser is not None and ser < -0.10):
        numeric = "却下"
    elif (
        n >= 50 and er > 0.20 and isinstance(pf, (int, float)) and pf >= 1.30
        and ratio is not None and ratio >= 2.0 and dd <= 0.15
        and ser is not None and ser > 0
        and dev_m["expectancy_R"] is not None and dev_m["expectancy_R"] > 0
        and hold_m["expectancy_R"] is not None and hold_m["expectancy_R"] > 0
    ):
        numeric = "採用"
    elif (
        n >= 30 and er > 0 and isinstance(pf, (int, float)) and pf > 1.10
        and ratio is not None and ratio > 1.5 and dd <= 0.20
        and ser is not None and ser >= 0
    ):
        numeric = "条件付き採用"
    else:
        numeric = "保留"
    final = "条件付き採用" if numeric == "採用" else numeric
    return numeric, final

def main():
    spec = verify_spec()
    uni = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    codes = [str(x).zfill(4) for x in uni["codes_4digit"]]
    frames, usable, issues = download_all(codes)
    regime = make_regime(frames[BENCHMARK])

    runs = {}
    for label, rt in [("cost_0_10pct", 0.001), ("cost_0_20pct", 0.002)]:
        tdf, ddf, skipped, open_pos = simulate(frames, usable, regime, rt)
        maxdd = drawdown(ddf.set_index("date")["equity"]) if len(ddf) else 0.0
        m = metrics_from_trades(tdf, maxdd)
        dev = phase_metrics(tdf, "development")
        hold = phase_metrics(tdf, "holdout")
        runs[label] = {
            "round_trip_cost": rt,
            "metrics": m,
            "development_2020_2022": dev,
            "holdout_2023_2025": hold,
            "year_by_entry": by_groups(tdf, "entry_year"),
            "regime_by_entry": by_groups(tdf, "regime"),
            "annual_equity": annual_equity_returns(ddf),
            "winner_removal": {
                "remove_top1": winner_removal(tdf, 1),
                "remove_top3": winner_removal(tdf, 3),
                "remove_top5": winner_removal(tdf, 5),
            },
            "skipped": skipped,
            "open_position_at_data_end": open_pos,
            "ending_equity_jpy": float(ddf.iloc[-1]["equity"]) if len(ddf) else CAPITAL0,
        }
        tdf.to_csv(OUT / f"trades_{label}.csv", index=False, encoding="utf-8-sig")
        ddf.to_csv(OUT / f"daily_equity_{label}.csv", index=False, encoding="utf-8-sig")

    base = runs["cost_0_10pct"]
    stress = runs["cost_0_20pct"]
    numeric, final = verdict(
        base["metrics"], stress["metrics"],
        base["development_2020_2022"], base["holdout_2023_2025"]
    )
    summary = {
        "status": "PASS_FROZEN2_PRETEST",
        "spec_sha256": EXPECTED_SPEC_SHA,
        "spec": spec,
        "requested_universe_count": len(codes),
        "usable_universe_count": len(usable),
        "universe_data_issues": issues,
        "data_source": "Yahoo Finance via yfinance; actual Japanese OHLCV; raw historical yen/share reconstructed for split handling",
        "evidence_limitations": [
            "Current monitored TOPIX100 constituents are applied historically (survivorship/constituent bias).",
            "This is not the exact historical all-TSE universe required by the ideal FROZEN-2 target.",
            "J-Quants official execution backtest remains outstanding."
        ],
        "runs": runs,
        "numeric_verdict_before_evidence_cap": numeric,
        "final_verdict_with_evidence_cap": final,
        "generated_at_utc": datetime.now(timezone.utc).isoformat()
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    bm = base["metrics"]
    sm = stress["metrics"]
    def pct(x): return "N/A" if x is None else f"{100*x:.2f}%"
    def num(x, digits=2): return "N/A" if x is None else f"{x:.{digits}f}"
    lines = [
        "# FROZEN-2 PRETEST 1.0", "",
        f"- Spec SHA256: `{EXPECTED_SPEC_SHA}`",
        "- Rules were frozen before this backtest.",
        f"- Universe proxy: current monitored TOPIX100 ({len(usable)}/{len(codes)} usable), not historical all-TSE.",
        "- Data: Yahoo Finance/yfinance actual Japanese OHLCV; split handling reconstructed to historical raw yen/share.", "",
        "## Base cost 0.10% round trip", "",
        f"- Completed trades: **{bm['n']}**",
        f"- Win rate: **{pct(bm['win_rate'])}**",
        f"- Avg gain: **¥{num(bm['avg_gain_jpy'],0)} / {num(bm['avg_gain_R'])}R**",
        f"- Avg loss: **¥{num(bm['avg_loss_jpy'],0)} / {num(bm['avg_loss_R'])}R**",
        f"- Avg win/loss R ratio: **{num(bm['avg_win_loss_R_ratio'])}x**",
        f"- Expectancy: **¥{num(bm['expectancy_jpy'],0)} / {num(bm['expectancy_R'])}R per trade**",
        f"- Profit factor: **{bm['profit_factor']}**",
        f"- Max drawdown: **{pct(bm['max_drawdown'])}**",
        f"- Max losing streak: **{bm['max_losing_streak']}**",
        f"- Avg / median / max holding sessions: **{num(bm['avg_holding_sessions'])} / {num(bm['median_holding_sessions'])} / {bm['max_holding_sessions']}**",
        f"- Largest winner: **¥{num(bm['largest_winner_jpy'],0)} / {num(bm['largest_winner_R'])}R**",
        f"- Largest loser: **¥{num(bm['largest_loser_jpy'],0)} / {num(bm['largest_loser_R'])}R**",
        f"- Ending equity: **¥{num(base['ending_equity_jpy'],0)}**", "",
        "## Stress cost 0.20%",
        f"- Expectancy R: **{num(sm['expectancy_R'])}R**",
        f"- Profit factor: **{sm['profit_factor']}**",
        f"- Ending equity: **¥{num(stress['ending_equity_jpy'],0)}**", "",
        "## Temporal split",
        f"- 2020-2022 expectancy R: **{num(base['development_2020_2022']['expectancy_R'])}R**, PF **{base['development_2020_2022']['profit_factor']}**",
        f"- 2023-2025 expectancy R: **{num(base['holdout_2023_2025']['expectancy_R'])}R**, PF **{base['holdout_2023_2025']['profit_factor']}**", "",
        "## Winner concentration",
        f"- Top1 share of gross profits: **{pct(bm['top1_gross_profit_share'])}**",
        f"- Top5 share of gross profits: **{pct(bm['top5_gross_profit_share'])}**",
        f"- Top10 share of gross profits: **{pct(bm['top10_gross_profit_share'])}**",
        f"- Remove top1: **{json.dumps(base['winner_removal']['remove_top1'], ensure_ascii=False)}**",
        f"- Remove top3: **{json.dumps(base['winner_removal']['remove_top3'], ensure_ascii=False)}**",
        f"- Remove top5: **{json.dumps(base['winner_removal']['remove_top5'], ensure_ascii=False)}**", "",
        "## Verdict",
        f"- Numeric verdict: **{numeric}**",
        f"- Final verdict after evidence-quality cap: **{final}**", "",
        "The evidence-quality cap is pre-registered because this run is not historical all-TSE J-Quants."
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status":"PASS","verdict":final,"base":bm,"stress":sm}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
