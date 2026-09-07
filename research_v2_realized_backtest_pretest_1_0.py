#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("research")
OUT.mkdir(parents=True, exist_ok=True)

SPEC_PATH = Path("v2_execution_spec_pretest_1_0.json")
EXPECTED_SPEC_SHA = "27891cb9d0f43ae59266a83c263d96cc17c2a95e624ce4b4400d0e9948a4dd83"
V1_2_SHA = "c2aef8f85dfdf125285623b7b6556b78a43509124ee09985b047b365f929d3a6"
EVENTS_PATH = OUT / "expectancy_foundation_enriched.csv"
CAPITAL = 300_000.0
LOT = 100.0

DISCOVERY_SIGNAL_END = pd.Timestamp("2024-07-08")
DISCOVERY_MARK_END = pd.Timestamp("2024-09-06")
VALIDATION_SIGNAL_START = pd.Timestamp("2024-09-09")
MIN_WINNER_TRADES = 30

STRATEGIES = {
    "S1_V2_FOUNDATION": lambda d: pd.Series(True, index=d.index),
    "S2_V2_PLUS_SMA75_SLOPE_GE_0_75PCT": lambda d: pd.to_numeric(d["sma75_slope5_pct"], errors="coerce") >= 0.0075,
    "S3_V2_PLUS_RS20_GE_3PCT": lambda d: pd.to_numeric(d["rs20_vs_topix"], errors="coerce") >= 0.03,
    "S4_V2_PLUS_BOTH": lambda d: (
        (pd.to_numeric(d["sma75_slope5_pct"], errors="coerce") >= 0.0075)
        & (pd.to_numeric(d["rs20_vs_topix"], errors="coerce") >= 0.03)
    ),
}


def stop(reason: str, details=None):
    obj = {
        "status": "STOP",
        "reason": reason,
        "details": details or {},
        "spec_sha256_expected": EXPECTED_SPEC_SHA,
        "v1_2_sha256": V1_2_SHA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "v2_realized_summary_pretest_1_0.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    raise SystemExit(2)


def verify_spec():
    if not SPEC_PATH.exists():
        stop("PRETEST_SPEC_MISSING")
    actual = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_SPEC_SHA:
        stop("PRETEST_SPEC_SHA_MISMATCH", {"actual": actual})
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["lineage"]["frozen_v1_2_sha256"] != V1_2_SHA or not spec["lineage"]["v1_2_unchanged"]:
        stop("V1_2_LINEAGE_MISMATCH")
    return spec


def tick_size(price: float) -> float:
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
    t = tick_size(float(x))
    return math.floor(float(x) / t + 1.0 + 1e-12) * t


def one_tick_below(x: float) -> float:
    t = tick_size(float(x))
    return math.ceil(float(x) / t - 1.0 - 1e-12) * t


def extract_ticker_frame(big: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(big.columns, pd.MultiIndex):
        return big.copy()
    if ticker in big.columns.get_level_values(0):
        return big[ticker].copy()
    if ticker in big.columns.get_level_values(1):
        return big.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def split_only_sma75(g: pd.DataFrame) -> pd.Series:
    """
    At each date i, express the previous 75 raw closes in date-i share units
    using only splits that have become effective by i. Dividends are ignored.
    For cumulative split factor F, normalized historical close_j = close_j*F_j/F_i.
    """
    close = pd.to_numeric(g["Close"], errors="coerce").astype(float)
    splits = pd.to_numeric(g.get("Stock Splits", 0.0), errors="coerce").fillna(0.0).astype(float)
    split_factor = splits.where(splits > 0, 1.0)
    cumulative = split_factor.cumprod()
    scaled = close * cumulative
    sma = scaled.rolling(75).mean() / cumulative
    return sma


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
        stop("YFINANCE_FETCH_EXCEPTION", {"error": repr(e)})
    if raw is None or raw.empty:
        stop("YFINANCE_FETCH_EMPTY")

    now_jst = datetime.now(ZoneInfo("Asia/Tokyo"))
    required = ["Open", "High", "Low", "Close", "Volume"]
    frames = {}
    problems = []
    split_col_missing = []

    for code, ticker in zip(codes, tickers):
        g = extract_ticker_frame(raw, ticker)
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
        bad = g[
            (g["High"] < g[["Open", "Close", "Low"]].max(axis=1))
            | (g["Low"] > g[["Open", "Close", "High"]].min(axis=1))
            | (g["Volume"] < 0)
        ]
        if len(bad):
            problems.append({"code": code, "problem": "ohlcv_sanity", "rows": int(len(bad))})
            continue
        g["ExecSMA75"] = split_only_sma75(g)
        frames[code] = g

    if problems:
        stop("PRICE_DATA_PROBLEM", {"count": len(problems), "items": problems[:100]})
    common_latest = min(g.index.max() for g in frames.values())
    frames = {c: g[g.index <= common_latest].copy() for c, g in frames.items()}
    return frames, common_latest, split_col_missing


def next_trading_date(g: pd.DataFrame, d: pd.Timestamp):
    idx = g.index.searchsorted(d, side="right")
    if idx >= len(g.index):
        return None
    return g.index[idx]


def make_orders(events: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> list[dict]:
    orders = []
    missing = []
    for _, r in events.iterrows():
        code = str(r["code"]).zfill(4)
        d = pd.Timestamp(r["date"])
        g = frames.get(code)
        if g is None or d not in g.index:
            missing.append({"code": code, "date": d.date().isoformat(), "problem": "signal_date_missing"})
            continue
        sma = g.at[d, "ExecSMA75"]
        if pd.isna(sma):
            missing.append({"code": code, "date": d.date().isoformat(), "problem": "exec_sma75_nan"})
            continue
        nd = next_trading_date(g, d)
        if nd is None:
            continue
        orders.append({
            "code": code,
            "signal_date": d,
            "order_date": nd,
            "trigger": float(one_tick_above(float(sma))),
            "signal_exec_sma75": float(sma),
            "sma75_slope5_pct": float(r["sma75_slope5_pct"]) if pd.notna(r["sma75_slope5_pct"]) else np.nan,
            "rs20_vs_topix": float(r["rs20_vs_topix"]) if pd.notna(r["rs20_vs_topix"]) else np.nan,
        })
    if missing:
        stop("EVENT_EXECUTION_INPUT_MISSING", {"count": len(missing), "items": missing[:100]})
    return orders


def close_position(code, pos, date, price, reason, cash, trades, strategy, phase):
    proceeds = float(pos["shares"]) * float(price)
    cash += proceeds
    entry_cost = float(pos["entry_cost"])
    pnl = proceeds - entry_cost
    ret = pnl / entry_cost if entry_cost else np.nan
    trades.append({
        "strategy": strategy,
        "phase": phase,
        "code": code,
        "signal_date": pos["signal_date"].date().isoformat(),
        "entry_date": pos["entry_date"].date().isoformat(),
        "entry_price": float(pos["entry_price"]),
        "entry_shares": float(pos["entry_shares"]),
        "entry_cost_jpy": entry_cost,
        "initial_stop": float(pos["initial_stop"]) if pos.get("initial_stop") is not None else np.nan,
        "exit_date": pd.Timestamp(date).date().isoformat(),
        "exit_price": float(price),
        "exit_shares": float(pos["shares"]),
        "exit_reason": reason,
        "pnl_jpy": float(pnl),
        "trade_return": float(ret),
        "holding_calendar_days": int((pd.Timestamp(date) - pos["entry_date"]).days),
    })
    return cash


def simulate(strategy: str, phase: str, events: pd.DataFrame, frames: dict[str, pd.DataFrame],
             common_start: pd.Timestamp, common_latest: pd.Timestamp):
    mask = STRATEGIES[strategy](events)
    ev = events.loc[mask].copy()
    ev["date"] = pd.to_datetime(ev["date"])

    if phase == "discovery":
        sig = ev[ev["date"] <= DISCOVERY_SIGNAL_END].copy()
        sim_start = common_start
        sim_end = min(DISCOVERY_MARK_END, common_latest)
    elif phase == "validation":
        sig = ev[ev["date"] >= VALIDATION_SIGNAL_START].copy()
        sim_start = VALIDATION_SIGNAL_START
        sim_end = common_latest
    elif phase == "all":
        sig = ev.copy()
        sim_start = common_start
        sim_end = common_latest
    else:
        raise ValueError(phase)

    orders = make_orders(sig, frames)
    orders_by_date = defaultdict(list)
    for o in orders:
        if sim_start <= o["order_date"] <= sim_end:
            orders_by_date[o["order_date"]].append(o)

    all_dates = sorted({
        d for g in frames.values() for d in g.index
        if sim_start <= d <= sim_end
    })
    if not all_dates:
        stop("SIMULATION_DATE_RANGE_EMPTY", {"phase": phase})

    cash = CAPITAL
    positions = {}
    trades = []
    daily = []
    last_close = {}
    counters = {
        "candidate_event_count": int(len(sig)),
        "filled_entry_count": 0,
        "unfilled_expired_count": 0,
        "skipped_insufficient_cash_count": 0,
        "skipped_already_holding_count": 0,
    }

    for day in all_dates:
        today_orders = [dict(x) for x in orders_by_date.get(day, [])]
        for code, g in frames.items():
            if day not in g.index:
                continue
            ratio = float(g.at[day, "Stock Splits"]) if pd.notna(g.at[day, "Stock Splits"]) else 0.0
            if ratio > 0 and abs(ratio - 1.0) > 1e-12:
                if code in positions:
                    positions[code]["shares"] *= ratio
                    if positions[code].get("stop") is not None:
                        positions[code]["stop"] /= ratio
                    positions[code]["initial_stop"] = (
                        positions[code]["initial_stop"] / ratio
                        if positions[code].get("initial_stop") is not None else None
                    )
                for o in today_orders:
                    if o["code"] == code:
                        o["trigger"] /= ratio

        exited_at_open = set()

        for code in sorted(list(positions)):
            pos = positions.get(code)
            if not pos or not pos.get("scheduled_exit"):
                continue
            g = frames[code]
            if day not in g.index:
                continue
            px = float(g.at[day, "Open"])
            cash = close_position(code, pos, day, px, "SMA75_CLOSE_BREAK_NEXT_OPEN",
                                  cash, trades, strategy, phase)
            del positions[code]
            exited_at_open.add(code)

        for code in sorted(list(positions)):
            pos = positions.get(code)
            if not pos or pos.get("stop") is None or pos["entry_date"] >= day:
                continue
            g = frames[code]
            if day not in g.index:
                continue
            op = float(g.at[day, "Open"])
            if op <= float(pos["stop"]):
                cash = close_position(code, pos, day, op, "STOP_GAP_OPEN",
                                      cash, trades, strategy, phase)
                del positions[code]
                exited_at_open.add(code)

        reservations = []
        reserved_cash = 0.0
        for o in sorted(today_orders, key=lambda x: x["code"]):
            code = o["code"]
            g = frames[code]
            if day not in g.index:
                counters["unfilled_expired_count"] += 1
                continue
            if code in positions or code in exited_at_open:
                counters["skipped_already_holding_count"] += 1
                continue
            op = float(g.at[day, "Open"])
            reserve_price = max(float(o["trigger"]), op)
            reserve_cost = reserve_price * LOT
            if reserve_cost > cash - reserved_cash + 1e-9:
                counters["skipped_insufficient_cash_count"] += 1
                continue
            reserved_cash += reserve_cost
            o["reserve_cost"] = reserve_cost
            reservations.append(o)

        exited_intraday = set()
        for code in sorted(list(positions)):
            pos = positions.get(code)
            if not pos or pos.get("stop") is None or pos["entry_date"] >= day:
                continue
            g = frames[code]
            if day not in g.index:
                continue
            low = float(g.at[day, "Low"])
            if low <= float(pos["stop"]):
                cash = close_position(code, pos, day, float(pos["stop"]), "STOP_INTRADAY",
                                      cash, trades, strategy, phase)
                del positions[code]
                exited_intraday.add(code)

        newly_filled = set()
        for o in reservations:
            code = o["code"]
            if code in positions or code in exited_intraday:
                counters["skipped_already_holding_count"] += 1
                continue
            g = frames[code]
            op = float(g.at[day, "Open"])
            hi = float(g.at[day, "High"])
            trigger = float(o["trigger"])
            if op >= trigger:
                fill = op
                fill_reason = "BUY_GAP_OPEN"
            elif hi >= trigger:
                fill = trigger
                fill_reason = "BUY_STOP_INTRADAY"
            else:
                counters["unfilled_expired_count"] += 1
                continue
            cost = fill * LOT
            if cost > cash + 1e-6:
                stop("RESERVATION_CASH_INVARIANT_BROKEN", {
                    "strategy": strategy, "phase": phase, "code": code,
                    "day": day.date().isoformat(), "cost": cost, "cash": cash
                })
            cash -= cost
            positions[code] = {
                "signal_date": o["signal_date"],
                "entry_date": day,
                "entry_price": fill,
                "entry_shares": LOT,
                "shares": LOT,
                "entry_cost": cost,
                "stop": None,
                "initial_stop": None,
                "scheduled_exit": False,
                "entry_reason": fill_reason,
            }
            newly_filled.add(code)
            counters["filled_entry_count"] += 1

        for code in sorted(list(positions)):
            pos = positions[code]
            g = frames[code]
            if day not in g.index:
                continue
            cl = float(g.at[day, "Close"])
            sma = float(g.at[day, "ExecSMA75"]) if pd.notna(g.at[day, "ExecSMA75"]) else np.nan
            if code in newly_filled:
                if not np.isfinite(sma):
                    stop("ENTRY_DAY_EXEC_SMA75_NAN", {"code": code, "day": day.date().isoformat()})
                st = float(one_tick_below(sma))
                pos["stop"] = st
                pos["initial_stop"] = st
            if np.isfinite(sma) and cl < sma:
                pos["scheduled_exit"] = True
            else:
                pos["scheduled_exit"] = False

        market_value = 0.0
        for code, pos in positions.items():
            g = frames[code]
            if day in g.index:
                last_close[code] = float(g.at[day, "Close"])
            if code not in last_close:
                continue
            market_value += float(pos["shares"]) * last_close[code]
        equity = cash + market_value
        util = market_value / equity if equity > 0 else np.nan
        daily.append({
            "strategy": strategy,
            "phase": phase,
            "date": day.date().isoformat(),
            "cash_jpy": cash,
            "market_value_jpy": market_value,
            "equity_jpy": equity,
            "capital_utilization": util,
            "open_positions": len(positions),
        })

    daily_df = pd.DataFrame(daily)
    trade_df = pd.DataFrame(trades)
    eq = pd.to_numeric(daily_df["equity_jpy"], errors="coerce")
    running_max = eq.cummax()
    dd = eq / running_max - 1.0

    if len(trade_df):
        rets = pd.to_numeric(trade_df["trade_return"], errors="coerce")
        pnls = pd.to_numeric(trade_df["pnl_jpy"], errors="coerce")
        pos = rets[rets > 0]
        neg = rets[rets < 0]
        avg_gain = float(pos.mean()) if len(pos) else None
        avg_loss = float(neg.mean()) if len(neg) else None
        payoff = (avg_gain / abs(avg_loss)) if (avg_gain is not None and avg_loss not in [None, 0]) else None
        win_rate = float((rets > 0).mean())
        mean_ret = float(rets.mean())
        mean_pnl = float(pnls.mean())
    else:
        win_rate = mean_ret = mean_pnl = avg_gain = avg_loss = payoff = None

    ending_equity = float(eq.iloc[-1])
    total_return = ending_equity / CAPITAL - 1.0
    start_day = pd.Timestamp(daily_df.iloc[0]["date"])
    end_day = pd.Timestamp(daily_df.iloc[-1]["date"])
    days = max(1, int((end_day - start_day).days))
    annualized = (ending_equity / CAPITAL) ** (365.25 / days) - 1.0 if ending_equity > 0 else None

    summary = {
        **counters,
        "closed_trade_count": int(len(trade_df)),
        "win_rate": win_rate,
        "mean_pnl_per_trade_jpy": mean_pnl,
        "mean_trade_return": mean_ret,
        "avg_gain_return": avg_gain,
        "avg_loss_return": avg_loss,
        "payoff_ratio": payoff,
        "max_drawdown": float(dd.min()) if len(dd) else None,
        "total_return": float(total_return),
        "annualized_return": float(annualized) if annualized is not None else None,
        "ending_equity_jpy": ending_equity,
        "average_capital_utilization": float(pd.to_numeric(daily_df["capital_utilization"], errors="coerce").mean()),
        "open_positions_end": int(len(positions)),
        "simulation_start": start_day.date().isoformat(),
        "simulation_end": end_day.date().isoformat(),
    }
    return summary, trade_df, daily_df


def winner_analysis(results):
    eligibility = {}
    for s in STRATEGIES:
        d = results[s]["discovery"]
        v = results[s]["validation"]
        eligibility[s] = (
            d["closed_trade_count"] >= MIN_WINNER_TRADES
            and v["closed_trade_count"] >= MIN_WINNER_TRADES
        )

    pos_both = [
        s for s in STRATEGIES
        if eligibility[s]
        and results[s]["discovery"]["mean_trade_return"] is not None
        and results[s]["validation"]["mean_trade_return"] is not None
        and results[s]["discovery"]["mean_trade_return"] > 0
        and results[s]["validation"]["mean_trade_return"] > 0
    ]
    highest = max(pos_both, key=lambda s: results[s]["validation"]["mean_trade_return"]) if pos_both else None

    stable = None
    if pos_both:
        stable = min(
            pos_both,
            key=lambda s: (
                abs(results[s]["discovery"]["mean_trade_return"] - results[s]["validation"]["mean_trade_return"]),
                abs(results[s]["validation"]["max_drawdown"] or 0.0),
            ),
        )

    risk_pool = [
        s for s in STRATEGIES
        if eligibility[s]
        and results[s]["validation"]["mean_trade_return"] is not None
        and results[s]["validation"]["mean_trade_return"] > 0
    ]
    lowest_risk = min(
        risk_pool, key=lambda s: abs(results[s]["validation"]["max_drawdown"] or 0.0)
    ) if risk_pool else None

    return {
        "minimum_closed_trades_each_phase": MIN_WINNER_TRADES,
        "eligible": eligibility,
        "highest_expectancy": highest,
        "most_stable": stable,
        "lowest_risk": lowest_risk,
        "note": "事前固定した勝者ルールをそのまま適用。サンプル不足戦略は数値が良くても勝者にしない。",
    }


def pct(x):
    return "N/A" if x is None else f"{x*100:.2f}%"


def yen(x):
    return "N/A" if x is None else f"¥{x:,.0f}"


def build_report(summary):
    lines = []
    lines.append("# v2実売買実行仕様 PRETEST 1.0 — 事前固定後バックテスト")
    lines.append("")
    lines.append(f"- v1.2変更: **なし** (`{V1_2_SHA}`)")
    lines.append(f"- v2実行仕様SHA256: `{EXPECTED_SPEC_SHA}`")
    lines.append("- 仕様固定 → マニフェスト固定 → その後に初めて本バックテストを実行。")
    lines.append("- 金額は手数料・税金・配当なしの価格売買バックテスト。")
    lines.append("")
    lines.append("## Validation（2024-09-09以降）")
    lines.append("")
    lines.append("| strategy | 候補 | 約定 | 決済 | 勝率 | 1取引平均 | 平均取引率 | 最大DD | 30万円→ | 年率 | 稼働率 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in STRATEGIES:
        m = summary["results"][s]["validation"]
        lines.append(
            f"| {s} | {m['candidate_event_count']} | {m['filled_entry_count']} | {m['closed_trade_count']} | "
            f"{pct(m['win_rate'])} | {yen(m['mean_pnl_per_trade_jpy'])} | {pct(m['mean_trade_return'])} | "
            f"{pct(m['max_drawdown'])} | {yen(m['ending_equity_jpy'])} | {pct(m['annualized_return'])} | "
            f"{pct(m['average_capital_utilization'])} |"
        )
    lines.append("")
    lines.append("## Discovery")
    lines.append("")
    lines.append("| strategy | 候補 | 約定 | 決済 | 勝率 | 平均取引率 | 最大DD | 30万円→ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in STRATEGIES:
        m = summary["results"][s]["discovery"]
        lines.append(
            f"| {s} | {m['candidate_event_count']} | {m['filled_entry_count']} | {m['closed_trade_count']} | "
            f"{pct(m['win_rate'])} | {pct(m['mean_trade_return'])} | {pct(m['max_drawdown'])} | {yen(m['ending_equity_jpy'])} |"
        )
    lines.append("")
    w = summary["winner_analysis"]
    lines.append("## 事前固定ルールによる判定")
    lines.append("")
    lines.append(f"- 期待値最高: **{w['highest_expectancy'] or '該当なし'}**")
    lines.append(f"- 安定性最高: **{w['most_stable'] or '該当なし'}**")
    lines.append(f"- リスク最低: **{w['lowest_risk'] or '該当なし'}**")
    lines.append("- 勝者資格はDiscovery/Validationの両方で決済30件以上。")
    lines.append("")
    lines.append("## 注意")
    lines.append("")
    lines.append("- v2候補イベント自体は過去研究から作られているため、この結果は完全未汚染のOOSではない。Validationは頑健性確認。")
    lines.append("- 現在の監視構成銘柄を過去へ遡って使っているため、構成銘柄/生存者バイアスが残る。")
    lines.append("- PRETEST 1.0の売却は初期ストップ + 終値75MA割れ翌日寄付き。MACD/BB/ローソク足/高値再挑戦は入れていない。")
    lines.append("- 100株固定なので、本の上昇1/2/3回目の資金配分レンジを再現したものではない。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    verify_spec()

    if not EVENTS_PATH.exists():
        stop("FROZEN_V2_EVENT_FILE_MISSING")
    events = pd.read_csv(EVENTS_PATH, dtype={"code": str}, encoding="utf-8-sig")
    needed = {"code", "date", "sma75_slope5_pct", "rs20_vs_topix"}
    if not needed.issubset(events.columns):
        stop("EVENT_COLUMNS_MISSING", {"missing": sorted(needed - set(events.columns))})
    events["code"] = events["code"].astype(str).str.zfill(4)
    events["date"] = pd.to_datetime(events["date"])

    universe = json.loads(Path("universe.json").read_text(encoding="utf-8"))
    codes = [str(x).zfill(4) for x in universe["codes_4digit"]]
    frames, common_latest, split_col_missing = download_frames(codes)

    event_codes = set(events["code"])
    missing_codes = sorted(event_codes - set(frames))
    if missing_codes:
        stop("EVENT_CODES_MISSING_FROM_PRICE_DATA", {"codes": missing_codes})

    common_start = events["date"].min()
    if common_latest < VALIDATION_SIGNAL_START:
        stop("VALIDATION_RANGE_UNAVAILABLE", {"latest": common_latest.date().isoformat()})

    results = {}
    all_trades = []
    all_daily = []
    for strategy in STRATEGIES:
        results[strategy] = {}
        for phase in ["all", "discovery", "validation"]:
            m, t, d = simulate(strategy, phase, events, frames, common_start, common_latest)
            results[strategy][phase] = m
            if len(t):
                all_trades.append(t)
            if len(d):
                all_daily.append(d)

    winners = winner_analysis(results)
    summary = {
        "status": "PASS_REALIZED_BACKTEST_PRETEST_1_0",
        "v1_2_unchanged": True,
        "v1_2_sha256": V1_2_SHA,
        "execution_spec_file": str(SPEC_PATH),
        "execution_spec_sha256": EXPECTED_SPEC_SHA,
        "data_source": "Yahoo Finance via yfinance, auto_adjust=False, actions=True",
        "common_latest_confirmed_date": common_latest.date().isoformat(),
        "frozen_event_file": str(EVENTS_PATH),
        "frozen_event_count": int(len(events)),
        "capital_jpy": CAPITAL,
        "lot_shares": LOT,
        "split_action_column_missing_codes": split_col_missing,
        "results": results,
        "winner_analysis": winners,
        "caveats": [
            "fees/taxes/slippage/dividends excluded",
            "current monitored constituents are applied historically; survivorship/constituent bias remains",
            "v2 event foundation was motivated by prior historical research; validation is not pristine untouched OOS",
            "PRETEST 1.0 is a separate mechanized execution model, not a claim that all added assumptions are book rules",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    trades_df.to_csv(OUT / "v2_realized_trades_pretest_1_0.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT / "v2_realized_daily_equity_pretest_1_0.csv", index=False, encoding="utf-8-sig")
    (OUT / "v2_realized_summary_pretest_1_0.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "v2_realized_report_pretest_1_0.md").write_text(
        build_report(summary), encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "latest": summary["common_latest_confirmed_date"],
        "winner_analysis": winners,
        "validation": {s: results[s]["validation"] for s in STRATEGIES},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
