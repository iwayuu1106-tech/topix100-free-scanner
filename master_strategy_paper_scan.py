#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = Path('universe.json')
OUT = Path('paper_master_strategy')
OUT.mkdir(exist_ok=True)
LATEST = OUT / 'latest.json'
STATE = OUT / 'state.json'
LOG = OUT / 'trade_log.csv'

CAPITAL = 300_000.0
RISK_FRACTION = 0.01
LOT = 100
MIN_HISTORY = 120
MIN_TURNOVER20 = 500_000_000.0

# IMPORTANT: This scanner is PROVISIONAL PAPER-TRADING ONLY.
# It uses Yahoo Finance/yfinance because J-Quants is not connected in this repo/session.
# Do not treat this as the formal J-Quants backtest or as authorization for real-money trading.


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'position': None, 'completed_trades': 0, 'paper_equity': CAPITAL}


def save_state(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')


def future_split_factor(splits: pd.Series) -> pd.Series:
    sf = pd.to_numeric(splits, errors='coerce').fillna(0.0).astype(float)
    sf = sf.where(sf > 0, 1.0)
    inclusive = sf.iloc[::-1].cumprod().iloc[::-1]
    return inclusive / sf


def extract(big, ticker):
    if not isinstance(big.columns, pd.MultiIndex):
        return big.copy()
    if ticker in big.columns.get_level_values(0):
        return big[ticker].copy()
    if ticker in big.columns.get_level_values(1):
        return big.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy().dropna(subset=['Open','High','Low','Close','Volume']).sort_index()
    splits = pd.to_numeric(g.get('Stock Splits', 0.0), errors='coerce').fillna(0.0)
    fac = future_split_factor(splits)
    # reconstruct raw historical yen/share around future splits
    for c in ['Open','High','Low','Close']:
        g[c] = pd.to_numeric(g[c], errors='coerce').astype(float) * fac
    split_mult = splits.where(splits > 0, 1.0)
    cum = split_mult.cumprod()
    close = g['Close'].astype(float)
    high = g['High'].astype(float)
    low = g['Low'].astype(float)
    sc, sh, sl = close*cum, high*cum, low*cum
    g['SMA20'] = sc.rolling(20).mean()/cum
    g['SMA25'] = sc.rolling(25).mean()/cum
    g['SMA75'] = sc.rolling(75).mean()/cum
    prev_sc = sc.shift(1)
    tr = pd.concat([(sh-sl),(sh-prev_sc).abs(),(sl-prev_sc).abs()],axis=1).max(axis=1)
    g['ATR14'] = tr.rolling(14).mean()/cum
    g['Turnover'] = close * pd.to_numeric(g['Volume'], errors='coerce').astype(float)
    g['Turnover20Med'] = g['Turnover'].rolling(20).median()
    return g


def append_log(row):
    exists = LOG.exists()
    with LOG.open('a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)


def main():
    u = json.loads(UNIVERSE.read_text(encoding='utf-8'))
    codes = u['codes_4digit']
    tickers = [f'{c}.T' for c in codes]
    raw = yf.download(tickers=tickers, period='1y', interval='1d', group_by='ticker', auto_adjust=False,
                      actions=True, repair=False, threads=True, progress=False, timeout=90)
    if raw is None or raw.empty:
        raise SystemExit('DATA STOP: yfinance returned no data')

    frames = {}
    for c,t in zip(codes,tickers):
        g = extract(raw,t)
        if g.empty or any(x not in g.columns for x in ['Open','High','Low','Close','Volume']):
            continue
        if 'Stock Splits' not in g.columns: g['Stock Splits'] = 0.0
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g = indicators(g)
        if len(g) >= MIN_HISTORY:
            frames[c] = g

    state = load_state()
    pos = state.get('position')
    now = datetime.now(timezone.utc).isoformat()
    latest = {'generated_at_utc': now, 'mode':'PROVISIONAL_PAPER_ONLY', 'source':'Yahoo Finance/yfinance',
              'strategy':'25/75 golden cross + initial 1ATR risk + exit next open after close < SMA20',
              'paper_equity': state.get('paper_equity', CAPITAL), 'position': pos, 'action':'NONE'}

    # Manage an existing paper position first.
    if pos:
        c = pos['code']
        g = frames.get(c)
        if g is not None and len(g):
            d = g.index[-1]
            row = g.iloc[-1]
            close = float(row['Close']); sma20 = float(row['SMA20'])
            latest.update({'as_of':d.date().isoformat(),'held_code':c,'close':close,'sma20':sma20})
            if close < sma20:
                latest['action'] = 'EXIT_NEXT_OPEN'
                latest['reason'] = 'Close below SMA20'
            else:
                latest['action'] = 'HOLD'
                latest['reason'] = 'Close remains at/above SMA20'
        LATEST.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(latest, ensure_ascii=False, indent=2)); return

    # No position: scan for a fresh golden cross today.
    candidates = []
    for c,g in frames.items():
        if len(g) < 2: continue
        r0, r1 = g.iloc[-2], g.iloc[-1]
        vals = [r0['SMA25'],r0['SMA75'],r1['SMA25'],r1['SMA75'],r1['ATR14'],r1['Turnover20Med']]
        if any(pd.isna(v) for v in vals): continue
        if float(r1['Turnover20Med']) < MIN_TURNOVER20: continue
        cross = float(r0['SMA25']) <= float(r0['SMA75']) and float(r1['SMA25']) > float(r1['SMA75'])
        if not cross: continue
        candidates.append({'code':c,'signal_date':g.index[-1].date().isoformat(),
                           'sma25':float(r1['SMA25']),'sma75':float(r1['SMA75']),
                           'atr14':float(r1['ATR14']),'turnover20_median':float(r1['Turnover20Med'])})

    candidates.sort(key=lambda x:x['turnover20_median'], reverse=True)
    latest['candidates'] = candidates
    if not candidates:
        latest['action'] = 'NO_SIGNAL'
        latest['reason'] = 'No fresh 25/75 golden cross among eligible monitored names'
    else:
        pick = candidates[0]
        risk_budget = float(state.get('paper_equity', CAPITAL)) * RISK_FRACTION
        # Exact entry/stop/shares are finalized after next-session open is known.
        latest['action'] = 'ENTER_NEXT_OPEN_IF_AFFORDABLE'
        latest['selected'] = pick
        latest['risk_budget_jpy'] = risk_budget
        latest['sizing_rule'] = 'At next open: stop = open - ATR14(signal close); choose max 100-share lots affordable and with initial stop risk <= 1% equity; skip if even 100 shares exceed risk/cash.'

    LATEST.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    save_state(state)
    print(json.dumps(latest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
