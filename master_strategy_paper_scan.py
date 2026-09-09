#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from datetime import datetime, timezone
from pathlib import Path

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
RT_COST = 0.001
ONE_WAY_COST = RT_COST / 2

# PROVISIONAL PAPER-TRADING ONLY.
# Formal project data source remains J-Quants; this temporary live-forward scanner
# uses Yahoo Finance/yfinance explicitly and must not be treated as a real-money signal service.


def default_state():
    return {'position': None, 'pending_entry': None, 'pending_exit': None,
            'completed_trades': 0, 'paper_equity': CAPITAL}


def load_state():
    s = default_state()
    if STATE.exists():
        s.update(json.loads(STATE.read_text(encoding='utf-8')))
    return s


def save_state(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')


def future_split_factor(splits: pd.Series) -> pd.Series:
    sf = pd.to_numeric(splits, errors='coerce').fillna(0.0).astype(float)
    sf = sf.where(sf > 0, 1.0)
    inclusive = sf.iloc[::-1].cumprod().iloc[::-1]
    return inclusive / sf


def extract(big, ticker):
    if not isinstance(big.columns, pd.MultiIndex): return big.copy()
    if ticker in big.columns.get_level_values(0): return big[ticker].copy()
    if ticker in big.columns.get_level_values(1): return big.xs(ticker, axis=1, level=1).copy()
    return pd.DataFrame()


def indicators(g):
    g = g.copy().dropna(subset=['Open','High','Low','Close','Volume']).sort_index()
    splits = pd.to_numeric(g.get('Stock Splits', 0.0), errors='coerce').fillna(0.0)
    fac = future_split_factor(splits)
    for c in ['Open','High','Low','Close']:
        g[c] = pd.to_numeric(g[c], errors='coerce').astype(float) * fac
    sm = splits.where(splits > 0, 1.0)
    cum = sm.cumprod()
    close, high, low = g['Close'].astype(float), g['High'].astype(float), g['Low'].astype(float)
    sc, sh, sl = close*cum, high*cum, low*cum
    g['SMA20'] = sc.rolling(20).mean()/cum
    g['SMA25'] = sc.rolling(25).mean()/cum
    g['SMA75'] = sc.rolling(75).mean()/cum
    prev = sc.shift(1)
    tr = pd.concat([(sh-sl),(sh-prev).abs(),(sl-prev).abs()],axis=1).max(axis=1)
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


def close_trade(state, pos, date, exit_price, reason):
    shares = int(pos['shares']); entry = float(pos['entry_price']); exit_price = float(exit_price)
    gross = shares * (exit_price - entry)
    fees = shares * entry * ONE_WAY_COST + shares * exit_price * ONE_WAY_COST
    pnl = gross - fees
    init_risk = float(pos['initial_risk_jpy'])
    r = pnl / init_risk if init_risk else None
    state['paper_equity'] = float(state['paper_equity']) + pnl
    state['completed_trades'] = int(state.get('completed_trades', 0)) + 1
    append_log({'code':pos['code'],'signal_date':pos['signal_date'],'entry_date':pos['entry_date'],
                'exit_date':date,'entry_price':entry,'exit_price':exit_price,'shares':shares,
                'initial_stop':pos['initial_stop'],'initial_risk_jpy':init_risk,
                'pnl_jpy':pnl,'R_multiple':r,'exit_reason':reason,
                'paper_equity_after':state['paper_equity']})
    state['position'] = None; state['pending_exit'] = None
    return pnl, r


def scan_candidates(frames):
    out=[]
    for c,g in frames.items():
        if len(g)<2: continue
        a,b=g.iloc[-2],g.iloc[-1]
        vals=[a['SMA25'],a['SMA75'],b['SMA25'],b['SMA75'],b['ATR14'],b['Turnover20Med']]
        if any(pd.isna(v) for v in vals): continue
        if float(b['Turnover20Med'])<MIN_TURNOVER20: continue
        if float(a['SMA25'])<=float(a['SMA75']) and float(b['SMA25'])>float(b['SMA75']):
            out.append({'code':c,'signal_date':g.index[-1].date().isoformat(),
                        'sma25':float(b['SMA25']),'sma75':float(b['SMA75']),
                        'atr14':float(b['ATR14']),'turnover20_median':float(b['Turnover20Med'])})
    out.sort(key=lambda x:x['turnover20_median'], reverse=True)
    return out


def main():
    codes=json.loads(UNIVERSE.read_text(encoding='utf-8'))['codes_4digit']
    tickers=[f'{c}.T' for c in codes]
    raw=yf.download(tickers=tickers,period='1y',interval='1d',group_by='ticker',auto_adjust=False,
                    actions=True,repair=False,threads=True,progress=False,timeout=90)
    if raw is None or raw.empty: raise SystemExit('DATA STOP: yfinance returned no data')
    frames={}
    for c,t in zip(codes,tickers):
        g=extract(raw,t)
        if g.empty or any(x not in g.columns for x in ['Open','High','Low','Close','Volume']): continue
        if 'Stock Splits' not in g.columns: g['Stock Splits']=0.0
        g.index=pd.to_datetime(g.index).tz_localize(None)
        g=indicators(g)
        if len(g)>=MIN_HISTORY: frames[c]=g
    if not frames: raise SystemExit('DATA STOP: no usable frames')

    state=load_state(); latest_date=max(g.index[-1] for g in frames.values())
    today=latest_date.date().isoformat()
    latest={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'as_of':today,
            'mode':'PROVISIONAL_PAPER_ONLY','source':'Yahoo Finance/yfinance',
            'strategy':'25/75 golden cross + initial 1ATR stop + exit next open after close < SMA20',
            'paper_equity':float(state['paper_equity']),'completed_trades':int(state['completed_trades']),
            'action':'NONE'}

    # 1) Execute a previously scheduled SMA20 exit at today's open.
    if state.get('position') and state.get('pending_exit'):
        pe=state['pending_exit']; pos=state['position']; g=frames.get(pos['code'])
        if g is not None and pd.Timestamp(pe['signal_date']) < latest_date and latest_date in g.index:
            px=float(g.at[latest_date,'Open'])
            pnl,r=close_trade(state,pos,today,px,'SMA20_BREAK_NEXT_OPEN')
            latest.update({'action':'PAPER_EXITED','code':pos['code'],'exit_price':px,'pnl_jpy':pnl,'R_multiple':r})

    # 2) Execute a previously scheduled entry at today's open, then apply initial hard stop.
    if state.get('position') is None and state.get('pending_entry'):
        pe=state['pending_entry']; g=frames.get(pe['code'])
        if g is not None and pd.Timestamp(pe['signal_date']) < latest_date and latest_date in g.index:
            op=float(g.at[latest_date,'Open']); atr=float(pe['atr14']); equity=float(state['paper_equity'])
            budget=equity*RISK_FRACTION
            risk_qty=math.floor((budget/atr)/LOT)*LOT if atr>0 else 0
            cash_qty=math.floor((equity/op)/LOT)*LOT if op>0 else 0
            shares=int(min(risk_qty,cash_qty))
            if shares < LOT:
                latest.update({'action':'PAPER_ENTRY_SKIPPED','code':pe['code'],'open':op,
                               'reason':'100 shares exceed risk budget or available paper capital'})
                state['pending_entry']=None
            else:
                stop=op-atr; init_risk=shares*atr
                pos={'code':pe['code'],'signal_date':pe['signal_date'],'entry_date':today,
                     'entry_price':op,'shares':shares,'initial_stop':stop,'initial_risk_jpy':init_risk}
                state['pending_entry']=None; state['position']=pos
                low=float(g.at[latest_date,'Low'])
                if low <= stop:
                    pnl,r=close_trade(state,pos,today,stop,'INITIAL_STOP_INTRADAY')
                    latest.update({'action':'PAPER_ENTERED_AND_STOPPED','code':pe['code'],'entry_price':op,
                                   'shares':shares,'initial_stop':stop,'pnl_jpy':pnl,'R_multiple':r})
                else:
                    latest.update({'action':'PAPER_ENTERED','code':pe['code'],'entry_price':op,
                                   'shares':shares,'initial_stop':stop,'initial_risk_jpy':init_risk})

    # 3) Manage any surviving position with hard stop first, then SMA20 close-break scheduling.
    if state.get('position'):
        pos=state['position']; g=frames.get(pos['code'])
        if g is not None and latest_date in g.index and today != pos['entry_date']:
            op=float(g.at[latest_date,'Open']); low=float(g.at[latest_date,'Low']); stop=float(pos['initial_stop'])
            if op <= stop:
                pnl,r=close_trade(state,pos,today,op,'INITIAL_STOP_GAP')
                latest.update({'action':'PAPER_STOPPED','code':pos['code'],'exit_price':op,'pnl_jpy':pnl,'R_multiple':r})
            elif low <= stop:
                pnl,r=close_trade(state,pos,today,stop,'INITIAL_STOP_INTRADAY')
                latest.update({'action':'PAPER_STOPPED','code':pos['code'],'exit_price':stop,'pnl_jpy':pnl,'R_multiple':r})

    if state.get('position'):
        pos=state['position']; g=frames[pos['code']]
        close=float(g.at[latest_date,'Close']); sma20=float(g.at[latest_date,'SMA20'])
        latest.update({'position':pos,'close':close,'sma20':sma20})
        if close < sma20:
            state['pending_exit']={'signal_date':today}
            latest['action']='EXIT_NEXT_OPEN'; latest['reason']='Close below SMA20'
        elif latest['action']=='NONE':
            latest['action']='HOLD'; latest['reason']='Close remains at/above SMA20'
    else:
        state['pending_exit']=None
        # 4) Only after all prior actions, scan today's close for tomorrow's fresh entry.
        if state.get('pending_entry') is None:
            candidates=scan_candidates(frames); latest['candidates']=candidates
            if candidates:
                pick=candidates[0]; state['pending_entry']=pick
                latest.update({'action':'ENTER_NEXT_OPEN_IF_AFFORDABLE','selected':pick,
                               'risk_budget_jpy':float(state['paper_equity'])*RISK_FRACTION,
                               'sizing_rule':'Tomorrow open: stop=open-ATR14(signal close); size in 100-share lots so initial risk <=1% equity and notional <= equity.'})
            elif latest['action']=='NONE':
                latest['action']='NO_SIGNAL'; latest['reason']='No fresh eligible 25/75 golden cross'
        else:
            latest['pending_entry']=state['pending_entry']

    latest['paper_equity']=float(state['paper_equity']); latest['completed_trades']=int(state['completed_trades'])
    save_state(state); LATEST.write_text(json.dumps(latest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(latest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
