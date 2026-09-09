#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import backtest_frozen2 as base

SPEC_PATH = Path("profit_tournament_spec_v1.json")
EXPECTED_SHA = "cfcfc923c5f1c2fc3d735b294f34299923fef97b985fbef35556d24bebd627fa"
OUT = Path("research/profit_tournament_v1")
OUT.mkdir(parents=True, exist_ok=True)
CAPITAL0=300000.0
LOT=100
RISK_FRAC=0.01
MIN_TURNOVER=500_000_000.0
MIN_HISTORY=260
BASE_COST=0.001
STRESS_COST=0.002
DEV_START=pd.Timestamp("2020-01-01")
DEV_END=pd.Timestamp("2022-12-30")
HOLD_START=pd.Timestamp("2023-01-01")
HOLD_END=pd.Timestamp("2025-12-30")
FULL_END=HOLD_END

ENTRIES=[
"ma75_reclaim","breakout20_vol15","breakout55","breakout100",
"reversal5_bottom10","reversal10_bottom10","reversal20_bottom10",
"momentum20_top10","momentum60_top10","near52w_high","golden25_75","sma25_reclaim_uptrend"]
EXITS=["fixed_3d_1p5R","trail_2ATR","trail_3ATR","trail_4ATR","sma20_break","sma75_break"]

def verify():
    if not SPEC_PATH.exists(): raise SystemExit("SPEC_MISSING")
    actual=hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    if actual!=EXPECTED_SHA: raise SystemExit(f"SPEC_SHA_MISMATCH {actual}")

def enrich(g):
    g=g.copy()
    cum=g["CumSplit"]
    close=pd.to_numeric(g["Close"],errors="coerce").astype(float)
    high=pd.to_numeric(g["High"],errors="coerce").astype(float)
    vol=pd.to_numeric(g["Volume"],errors="coerce").astype(float)
    sc=close*cum; sh=high*cum
    g["SMA20"]=sc.rolling(20).mean()/cum
    g["High20Prior"]=sh.shift(1).rolling(20).max()/cum
    g["High55Prior"]=sh.shift(1).rolling(55).max()/cum
    g["High100Prior"]=sh.shift(1).rolling(100).max()/cum
    g["High252"]=sc.rolling(252).max()/cum
    g["Vol20PriorMed"]=vol.shift(1).rolling(20).median()
    for n in [5,10,20,60]:
        g[f"Ret{n}"]=sc/sc.shift(n)-1.0
    g["Eligible"]=(g["HistoryRows"]>=MIN_HISTORY)&(g["Turnover20Med"]>=MIN_TURNOVER)&g["ATR14"].notna()
    g["Sig_ma75_reclaim"]=g["Eligible"]&(g["SMA75"]>g["SMA75_5"])&(g["SMA25"]>g["SMA75"])&(close<=g["SMA25"])&(close.shift(1)<=g["SMA75"].shift(1))&(close>g["SMA75"])
    g["Sig_breakout20_vol15"]=g["Eligible"]&(close>g["High20Prior"])&(vol>=1.5*g["Vol20PriorMed"])
    g["Sig_breakout55"]=g["Eligible"]&(close>g["High55Prior"])
    g["Sig_breakout100"]=g["Eligible"]&(close>g["High100Prior"])
    g["Sig_near52w_high"]=g["Eligible"]&(close>=0.98*g["High252"])&(g["Ret20"]>0)
    g["Sig_golden25_75"]=g["Eligible"]&(g["SMA25"].shift(1)<=g["SMA75"].shift(1))&(g["SMA25"]>g["SMA75"])
    g["Sig_sma25_reclaim_uptrend"]=g["Eligible"]&(g["SMA75"]>g["SMA75_5"])&(g["SMA25"]>g["SMA75"])&(close.shift(1)<=g["SMA25"].shift(1))&(close>g["SMA25"])
    return g

def build_signal_maps(frames,codes):
    maps={e:defaultdict(list) for e in ENTRIES}
    noncross=["ma75_reclaim","breakout20_vol15","breakout55","breakout100","near52w_high","golden25_75","sma25_reclaim_uptrend"]
    for code in codes:
        g=frames[code]
        for e in noncross:
            q=g.loc[(g.index>=DEV_START)&(g.index<=HOLD_END)&g[f"Sig_{e}"]]
            for d,r in q.iterrows():
                maps[e][pd.Timestamp(d)].append((code,float(r["Turnover20Med"]),float(r["ATR14"])))
    cal=frames[base.BENCHMARK].index
    cal=cal[(cal>=DEV_START)&(cal<=HOLD_END)]
    cross=[("reversal5_bottom10","Ret5","low"),("reversal10_bottom10","Ret10","low"),("reversal20_bottom10","Ret20","low"),
           ("momentum20_top10","Ret20","high"),("momentum60_top10","Ret60","high")]
    for d in cal:
        rows=[]
        for code in codes:
            g=frames[code]
            if d in g.index and bool(g.at[d,"Eligible"]):
                rows.append((code,float(g.at[d,"Turnover20Med"]),float(g.at[d,"ATR14"]),
                             g.at[d,"Ret5"],g.at[d,"Ret10"],g.at[d,"Ret20"],g.at[d,"Ret60"]))
        if not rows: continue
        df=pd.DataFrame(rows,columns=["code","turn","atr","Ret5","Ret10","Ret20","Ret60"])
        for e,col,side in cross:
            s=pd.to_numeric(df[col],errors="coerce").dropna()
            if len(s)<10: continue
            q=s.quantile(0.10 if side=="low" else 0.90)
            use=df[pd.to_numeric(df[col],errors="coerce")<=q] if side=="low" else df[pd.to_numeric(df[col],errors="coerce")>=q]
            for _,r in use.iterrows():
                maps[e][pd.Timestamp(d)].append((r["code"],float(r["turn"]),float(r["atr"])))
    return maps

def next_date(g,d):
    i=g.index.searchsorted(d,side="right")
    return None if i>=len(g) else pd.Timestamp(g.index[i])

def max_losing_streak(vals):
    best=cur=0
    for x in vals:
        if x<0: cur+=1; best=max(best,cur)
        else: cur=0
    return best

def metrics(trades, daily, ending_equity):
    if not trades:
        return {"n":0,"win_rate":None,"total_pnl_jpy":ending_equity-CAPITAL0,"ending_equity":ending_equity,
                "expectancy_jpy":None,"expectancy_R":None,"profit_factor":None,"max_drawdown":0.0,
                "max_losing_streak":0,"avg_gain_R":None,"avg_loss_R":None,"avg_win_loss_R_ratio":None,
                "avg_holding_sessions":None,"largest_winner_R":None}
    d=pd.DataFrame(trades)
    p=d.pnl_jpy.astype(float); r=d.R_multiple.astype(float)
    w=d[p>0]; l=d[p<0]
    gp=float(w.pnl_jpy.sum()) if len(w) else 0.0
    gl=float(-l.pnl_jpy.sum()) if len(l) else 0.0
    pf=gp/gl if gl>0 else (float("inf") if gp>0 else None)
    agr=float(w.R_multiple.mean()) if len(w) else None
    alr=float(l.R_multiple.mean()) if len(l) else None
    eq=pd.Series([x["equity"] for x in daily],dtype=float)
    dd=float((eq/eq.cummax()-1).min()) if len(eq) else 0.0
    return {"n":int(len(d)),"win_rate":float((p>0).mean()),"total_pnl_jpy":float(ending_equity-CAPITAL0),
            "ending_equity":float(ending_equity),"expectancy_jpy":float(p.mean()),"expectancy_R":float(r.mean()),
            "profit_factor":float(pf) if pf is not None and np.isfinite(pf) else ("INF" if pf==float("inf") else None),
            "max_drawdown":dd,"max_losing_streak":int(max_losing_streak(p.tolist())),
            "avg_gain_R":agr,"avg_loss_R":alr,
            "avg_win_loss_R_ratio":(agr/abs(alr)) if agr is not None and alr not in (None,0) else None,
            "avg_holding_sessions":float(d.holding_sessions.mean()),"largest_winner_R":float(r.max())}

def simulate(frames,codes,maps,entry_name,exit_name,start,end,rt_cost):
    one=rt_cost/2.0
    cal=frames[base.BENCHMARK].index
    cal=cal[(cal>=start)&(cal<=end)]
    cash=CAPITAL0; pos=None; scheduled=None; trades=[]; daily=[]
    for day in cal:
        if pos is not None:
            g=frames[pos["code"]]
            if day in g.index:
                ratio=float(g.at[day,"Stock Splits"]) if pd.notna(g.at[day,"Stock Splits"]) else 0.0
                if ratio>0 and abs(ratio-1.0)>1e-12 and day>pos["entry_date"]:
                    pos["shares"]*=ratio; pos["stop"]/=ratio; pos["high_water"]/=ratio
                    pos["entry_price"]/=ratio
                    if pos.get("target") is not None: pos["target"]/=ratio
        if pos is not None and pos.get("exit_next_open"):
            g=frames[pos["code"]]
            if day in g.index:
                px=float(g.at[day,"Open"]); proceeds=pos["shares"]*px*(1-one)
                pnl=proceeds-pos["entry_cash_out"]; cash+=proceeds
                trades.append({"pnl_jpy":pnl,"R_multiple":pnl/pos["initial_risk"],"holding_sessions":pos["held"],"entry_date":pos["entry_date"],"exit_date":day})
                pos=None
        if pos is not None:
            g=frames[pos["code"]]
            if day in g.index:
                op=float(g.at[day,"Open"]); hi=float(g.at[day,"High"]); lo=float(g.at[day,"Low"]); cl=float(g.at[day,"Close"])
                pos["held"]+=1
                exit_px=None
                if op<=pos["stop"]: exit_px=op
                elif lo<=pos["stop"]: exit_px=pos["stop"]
                elif exit_name=="fixed_3d_1p5R" and hi>=pos["target"]: exit_px=pos["target"]
                elif exit_name=="fixed_3d_1p5R" and pos["held"]>=3: exit_px=cl
                if exit_px is not None:
                    proceeds=pos["shares"]*exit_px*(1-one); pnl=proceeds-pos["entry_cash_out"]; cash+=proceeds
                    trades.append({"pnl_jpy":pnl,"R_multiple":pnl/pos["initial_risk"],"holding_sessions":pos["held"],"entry_date":pos["entry_date"],"exit_date":day})
                    pos=None
                else:
                    pos["high_water"]=max(pos["high_water"],hi)
                    if exit_name.startswith("trail_"):
                        mult=float(exit_name.split("_")[1].replace("ATR",""))
                        atr=float(g.at[day,"ATR14"]) if pd.notna(g.at[day,"ATR14"]) else np.nan
                        if np.isfinite(atr): pos["stop"]=max(pos["stop"],pos["high_water"]-mult*atr)
                    elif exit_name=="sma20_break" and pd.notna(g.at[day,"SMA20"]) and cl<float(g.at[day,"SMA20"]):
                        pos["exit_next_open"]=True
                    elif exit_name=="sma75_break" and pd.notna(g.at[day,"SMA75"]) and cl<float(g.at[day,"SMA75"]):
                        pos["exit_next_open"]=True
        if pos is None and scheduled is not None and scheduled["entry_date"]==day:
            code=scheduled["code"]; g=frames[code]
            if day in g.index:
                entry=float(g.at[day,"Open"]); atr=scheduled["atr"]; stop=entry-atr; risk_per=max(entry-stop,0)
                risk_budget=cash*RISK_FRAC
                lots_risk=math.floor(risk_budget/(risk_per*LOT)) if risk_per>0 else 0
                lots_cash=math.floor(cash/(entry*LOT*(1+one)))
                lots=min(lots_risk,lots_cash)
                if lots>=1:
                    shares=lots*LOT; notional=shares*entry; cashout=notional*(1+one); cash-=cashout
                    initial_risk=shares*risk_per
                    pos={"code":code,"entry_date":day,"entry_price":entry,"shares":float(shares),"stop":stop,
                         "initial_risk":initial_risk,"entry_cash_out":cashout,"high_water":float(g.at[day,"High"]),
                         "held":1,"exit_next_open":False,"target":entry+1.5*risk_per if exit_name=="fixed_3d_1p5R" else None}
                    lo=float(g.at[day,"Low"]); hi=float(g.at[day,"High"]); cl=float(g.at[day,"Close"]); ex=None
                    if lo<=stop: ex=stop
                    elif exit_name=="fixed_3d_1p5R" and hi>=pos["target"]: ex=pos["target"]
                    if ex is not None:
                        proceeds=shares*ex*(1-one); pnl=proceeds-cashout; cash+=proceeds
                        trades.append({"pnl_jpy":pnl,"R_multiple":pnl/initial_risk,"holding_sessions":1,"entry_date":day,"exit_date":day}); pos=None
                    elif exit_name.startswith("trail_"):
                        mult=float(exit_name.split("_")[1].replace("ATR","")); atr_now=float(g.at[day,"ATR14"]) if pd.notna(g.at[day,"ATR14"]) else atr
                        pos["stop"]=max(pos["stop"],pos["high_water"]-mult*atr_now)
                    elif exit_name=="sma20_break" and pd.notna(g.at[day,"SMA20"]) and cl<float(g.at[day,"SMA20"]): pos["exit_next_open"]=True
                    elif exit_name=="sma75_break" and pd.notna(g.at[day,"SMA75"]) and cl<float(g.at[day,"SMA75"]): pos["exit_next_open"]=True
            scheduled=None
        if pos is None and scheduled is None:
            cands=maps[entry_name].get(pd.Timestamp(day),[])
            if cands:
                code,turn,atr=max(cands,key=lambda x:x[1]); nd=next_date(frames[code],day)
                if nd is not None and nd<=end: scheduled={"code":code,"entry_date":nd,"atr":atr}
        eq=cash
        if pos is not None and day in frames[pos["code"]].index: eq+=pos["shares"]*float(frames[pos["code"]].at[day,"Close"])
        daily.append({"date":day,"equity":float(eq)})
    if pos is not None:
        g=frames[pos["code"]]; valid=g[g.index<=end]
        if len(valid):
            d=valid.index[-1]; px=float(valid.iloc[-1]["Close"]); proceeds=pos["shares"]*px*(1-one); pnl=proceeds-pos["entry_cash_out"]; cash+=proceeds
            trades.append({"pnl_jpy":pnl,"R_multiple":pnl/pos["initial_risk"],"holding_sessions":pos["held"],"entry_date":pos["entry_date"],"exit_date":d})
    return metrics(trades,daily,float(cash)), pd.DataFrame(trades)

def main():
    verify()
    uni=json.loads(Path("universe.json").read_text(encoding="utf-8")); codes=[str(x).zfill(4) for x in uni["codes_4digit"]]
    frames,usable,issues=base.download_all(codes)
    for c in list(frames): frames[c]=enrich(frames[c])
    maps=build_signal_maps(frames,usable)
    disc=[]
    for e in ENTRIES:
        for x in EXITS:
            m,_=simulate(frames,usable,maps,e,x,DEV_START,DEV_END,BASE_COST); disc.append({"entry":e,"exit":x,**m})
    discdf=pd.DataFrame(disc); discdf.to_csv(OUT/"discovery_all_72.csv",index=False,encoding="utf-8-sig")
    champs=[]
    for e in ENTRIES:
        d=discdf[discdf.entry==e].copy(); eligible=d[d.n>=10]; pool=eligible if len(eligible) else d
        row=pool.sort_values(["ending_equity","profit_factor"],ascending=[False,False]).iloc[0].to_dict(); row["low_n_selection"]=len(eligible)==0; champs.append(row)
    champdf=pd.DataFrame(champs); champdf.to_csv(OUT/"discovery_family_champions.csv",index=False,encoding="utf-8-sig")
    hold=[]
    for _,r in champdf.iterrows():
        e=r["entry"]; x=r["exit"]; m,_=simulate(frames,usable,maps,e,x,HOLD_START,HOLD_END,BASE_COST); ms,_=simulate(frames,usable,maps,e,x,HOLD_START,HOLD_END,STRESS_COST)
        hold.append({"entry":e,"exit":x,**m,"stress_ending_equity":ms["ending_equity"],"stress_total_pnl_jpy":ms["total_pnl_jpy"],"stress_expectancy_R":ms["expectancy_R"],"stress_profit_factor":ms["profit_factor"]})
    holddf=pd.DataFrame(hold).sort_values("ending_equity",ascending=False); holddf.to_csv(OUT/"holdout_family_champions.csv",index=False,encoding="utf-8-sig")
    full=[]
    for e in ENTRIES:
        for x in EXITS:
            m,_=simulate(frames,usable,maps,e,x,DEV_START,HOLD_END,BASE_COST); full.append({"entry":e,"exit":x,**m})
    fulldf=pd.DataFrame(full).sort_values("ending_equity",ascending=False); fulldf.to_csv(OUT/"fullsample_all_72_oracle.csv",index=False,encoding="utf-8-sig")
    winner=holddf.iloc[0].to_dict(); top5=holddf.head(5).to_dict(orient="records"); oracle=fulldf.iloc[0].to_dict()
    summary={"status":"PASS_PROFIT_TOURNAMENT_V1","spec_sha256":EXPECTED_SHA,"usable_universe":len(usable),"issues":issues,"tested_configurations":len(ENTRIES)*len(EXITS),"selection":"discovery selects exit per family; holdout selects family winner","winner_holdout":winner,"top5_holdout":top5,"fullsample_oracle_not_for_selection":oracle,"limitations":["current TOPIX100 applied historically","Yahoo/yfinance rather than J-Quants","daily-bar execution; no ORB/PEAD because required intraday/fundamental history is unavailable in this run"]}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    lines=["# Profit Tournament V1","",f"- Tested: **{len(ENTRIES)*len(EXITS)}** daily-rule configurations","- Discovery: 2020-2022 selects one exit per entry family","- Holdout: 2023-2025 ranks the 12 family champions","- Capital: ¥300,000; risk: 1% current equity; one position; 100-share lots; cost 0.10% base / 0.20% stress","","## Holdout winner",f"- Entry: **{winner['entry']}**",f"- Exit: **{winner['exit']}**",f"- Trades: **{int(winner['n'])}**",f"- Ending equity: **¥{winner['ending_equity']:.0f}**",f"- Net P/L: **¥{winner['total_pnl_jpy']:.0f}**",f"- Win rate: **{winner['win_rate']*100:.2f}%**" if winner["win_rate"] is not None else "- Win rate: N/A",f"- Expectancy: **{winner['expectancy_R']:.3f}R/trade**" if winner["expectancy_R"] is not None else "- Expectancy: N/A",f"- PF: **{winner['profit_factor']}**",f"- Max DD: **{winner['max_drawdown']*100:.2f}%**",f"- Stress ending equity: **¥{winner['stress_ending_equity']:.0f}**","","## Top 5 holdout"]
    for i,r in enumerate(top5,1): lines.append(f"{i}. {r['entry']} + {r['exit']}: ¥{r['ending_equity']:.0f} (P/L ¥{r['total_pnl_jpy']:.0f}, n={int(r['n'])}, PF={r['profit_factor']})")
    lines += ["","## Descriptive full-sample oracle (NOT selection winner)",f"- {oracle['entry']} + {oracle['exit']}: ¥{oracle['ending_equity']:.0f} (P/L ¥{oracle['total_pnl_jpy']:.0f})","","## Limitations","- Current 2026 TOPIX100 constituents are applied historically; survivorship/constituent bias remains.","- Yahoo/yfinance daily data is used; this is not the official J-Quants all-TSE test.","- ORB and PEAD are not included in this daily-price tournament because the required six-year intraday/fundamental event dataset is unavailable here."]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__": main()
