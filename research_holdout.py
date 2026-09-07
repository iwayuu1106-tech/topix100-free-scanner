#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HORIZONS = [5,10,20,40]
MIN_N40 = 30
OUT = Path('research')


def metric_block(df, h):
    s = pd.to_numeric(df[f'ret_{h}d'], errors='coerce').dropna()
    if len(s) == 0:
        return {'n': 0}
    pos = s[s > 0]
    neg = s[s < 0]
    return {
        'n': int(len(s)),
        'win_rate': float((s > 0).mean()),
        'mean_return': float(s.mean()),
        'median_return': float(s.median()),
        'avg_gain': float(pos.mean()) if len(pos) else None,
        'avg_loss': float(neg.mean()) if len(neg) else None,
    }


def summarize(df):
    return {f'h{h}': metric_block(df,h) for h in HORIZONS}


def score(group, baseline):
    pos_ret = pos_win = 0
    val = 0.0
    deltas = {}
    for h in HORIZONS:
        g = group[f'h{h}']; b = baseline[f'h{h}']
        if not g.get('n') or not b.get('n'):
            continue
        dr = g['mean_return'] - b['mean_return']
        dw = g['win_rate'] - b['win_rate']
        deltas[f'{h}d_mean_return_delta'] = dr
        deltas[f'{h}d_win_rate_delta'] = dw
        pos_ret += dr > 0
        pos_win += dw > 0
        val += dr * (1 if h in [20,40] else .5) + dw * .15
    return float(val), int(pos_ret), int(pos_win), deltas


def masks(ev):
    return {
        'SMA75_RISING_TRUE': ev['sma75_rising'].astype(str).str.lower().eq('true'),
        'SMA75_RISING_FALSE': ev['sma75_rising'].astype(str).str.lower().eq('false'),
        'MACD_ABOVE_SIGNAL_TRUE': ev['macd_above_signal'].astype(str).str.lower().eq('true'),
        'MACD_ABOVE_SIGNAL_FALSE': ev['macd_above_signal'].astype(str).str.lower().eq('false'),
        'MACD_ABOVE_ZERO_TRUE': ev['macd_above_zero'].astype(str).str.lower().eq('true'),
        'MACD_ABOVE_ZERO_FALSE': ev['macd_above_zero'].astype(str).str.lower().eq('false'),
        'PRICE_ABOVE_SMA25_TRUE': ev['price_above_sma25'].astype(str).str.lower().eq('true'),
        'PRICE_ABOVE_SMA25_FALSE': ev['price_above_sma25'].astype(str).str.lower().eq('false'),
        'SMA25_ABOVE_SMA75_TRUE': ev['sma25_above_sma75'].astype(str).str.lower().eq('true'),
        'SMA25_ABOVE_SMA75_FALSE': ev['sma25_above_sma75'].astype(str).str.lower().eq('false'),
        'VOL_LT_1.0X': ev['volume_ratio20'] < 1.0,
        'VOL_1.0_TO_1.5X': (ev['volume_ratio20'] >= 1.0) & (ev['volume_ratio20'] < 1.5),
        'VOL_GE_1.5X': ev['volume_ratio20'] >= 1.5,
        'SMA75_RISING_AND_MACD_ABOVE_SIGNAL': ev['sma75_rising'].astype(str).str.lower().eq('true') & ev['macd_above_signal'].astype(str).str.lower().eq('true'),
        'SMA75_RISING_AND_PRICE_ABOVE_SMA25': ev['sma75_rising'].astype(str).str.lower().eq('true') & ev['price_above_sma25'].astype(str).str.lower().eq('true'),
        'SMA75_RISING_AND_SMA25_ABOVE_SMA75': ev['sma75_rising'].astype(str).str.lower().eq('true') & ev['sma25_above_sma75'].astype(str).str.lower().eq('true'),
        'SMA75_RISING_AND_VOL_GE_1.0X': ev['sma75_rising'].astype(str).str.lower().eq('true') & (ev['volume_ratio20'] >= 1.0),
        'SMA75_RISING_AND_MACD_ABOVE_SIGNAL_AND_VOL_GE_1.0X': ev['sma75_rising'].astype(str).str.lower().eq('true') & ev['macd_above_signal'].astype(str).str.lower().eq('true') & (ev['volume_ratio20'] >= 1.0),
    }


def main():
    p = OUT / 'events_strict_3yplus.csv'
    ev = pd.read_csv(p, encoding='utf-8-sig')
    ev['date'] = pd.to_datetime(ev['date'])
    start = ev['date'].min()
    split = start + pd.DateOffset(years=3)
    discovery_end = split - pd.Timedelta(days=60)
    discovery = ev[ev['date'] <= discovery_end].copy()
    validation = ev[ev['date'] > split].copy()
    if discovery.empty or validation.empty:
        raise SystemExit('insufficient temporal split')

    db = summarize(discovery)
    vb = summarize(validation)
    dm = masks(discovery)
    vm = masks(validation)
    discovery_rank = []
    for name, mask in dm.items():
        g = summarize(discovery[mask])
        sc, pr, pw, dels = score(g, db)
        discovery_rank.append({
            'filter': name,
            'discovery_event_count': int(mask.sum()),
            'discovery_n40': g['h40'].get('n',0),
            'discovery_score': sc,
            'discovery_positive_return_horizons': pr,
            'discovery_positive_winrate_horizons': pw,
            'discovery_deltas': dels,
        })
    discovery_rank.sort(key=lambda x: x['discovery_score'], reverse=True)
    selected = [x for x in discovery_rank if x['discovery_n40'] >= MIN_N40 and x['discovery_positive_return_horizons'] >= 3 and x['discovery_positive_winrate_horizons'] >= 3][:8]

    validated = []
    for x in selected:
        name = x['filter']
        g = summarize(validation[vm[name]])
        sc, pr, pw, dels = score(g, vb)
        d20r = dels.get('20d_mean_return_delta', -999)
        d20w = dels.get('20d_win_rate_delta', -999)
        d40r = dels.get('40d_mean_return_delta', -999)
        d40w = dels.get('40d_win_rate_delta', -999)
        survives = g['h40'].get('n',0) >= MIN_N40 and d20r > 0 and d20w > 0 and d40r > 0 and d40w > 0
        validated.append({
            **x,
            'validation_event_count': int(vm[name].sum()),
            'validation_n40': g['h40'].get('n',0),
            'validation_score': sc,
            'validation_positive_return_horizons': pr,
            'validation_positive_winrate_horizons': pw,
            'validation_deltas': dels,
            'survives_holdout_20d_40d': bool(survives),
            'validation_metrics': g,
        })

    validated.sort(key=lambda x: (not x['survives_holdout_20d_40d'], -x['validation_score']))
    out = {
        'status': 'PASS',
        'method': 'Temporal holdout. Filters are ranked on discovery only; validation period is used only after selection.',
        'discovery_start': discovery['date'].min().date().isoformat(),
        'discovery_end': discovery['date'].max().date().isoformat(),
        'gap_to_prevent_40d_horizon_bleed': '60 calendar days before split',
        'validation_start': validation['date'].min().date().isoformat(),
        'validation_end': validation['date'].max().date().isoformat(),
        'discovery_event_count': int(len(discovery)),
        'validation_event_count': int(len(validation)),
        'discovery_baseline': db,
        'validation_baseline': vb,
        'selected_from_discovery': selected,
        'holdout_results': validated,
        'survivors': [x for x in validated if x['survives_holdout_20d_40d']],
        'warning': 'This reduces but does not eliminate data-mining and survivorship bias. v1.2 remains unchanged; any new rule must be frozen as v2 before future use.'
    }
    (OUT / 'holdout_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for x in validated:
        row = {k:v for k,v in x.items() if k not in ['discovery_deltas','validation_deltas','validation_metrics']}
        row.update({f'discovery_{k}':v for k,v in x['discovery_deltas'].items()})
        row.update({f'validation_{k}':v for k,v in x['validation_deltas'].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'holdout_candidates.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'status':'PASS','selected':len(selected),'survivors':[x['filter'] for x in out['survivors']]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
