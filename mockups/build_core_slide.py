"""Build mockups/us_gauge_slide.html from mockups/core_slide_data.json (written by research/us_gauge/core_slide_data.py).

Rule: every day, if the target differs from the current weight by 10 points or more (or we go to / leave cash),
we trade the next day, without waiting. The Wednesday committee reviews. All numbers come from
research/us_gauge/RESULTS_core.md (row 6).
"""
import json
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
GREEN, RED, AMBER, TRACK = '#16A34A', '#DC2626', '#D97706', '#E5E9F0'


def minus(s: str) -> str:
    return s.replace('-', '−')


def bar(value, lo, hi, color, mark, mark_label, w=440):
    X = lambda v: 6 + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - 12)
    return (f'<svg width="{w}" height="38"><rect x="6" y="8" width="{w - 12}" height="12" rx="6" fill="{TRACK}"/>'
            f'<rect x="6" y="8" width="{max(X(value) - 6, 12):.1f}" height="12" rx="6" fill="{color}"/>'
            f'<rect x="{X(mark) - 1.5:.1f}" y="3" width="3" height="22" rx="1.5" fill="#040C38"/>'
            f'<text x="{X(mark):.1f}" y="36" font-size="12" fill="#475569" text-anchor="middle" font-weight="700">{mark_label}</text>'
            f'<text x="6" y="36" font-size="11" fill="#94A3B8">{minus(f"{lo:g}")}</text>'
            f'<text x="{w - 6}" y="36" font-size="11" fill="#94A3B8" text-anchor="end">{minus(f"{hi:g}")}</text></svg>')


def gauge(title, sub, vtxt, status, scolor, svg):
    return (f'<div class="gg"><div class="gh"><div><b>{title}</b><small>{sub}</small></div>'
            f'<div class="gv"><strong>{vtxt}</strong><span style="color:{scolor}">{status}</span></div></div>{svg}</div>')


def wednesday_after(d: date) -> date:
    return d + timedelta(days=(2 - d.weekday()) % 7 or 7)


def main():
    D = json.loads((HERE / 'core_slide_data.json').read_text())
    n = D['now']
    trend_in, earn_in = n['dmas'] >= 45, n['rev_z'] >= 0
    held, want = n['weight'], n['desired']
    wed = wednesday_after(date.fromisoformat(D['asof']))
    wtxt = f'{wed.day} {wed.strftime("%B")}'
    prev_wed = wed - timedelta(days=7)
    fmt = lambda d: f'{date.fromisoformat(d).strftime("%a")} {date.fromisoformat(d).day} {date.fromisoformat(d).strftime("%b")}'
    T = D['trades']
    new = [t for t in T if t['trade'] > str(prev_wed) or not t['trade'][0].isdigit()]
    last = T[-1] if T else None
    verb = 'HOLD' if not new else ('TRADED' if new[-1]['trade'][0].isdigit() else 'TRADE TOMORROW')
    if new:
        move = ' · '.join(f'Traded {fmt(t["trade"]) if t["trade"][0].isdigit() else "next day"}: {t["frm"]:.0%} → {t["to"]:.0%}' for t in new)
    else:
        move = f'No trade since the last committee. Last trade: {fmt(last["trade"])}, {last["frm"]:.0%} → {last["to"]:.0%}.' if last else 'No trade yet.'
    sig = n['sigma21']
    head = f'US equities: {"hold" if verb == "HOLD" else "now"} {held:.0%} invested'
    keys = ('Trend weak, but analysts still raise earnings: we stay invested. ' if not trend_in and earn_in else
            'Trend and earnings both positive: we stay invested. ' if trend_in and earn_in else
            'Trend positive: we stay invested. ' if trend_in else 'Trend and earnings both negative: we are in cash. ')
    lead = keys + (f'Volatility of {sig:.0%} keeps the size below 100%.' if held and sig > .15 else '')
    # what would make us trade (5-point rounding, 10-point gap)
    first = (date.fromisoformat(D['asof']).replace(day=1) + timedelta(days=32)).replace(day=1)
    while first.weekday() > 4:
        first += timedelta(days=1)
    fday = f'{first.day} {first.strftime("%b")}'
    ic = lambda c, t: f'<div class="ic" style="background:{c}">{t}</div>'
    rows = []
    if held > 0:
        if held - .10 >= .05:
            s_cut = .15 / (held - .075)
            near = ' class="near"' if s_cut - sig < .01 else ''
            rows.append((ic(AMBER, '↓'), f'Cut to {held - .10:.0%} or less',
                         f'if volatility rises above <em{near}>{s_cut:.1%}</em> (now {sig:.1%})'))
        if held + .10 <= 1.0001:
            s_add = .15 / (held + .075)
            rows.append((ic(GREEN, '↑'), f'Raise to {held + .10:.0%} or more', f'if volatility falls below <em>{s_add:.1%}</em> (now {sig:.1%})'))
        rows.append((ic(RED, '✕'), 'Go to cash',
                     f'on {fday} if trend &lt; 45 <em>and</em> earnings &lt; 0 (now {n["dmas"]:.0f} and {minus(f"{n["rev_z"]:+.1f}")}); '
                     'any day if trend &lt; 30 and earnings &lt; 0'))
    else:
        rows.append((ic(GREEN, '↑'), 'Buy back', f'on {fday} if trend ≥ 45 or earnings ≥ 0, or earlier on a re-entry signal'))
    if held > 0:
        rows.append((ic('#94A3B8', '↺'), 'Early buy-back rules: standby',
                     f'used only in cash. Today: VIX {n["vix"]:.0f}, ' + minus(f'{-n["fall_1y"]:.0%}') + f' from the 1‑year high, breadth {n["breadth_ratio"]:.2f} (trigger 1.97)'))
    trig = ''.join(f'<div class="tr">{i}<div><b>{t}</b><span>{d}</span></div></div>' for i, t, d in rows)
    st = 'active' if held == 0 else 'standby, used only in cash'
    rer = ' · '.join((f'fear over: VIX {n["vix"]:.0f}', 'shock: ' + minus(f'{-n["fall_1y"]:.0%}') + ' from 1-year high',
                      f'breadth: up/down {n["breadth_ratio"]:.2f} (trigger 1.97)'))
    W = 268
    gauges = ''.join([
        gauge('Trend', 'score 0–100 · switch at 45', f'{n["dmas"]:.0f}', 'ON' if trend_in else 'OFF', GREEN if trend_in else RED,
              bar(n['dmas'], 0, 100, GREEN if trend_in else RED, 45, '45', W)),
        gauge('Earnings', 'revisions, 3 months · switch at 0', minus(f'{n["rev_z"]:+.1f}'), 'ON' if earn_in else 'OFF',
              GREEN if earn_in else RED, bar(n['rev_z'], -3, 3, GREEN if earn_in else RED, 0, '0', W)),
        gauge('Volatility', 'size = 15% ÷ volatility', f'{sig:.0%}', f'SIZE {want:.0%}',
              AMBER if n['target'] < 1 else GREEN, bar(sig * 100, 0, 40, AMBER if sig > .15 else GREEN, 15, '15%', W)),
    ])
    asof = '/'.join(reversed(D['asof'].split('-')))
    data = json.dumps({k: D[k] for k in ('dates', 'px', 'w', 'ma200')})
    html = (HERE / 'core_slide_template.html').read_text()
    for k, v in dict(MOVE=move, HEAD=head, LEAD=lead, PXLAST=f'{D["px"][-1]:,.0f}', WED=wtxt.upper(), VERB=verb, HELD=f'{held:.0%}',
                     HELDW=f'{held * 100:.0f}', CASHW=f'{1 - held:.0%}', TRIG=trig,
                     GAUGES=gauges, RE=st, RER=rer, ASOF=asof, DATA=data).items():
        html = html.replace('{{' + k + '}}', v)
    (HERE / 'us_gauge_slide.html').write_text(html)
    print('written', HERE / 'us_gauge_slide.html', '|', verb, f'{held:.0%}', '| target', f'{want:.0%}')


if __name__ == '__main__':
    main()
