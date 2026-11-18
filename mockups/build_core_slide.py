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
    lo_, hi_ = max(held - .10, 0), min(held + .10, 1)
    if new:
        move = ' · '.join(f'{fmt(t["trade"]) if t["trade"][0].isdigit() else "next day"}: {t["frm"]:.0%} → {t["to"]:.0%}' for t in new)
    else:
        move = (f'No trade since the last committee · next trade if the target goes below {lo_:.0%}'
                + (f' or above {hi_:.0%}' if hi_ < 1 else ''))
    head = f'S&amp;P 500: hold at {held:.0%}' if verb == 'HOLD' else f'S&amp;P 500: now at {held:.0%}'
    why_vol = (f'Volatility is {n["sigma21"]:.0%}, so the target is {want:.0%}' if n['sigma21'] > .15
               else f'Volatility is low, so the target is {want:.0%}')
    keys = ('The trend is weak, but analysts are still raising earnings, so we stay invested. ' if not trend_in and earn_in else
            'Trend and earnings are both positive. ' if trend_in and earn_in else
            'The trend is positive. ' if trend_in else 'Trend and earnings are both negative. ')
    lead = keys + (f'{why_vol}: less than 10 points from our {held:.0%}, so no trade.' if abs(want - held) < .10
                   else f'{why_vol}.')
    if last and verb == 'HOLD':
        lead += f' Last trade: {fmt(last["trade"])}, {last["frm"]:.0%} → {last["to"]:.0%}.'
    st = 'ACTIVE' if n['state'] == 0 else 'standby, used only in cash'
    re_rows = ''.join(f'<div class="rr"><b>{k}</b><span>{v}</span></div>' for k, v in (
        ('Fear over', f'VIX {n["vix"]:.0f} · back to 50% at half its peak, below 30'),
        ('Shock', minus(f'{-n["fall_1y"]:.0%}') + ' from 1-year high · back to 100% after −15%, +10%'),
        ('Breadth', f'up/down ratio {n["breadth_ratio"]:.2f} · back to 100% above 1.97')))
    gauges = ''.join([
        gauge('Trend', 'trend score 0–100 · switch at 45', f'{n["dmas"]:.0f}', 'ON' if trend_in else 'OFF', GREEN if trend_in else RED,
              bar(n['dmas'], 0, 100, GREEN if trend_in else RED, 45, '45')),
        gauge('Earnings', 'analysts’ revisions, 3 months · switch at 0', minus(f'{n["rev_z"]:+.1f}'), 'ON' if earn_in else 'OFF',
              GREEN if earn_in else RED, bar(n['rev_z'], -3, 3, GREEN if earn_in else RED, 0, '0')),
        gauge('Volatility', 'size = 15% ÷ volatility, at most 100%', f'{n["sigma21"]:.0%}', f'TARGET {want:.0%}',
              AMBER if n['target'] < 1 else GREEN, bar(n['sigma21'] * 100, 0, 40, AMBER if n['sigma21'] > .15 else GREEN, 15, '15%')),
    ])
    asof = '/'.join(reversed(D['asof'].split('-')))
    data = json.dumps({k: D[k] for k in ('dates', 'px', 'w', 'ma200')})
    html = (HERE / 'core_slide_template.html').read_text()
    for k, v in dict(MOVE=move, HEAD=head, LEAD=lead, PXLAST=f'{D["px"][-1]:,.0f}', WED=wtxt, VERB=f'{verb} · {held:.0%}', HELD=f'{held:.0%}',
                     GAUGES=gauges, RE=st, RER=re_rows, ASOF=asof, DATA=data).items():
        html = html.replace('{{' + k + '}}', v)
    (HERE / 'us_gauge_slide.html').write_text(html)
    print('written', HERE / 'us_gauge_slide.html', '|', verb, f'{held:.0%}', '| target', f'{want:.0%}')


if __name__ == '__main__':
    main()
