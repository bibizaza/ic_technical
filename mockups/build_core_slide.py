"""Build mockups/us_gauge_slide.html from mockups/core_slide_data.json (written by research/us_gauge/core_slide_data.py).

One decision per Wednesday committee: move to the target if it differs from the current weight by 10 points or more
(or if we go to / leave cash); otherwise hold. All numbers come from research/us_gauge/RESULTS_core.md (row 6).
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
    P = {p['window']: p for p in D['periods']}
    full = P[D['full_label']]
    trend_in, earn_in = n['dmas'] >= 45, n['rev_z'] >= 0
    held, want = n['weight'], n['desired']
    act = (want == 0) != (held == 0) or abs(want - held) >= .10 - 1e-12
    if not act:
        verb, new = 'HOLD', held
    elif want == 0:
        verb, new = 'GO TO CASH', 0.
    elif held == 0:
        verb, new = f'BUY BACK TO {want:.0%}', want
    else:
        verb, new = (f'REDUCE TO {want:.0%}' if want < held else f'INCREASE TO {want:.0%}'), want
    wed = wednesday_after(date.fromisoformat(D['asof']))
    wtxt = f'{wed.day} {wed.strftime("%B")}'
    ch = [(d, x) for i, (d, x) in enumerate(zip(D['dates'], D['w'])) if i and x != D['w'][i - 1]]
    last_d, last_w = ch[-1] if ch else (None, None)
    prev_w = D['w'][D['dates'].index(last_d) - 1] if last_d else None
    last_wed = date.fromisoformat(last_d) - timedelta(days=1) if last_d else None
    head = f'S&amp;P 500: hold at {held:.0%}' if verb == 'HOLD' else f'S&amp;P 500: {verb.lower()}'
    why_vol = (f'Volatility is {n["sigma21"]:.0%}, so the target size is {n["target"]:.0%}' if n['sigma21'] > .15
               else 'Volatility is low, so the target size is 100%')
    keys = ('The trend is weak, but analysts are still raising earnings, so we stay invested. ' if not trend_in and earn_in else
            'Trend and earnings are both positive. ' if trend_in and earn_in else
            'The trend is positive. ' if trend_in else 'Trend and earnings are both negative. ')
    lead = keys + (f'{why_vol}: within 10 points of today’s {held:.0%}, so no trade this week.' if verb == 'HOLD'
                   else f'{why_vol}, more than 10 points from today’s {held:.0%}: we trade.')
    if last_d and verb == 'HOLD':
        lead += f' Last change: {last_wed.day} {last_wed.strftime("%b")}, from {prev_w:.0%} to {last_w:.0%}.'
    st = 'ACTIVE' if held == 0 else 'standby, used only while in cash'
    gauges = ''.join([
        gauge('Trend', 'trend score 0–100 · switch at 45', f'{n["dmas"]:.0f}', 'ON' if trend_in else 'OFF', GREEN if trend_in else RED,
              bar(n['dmas'], 0, 100, GREEN if trend_in else RED, 45, '45')),
        gauge('Earnings', 'analysts’ revisions, 3 months · switch at 0', minus(f'{n["rev_z"]:+.1f}'), 'ON' if earn_in else 'OFF',
              GREEN if earn_in else RED, bar(n['rev_z'], -3, 3, GREEN if earn_in else RED, 0, '0')),
        gauge('Volatility', 'size = 15% ÷ volatility, at most 100%', f'{n["sigma21"]:.0%}', f'SIZE {n["target"]:.0%}',
              AMBER if n['target'] < 1 else GREEN, bar(n['sigma21'] * 100, 0, 40, AMBER if n['sigma21'] > .15 else GREEN, 15, '15%')),
    ])
    cols = [('2008 crisis', '2008 crisis 2007-10 to 2009-03'), ('2022', '2022 full year'),
            ('COVID rebound', 'COVID rebound 24 Mar-31 Aug 2020'), ('2025', '2025 full year')]
    th = ''.join(f'<th>{a}</th>' for a, _ in cols) + '<th>Sharpe 1999–2026</th><th>Worst fall</th>'
    tc = (''.join(f'<td>{minus(f"{P[k]['core_total']:+.0%}")}</td>' for _, k in cols)
          + f'<td>{full["core_sharpe"]:.2f}</td><td>{minus(f"{full['core_dd']:.0%}")}</td>')
    tb = (''.join(f'<td>{minus(f"{P[k]['bh_total']:+.0%}")}</td>' for _, k in cols)
          + f'<td>{full["bh_sharpe"]:.2f}</td><td>{minus(f"{full['bh_dd']:.0%}")}</td>')
    asof = '/'.join(reversed(D['asof'].split('-')))
    data = json.dumps({k: D[k] for k in ('dates', 'px', 'w')})
    html = (HERE / 'core_slide_template.html').read_text()
    for k, v in dict(MOVE=(f'S&amp;P 500 share stays at {held:.0%}' if verb == 'HOLD' else f'S&amp;P 500 share: {held:.0%} → {new:.0%}'),
                     TUE=(wed - timedelta(days=1)).strftime('%d/%m/%Y'), HEAD=head, LEAD=lead, PXLAST=f'{D["px"][-1]:,.0f}', WED=wtxt, VERB=verb, NEW=f'{new:.0%}', HELD=f'{held:.0%}',
                     GAUGES=gauges, RE=st, TH=th, TC=tc, TB=tb, ASOF=asof, DATA=data).items():
        html = html.replace('{{' + k + '}}', v)
    (HERE / 'us_gauge_slide.html').write_text(html)
    print('written', HERE / 'us_gauge_slide.html', '|', verb, f'{held:.0%} -> {new:.0%}')


if __name__ == '__main__':
    main()
