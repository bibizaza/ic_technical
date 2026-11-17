"""Build mockups/us_gauge_slide.html from mockups/core_slide_data.json (written by research/us_gauge/core_slide_data.py).

Every reading on the right panel is a gauge: a scale, the switch mark, the current value.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RED, AMBER, GREEN, TRACK, MUTE = '#F87171', '#FBBF24', '#4ADE80', '#2F4B6B', '#9FB2CC'


def minus(s: str) -> str:
    return s.replace('-', '−')


def hgauge(title, sub, value, lo, hi, zones, marks, vtxt, status, scolor, w=500, cls='g'):
    """Horizontal gauge. zones: [(a, b, color)]; marks: [(x, label)]."""
    x0, x1, yb, h = 8, w - 8, 30, 9
    X = lambda v: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (x1 - x0)
    s = [f'<rect x="{x0}" y="{yb}" width="{x1 - x0}" height="{h}" rx="5" fill="{TRACK}"/>']
    s += [f'<rect x="{X(a):.1f}" y="{yb}" width="{X(b) - X(a):.1f}" height="{h}" fill="{c}" opacity=".55"/>' for a, b, c in zones]
    for m, lab in marks:
        s.append(f'<line x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{yb - 4}" y2="{yb + h + 4}" stroke="#fff" stroke-width="2"/>'
                 f'<text x="{X(m):.1f}" y="{yb + h + 17}" font-size="11.5" fill="{MUTE}" text-anchor="middle">{lab}</text>')
    s.append(f'<path d="M{X(value):.1f},{yb - 2} l-7,-11 h14 z" fill="#fff"/>'
             f'<text x="{X(value):.1f}" y="{yb - 16}" font-size="15" font-weight="700" fill="#fff" text-anchor="middle">{vtxt}</text>')
    s.append(f'<text x="{x0}" y="{yb + h + 17}" font-size="11.5" fill="{MUTE}">{minus(f"{lo:g}")}</text>'
             f'<text x="{x1}" y="{yb + h + 17}" font-size="11.5" fill="{MUTE}" text-anchor="end">{minus(f"{hi:g}")}</text>')
    return (f'<div class="{cls}"><div class="gh"><div><b>{title}</b><small>{sub}</small></div><span style="color:{scolor}">{status}</span></div>'
            f'<svg width="{w}" height="58">{"".join(s)}</svg></div>')


def dial(weight, target):
    cx, cy, r = 150, 118, 96
    P = lambda f, rr=r: (cx - rr * math.cos(math.pi * f), cy - rr * math.sin(math.pi * f))
    def arc(a, b, c, wdt=18):
        (xa, ya), (xb, yb) = P(a), P(b)
        return f'<path d="M{xa:.1f},{ya:.1f} A{r},{r} 0 0 1 {xb:.1f},{yb:.1f}" stroke="{c}" stroke-width="{wdt}" fill="none"/>'
    s = arc(0, 1, TRACK) + arc(0, weight, '#38BDF8')
    tx, ty = P(target, r + 14)
    ix, iy = P(target, r - 14)
    s += f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{AMBER}" stroke-width="3"/>'
    lx, ly = P(target, r + 30)
    s += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="12" fill="{AMBER}" text-anchor="middle" font-weight="700">target {target:.0%}</text>'
    nx, ny = P(weight, r - 28)
    s += f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#fff" stroke-width="4" stroke-linecap="round"/><circle cx="{cx}" cy="{cy}" r="7" fill="#fff"/>'
    s += f'<text x="{cx - r}" y="{cy + 20}" font-size="12" fill="{MUTE}" text-anchor="middle">0% cash</text>'
    s += f'<text x="{cx + r}" y="{cy + 20}" font-size="12" fill="{MUTE}" text-anchor="middle">100%</text>'
    return f'<svg width="300" height="142">{s}</svg>'


def main():
    D = json.loads((HERE / 'core_slide_data.json').read_text())
    n = D['now']
    P = {p['window']: p for p in D['periods']}
    full = P['FULL 1999-02 to 2026-08']
    trend_in, earn_in = n['dmas'] >= 45, n['rev_z'] >= 0
    cash = not trend_in and not earn_in
    w, tg = n['weight'], n['target']
    state = 'CASH' if cash else 'INVESTED'
    head = ('S&amp;P 500: Invested — size trimmed' if not cash and tg < w - .05 else f'S&amp;P 500: {state.title()}')
    keys = [
        hgauge('Trend score', 'DMAS · below 45 the trend key turns red', n['dmas'], 0, 100,
               [(0, 45, RED), (45, 100, GREEN)], [(30, '30'), (45, '45 switch')], f'{n["dmas"]:.0f}',
               'IN' if trend_in else 'OUT', GREEN if trend_in else RED),
        hgauge('Earnings momentum', 'analysts’ 3-month revisions, in σ · below 0 turns red', n['rev_z'], -3, 3,
               [(-3, 0, RED), (0, 3, GREEN)], [(0, '0 switch')], minus(f'{n["rev_z"]:+.1f}'),
               'IN' if earn_in else 'OUT', GREEN if earn_in else RED),
        hgauge('Market volatility', f'above 15% the size is cut · size now {tg:.0%}', n['sigma21'] * 100, 0, 40,
               [(0, 15, GREEN), (15, 40, AMBER)], [(15, '15% full size')], f'{n["sigma21"]:.1%}',
               f'SIZE {tg:.0%}', AMBER if tg < 1 else GREEN),
    ]
    sb = 'ACTIVE' if cash else 'STANDBY'
    re = [
        hgauge('Fear over', 'VIX: half its peak, below 30', n['vix'], 0, 60,
               [(0, 30, GREEN), (30, 60, RED)], [(30, '30')], f'{n["vix"]:.1f}', 'ON' if n['f_on'] else sb, GREEN if n['f_on'] else MUTE, w=158, cls='g mini'),
        hgauge('Shock', 'fall from 1-year high', n['fall_1y'] * 100, 0, 30,
               [(15, 30, AMBER)], [(15, '15%')], f'{n["fall_1y"]:.1%}', 'ON' if n['s_on'] else sb, GREEN if n['s_on'] else MUTE, w=158, cls='g mini'),
        hgauge('Breadth', 'NYSE stocks up ÷ down', n['breadth_ratio'], 0, 3,
               [(1.97, 3, GREEN)], [(1.97, '1.97')], f'{n["breadth_ratio"]:.2f}', 'ON' if n['b_on'] else sb, GREEN if n['b_on'] else MUTE, w=158, cls='g mini'),
    ]
    cols = [('2008 crisis', '2008 crisis 2007-10 to 2009-03'), ('COVID fall', 'COVID fall 19 Feb-23 Mar 2020'), ('2022', '2022 full year'),
            ('Liberation Day fall', 'Liberation Day fall 19 Feb-8 Apr 2025'), ('COVID rebound', 'COVID rebound 24 Mar-31 Aug 2020'), ('2025', '2025 full year')]
    th = ''.join(f'<th>{a}</th>' for a, _ in cols) + '<th class="sep">Sharpe 1999–2026</th><th>Worst fall</th>'
    tc = ''.join(f'<td class="{"up" if P[k]["core_total"] > P[k]["bh_total"] else "dn"}">{minus(f"{P[k]['core_total']:+.0%}")}</td>' for _, k in cols)
    tc += f'<td class="sep up">{full["core_sharpe"]:.2f}</td><td class="up">{minus(f"{full['core_dd']:.0%}")}</td>'
    tb = ''.join(f'<td>{minus(f"{P[k]['bh_total']:+.0%}")}</td>' for _, k in cols)
    tb += f'<td class="sep">{full["bh_sharpe"]:.2f}</td><td>{minus(f"{full['bh_dd']:.0%}")}</td>'
    asof = '/'.join(reversed(D['asof'].split('-')))
    lead = (f'The trend score fell below its switch (45), but analysts are still raising earnings, so the gauge stays invested. '
            f'Volatility is up, so the position steps down to {tg:.0%} at the next rebalance.' if not cash and not trend_in and earn_in
            else 'Both keys are in: the gauge is invested.' if not cash else 'Both keys are out: the gauge is in cash.')
    data = json.dumps({k: D[k] for k in ('dates', 'px', 'w', 'dec')})
    html = (HERE / 'core_slide_template.html').read_text()
    for k, v in dict(HEAD=head, LEAD=lead, PXLAST=f'{D["px"][-1]:,.0f}', STATE=state, DIAL=dial(w, tg),
                     STATESUB=f'{w:.0%} of the portfolio in the S&amp;P 500 today', KEYS=''.join(keys), RE=''.join(re),
                     TH=th, TC=tc, TB=tb, ASOF=asof, DATA=data).items():
        html = html.replace('{{' + k + '}}', v)
    (HERE / 'us_gauge_slide.html').write_text(html)
    print('written', HERE / 'us_gauge_slide.html')


if __name__ == '__main__':
    main()
