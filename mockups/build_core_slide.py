"""Build mockups/us_gauge_slide.html from mockups/core_slide_data.json (written by research/us_gauge/core_slide_data.py).

Liquid-glass style: every reading is a glass capsule with a liquid fill, a glowing switch notch and its value.
Rebalancing rule (research/us_gauge/volsize.daily_weights): Tuesday decisions; on the first Tuesday of a month
the position moves to the volatility target if the gap is >= 10 points, on other Tuesdays only if >= 25 points.
"""
import json
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIQ = {'green': ('#6EE7B7', '#10B981', '#047857'), 'red': ('#FDA4AF', '#F43F5E', '#9F1239'),
       'amber': ('#FDE68A', '#F59E0B', '#B45309'), 'sky': ('#BAE6FD', '#38BDF8', '#1D4ED8'),
       'grey': ('#CBD5E1', '#64748B', '#334155')}
_uid = [0]


def minus(s: str) -> str:
    return s.replace('-', '−')


def capsule(value, lo, hi, color, marks=(), w=500, h=34, ghost=None, ghost_label=''):
    """Glass capsule. marks: [(x, label)]. ghost: a second level shown as a dashed glowing line."""
    _uid[0] += 1
    u = _uid[0]
    c1, c2, c3 = LIQ[color]
    pad, top = 4, 6
    X = lambda v: pad + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - 2 * pad)
    ih = h - 2 * pad
    r = ih / 2
    fw = max(X(value) - pad, ih)
    s = [f'''<defs>
 <linearGradient id="lq{u}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset=".45" stop-color="{c2}"/><stop offset="1" stop-color="{c3}"/></linearGradient>
 <linearGradient id="gl{u}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".75"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
 <linearGradient id="sh{u}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".22"/><stop offset=".5" stop-color="#fff" stop-opacity=".04"/><stop offset="1" stop-color="#fff" stop-opacity=".12"/></linearGradient>
 <filter id="bl{u}" x="-20%" y="-80%" width="140%" height="260%"><feGaussianBlur stdDeviation="7"/></filter>
</defs>''']
    s.append(f'<rect x="{pad}" y="{top + pad + 4}" width="{fw}" height="{ih}" rx="{r}" fill="{c2}" opacity=".55" filter="url(#bl{u})"/>')
    s.append(f'<rect x=".75" y="{top + .75}" width="{w - 1.5}" height="{h - 1.5}" rx="{h / 2}" fill="url(#sh{u})" stroke="rgba(255,255,255,.38)" stroke-width="1.2"/>')
    s.append(f'<rect x="{pad}" y="{top + pad}" width="{w - 2 * pad}" height="{ih}" rx="{r}" fill="rgba(2,6,23,.55)"/>')
    s.append(f'<rect x="{pad}" y="{top + pad}" width="{fw}" height="{ih}" rx="{r}" fill="url(#lq{u})"/>')
    s.append(f'<rect x="{pad + 5}" y="{top + pad + 2}" width="{max(fw - 10, 4)}" height="{ih * .42}" rx="{ih * .21}" fill="url(#gl{u})"/>')
    s.append(f'<rect x="{h / 2}" y="{top + 2}" width="{w - h}" height="2" rx="1" fill="#fff" opacity=".35"/>')
    if ghost is not None:
        gx = X(ghost)
        s.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top - 3}" y2="{top + h + 3}" stroke="#FBBF24" stroke-width="3" stroke-dasharray="5 3"/>'
                 f'<text x="{gx:.1f}" y="{top + h + 18}" font-size="13" font-weight="700" fill="#FBBF24" text-anchor="middle">{ghost_label}</text>')
    for m, lab in marks:
        mx = X(m)
        s.append(f'<rect x="{mx - 3:.1f}" y="{top}" width="6" height="{h}" rx="3" fill="#fff" opacity=".35" filter="url(#bl{u})"/>'
                 f'<rect x="{mx - 1.5:.1f}" y="{top + 1}" width="3" height="{h - 2}" rx="1.5" fill="#fff"/>'
                 f'<text x="{mx:.1f}" y="{top + h + 17}" font-size="12" fill="#A5B4CF" text-anchor="middle">{lab}</text>')
    s.append(f'<text x="2" y="{top + h + 17}" font-size="11.5" fill="#64748B">{minus(f"{lo:g}")}</text>'
             f'<text x="{w - 2}" y="{top + h + 17}" font-size="11.5" fill="#64748B" text-anchor="end">{minus(f"{hi:g}")}</text>')
    return f'<svg width="{w}" height="{h + 26}" style="overflow:visible">{"".join(s)}</svg>'


def chip(text, color):
    return f'<span class="chip {color}">{text}</span>'


def block(title, sub, value_txt, chip_html, svg, cls='blk'):
    return (f'<div class="{cls}"><div class="bh"><div class="bt"><b>{title}</b><small>{sub}</small></div>'
            f'<div class="bv"><strong>{value_txt}</strong>{chip_html}</div></div>{svg}</div>')


def next_rebalance(asof: str) -> date:
    d = date.fromisoformat(asof)
    first = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return first + timedelta(days=(1 - first.weekday()) % 7)        # first Tuesday of next month


def main():
    D = json.loads((HERE / 'core_slide_data.json').read_text())
    n = D['now']
    P = {p['window']: p for p in D['periods']}
    full = P[D['full_label']]
    trend_in, earn_in = n['dmas'] >= 45, n['rev_z'] >= 0
    cash = not trend_in and not earn_in
    w, tg = n['weight'], n['target'] * (0 if cash else 1)
    gap = abs(w - tg) * 100
    nr = next_rebalance(D['asof'])
    nr_txt = f'{nr.strftime("%a")} {nr.day} {nr.strftime("%b")}'
    if cash:
        head, move = 'S&amp;P 500: Cash', 'We stay in cash until a key turns back on, or an early re-entry fires.'
    elif gap >= 10:
        head = f'S&amp;P 500: Invested — stepping down to {tg:.0%} on {nr.day} {nr.strftime("%B")}'
        move = (f'Volatility rose, so the target is now {tg:.0%}. We keep {w:.0%} until the monthly rebalance on {nr_txt}, '
                f'unless the gap reaches 25 points before (today {gap:.0f}).')
    else:
        head, move = f'S&amp;P 500: Invested at {w:.0%}', f'The target ({tg:.0%}) is within 10 points of today: no trade at the {nr_txt} rebalance.'
    why = ('The trend score fell below its switch, but analysts still raise earnings, so we stay invested. '
           if not trend_in and earn_in else 'Trend and earnings both point up. ' if trend_in and earn_in else '')
    pos = (f'<div class="pos"><div class="ph"><div class="bt"><b>POSITION IN THE S&amp;P 500</b><small>the rest sits in T-bills</small></div>'
           f'<div class="pnow"><div><span>today</span><strong>{w:.0%}</strong></div><em>→</em>'
           f'<div><span>{nr_txt}</span><strong class="amb">{tg:.0%}</strong></div></div></div>'
           + capsule(w * 100, 0, 100, 'sky', [], w=524, h=36, ghost=tg * 100, ghost_label=f'target {tg:.0%}')
           + f'<div class="rule"><span><i>mid-month</i> trade only if gap ≥ 25 pts · now {gap:.0f}</span>'
             f'<span><i>{nr_txt}</i> move to target if gap ≥ 10 pts</span></div></div>')
    keys = [
        block('Trend score', 'DMAS 0–100 · switch at 45', f'{n["dmas"]:.0f}', chip('ON' if trend_in else 'OFF', 'g' if trend_in else 'r'),
              capsule(n['dmas'], 0, 100, 'green' if trend_in else 'red', [(45, '45')], w=524, h=28)),
        block('Earnings momentum', 'analysts’ 3-month revisions (σ) · switch at 0', minus(f'{n["rev_z"]:+.1f}'),
              chip('ON' if earn_in else 'OFF', 'g' if earn_in else 'r'),
              capsule(n['rev_z'], -3, 3, 'green' if earn_in else 'red', [(0, '0')], w=524, h=28)),
        block('Market volatility', '21-day · above 15% the position is scaled down', f'{n["sigma21"]:.1%}',
              chip(f'SIZE {n["target"]:.0%}', 'a' if n['target'] < 1 else 'g'),
              capsule(n['sigma21'] * 100, 0, 40, 'amber' if n['sigma21'] > .15 else 'green', [(15, '15%')], w=524, h=28)),
    ]
    st = 'ACTIVE' if cash else 'STANDBY'
    re = [
        block('Fear over', 'VIX halves, below 30', f'{n["vix"]:.1f}', chip('ON' if n['f_on'] else st, 'g' if n['f_on'] else 'm'),
              capsule(n['vix'], 0, 60, 'grey', [(30, '30')], w=152, h=20), 'mini'),
        block('Shock', 'fall from 1-yr high', f'{n["fall_1y"]:.1%}', chip('ON' if n['s_on'] else st, 'g' if n['s_on'] else 'm'),
              capsule(n['fall_1y'] * 100, 0, 30, 'grey', [(15, '15%')], w=152, h=20), 'mini'),
        block('Breadth', 'NYSE up ÷ down, 10 d', f'{n["breadth_ratio"]:.2f}', chip('ON' if n['b_on'] else st, 'g' if n['b_on'] else 'm'),
              capsule(n['breadth_ratio'], 0, 3, 'grey', [(1.97, '1.97')], w=152, h=20), 'mini'),
    ]
    cols = [('2008 crisis', '2008 crisis 2007-10 to 2009-03'), ('COVID fall', 'COVID fall 19 Feb-23 Mar 2020'), ('2022', '2022 full year'),
            ('Lib. Day fall', 'Liberation Day fall 19 Feb-8 Apr 2025'), ('COVID rebound', 'COVID rebound 24 Mar-31 Aug 2020'), ('2025', '2025 full year')]
    th = ''.join(f'<th>{a}</th>' for a, _ in cols) + '<th class="sep">Sharpe 1999–2026</th><th>Worst fall</th>'
    tc = ''.join(f'<td class="{"up" if P[k]["core_total"] > P[k]["bh_total"] else "dn"}">{minus(f"{P[k]['core_total']:+.0%}")}</td>' for _, k in cols)
    tc += f'<td class="sep up">{full["core_sharpe"]:.2f}</td><td class="up">{minus(f"{full['core_dd']:.1%}")}</td>'
    tb = ''.join(f'<td>{minus(f"{P[k]['bh_total']:+.0%}")}</td>' for _, k in cols)
    tb += f'<td class="sep">{full["bh_sharpe"]:.2f}</td><td>{minus(f"{full['bh_dd']:.1%}")}</td>'
    asof = '/'.join(reversed(D['asof'].split('-')))
    data = json.dumps({k: D[k] for k in ('dates', 'px', 'w', 'tgt', 'dec')})
    html = (HERE / 'core_slide_template.html').read_text()
    for k, v in dict(HEAD=head, LEAD=why + move, PXLAST=f'{D["px"][-1]:,.0f}', POS=pos, KEYS=''.join(keys), RE=''.join(re),
                     TH=th, TC=tc, TB=tb, ASOF=asof, DATA=data).items():
        html = html.replace('{{' + k + '}}', v)
    (HERE / 'us_gauge_slide.html').write_text(html)
    print('written', HERE / 'us_gauge_slide.html')


if __name__ == '__main__':
    main()
