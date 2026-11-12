"""MOCKUP ONLY — two 'Macro backdrop' slides for the Technical deck, for boss review.

Reads the Macro Navigator data file (market-herculis/web/data/navigator.json, built by
macro_health/tools/regime_market/navigator.py) and draws full slides in the deck style
(25.4 x 15.875 cm = 960 x 600 CSS px). Not wired into the pipeline.

  python3 mockups/macro_backdrop_mockup.py [--nav PATH] [--logo PATH]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from helpers.html_to_image import render_html_to_image  # noqa: E402

NAV = Path.home() / "Desktop/GitHub/Projects/market-herculis/web/data/navigator.json"
LOGO = HERE / "assets_logo.png"

NAVY, GOLD, CYAN, GREY = "#040C38", "#CC8932", "#00B0F0", "#666666"
SITE_GOLD = "#927636"
GREEN, AMBER, RED = "#10B981", "#F59E0B", "#EF4444"
W_PX, H_PX = 960, 600
CM = W_PX / 25.4  # px per cm

RNAME = {"improving|rising": "Heating up", "improving|steady": "Growth upswing", "improving|easing": "Sweet spot",
         "steady|rising": "Inflation pressure", "steady|steady": "Holding steady", "steady|easing": "Easing inflation",
         "slowing|rising": "Stagflation risk", "slowing|steady": "Slowdown", "slowing|easing": "Cooling down"}


def month(m: str) -> str:
    y, mo = m[:7].split("-")
    return date(int(y), int(mo), 1).strftime("%b %Y")


def long_month(m: str) -> str:
    y, mo = m[:7].split("-")
    return date(int(y), int(mo), 1).strftime("%B %Y")


def signed(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}".replace("-", "−")


# ── Slide frame: banner, gold rule, logo, bar, title, subtitle, source ──────────
CALIBRI_DIR = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")


def calibri_faces() -> str:
    """Embed the deck font when PowerPoint's copy is on this Mac (Chromium has no Calibri of its own)."""
    faces = []
    for fname, weight, style in [("Calibri.ttf", 400, "normal"), ("Calibrib.ttf", 700, "normal"), ("Calibrii.ttf", 400, "italic")]:
        p = CALIBRI_DIR / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            faces.append(f"@font-face{{font-family:Calibri;src:url(data:font/ttf;base64,{b64});font-weight:{weight};font-style:{style}}}")
    return "".join(faces)


def frame(section: str, title: str, subtitle: str, body: str, source: str, logo_b64: str) -> str:
    c = lambda v: f"{v * CM:.1f}px"  # noqa: E731
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{calibri_faces()}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W_PX}px;height:{H_PX}px;background:#fff;font-family:Calibri,Carlito,'Segoe UI','Helvetica Neue',Arial,sans-serif;color:{NAVY};position:relative;overflow:hidden}}
.banner{{position:absolute;left:0;top:0;width:100%;height:{c(1.37)};background:{NAVY}}}
.section{{position:absolute;left:{c(0.40)};top:0;height:{c(1.37)};display:flex;align-items:center;font:700 34.7px 'Times New Roman',Times,serif;color:#fff}}
.rule{{position:absolute;left:0;top:{c(1.37)};width:100%;height:2px;background:{GOLD}}}
.oval{{position:absolute;left:{c(22.13)};top:{c(0.22)};width:{c(2.80)};height:{c(2.80)};border-radius:50%;background:#fff}}
.logo{{position:absolute;left:{c(21.87)};top:{c(0.03)};width:{c(3.29)};height:{c(2.99)}}}
.bar{{position:absolute;left:{c(1.13)};top:{c(2.46)};width:6px;height:{c(2.05)};background:{NAVY}}}
.title{{position:absolute;left:{c(1.47)};top:{c(2.39)};width:{c(23.89)};padding:3px 10px}}
.title h1{{font-size:32px;font-weight:700;color:{CYAN};line-height:1.15}}
.title h2{{font-size:21.3px;font-weight:700;color:{NAVY};line-height:1.2;margin-top:2px}}
.body{{position:absolute;left:{c(2.20)};top:{c(4.94)};width:{c(21.0)};height:{c(9.25)}}}
.source{{position:absolute;left:{c(2.50)};top:{c(14.40)};font-size:9.3px;color:{GREY};font-style:italic}}
.mock{{position:absolute;right:10px;bottom:6px;font-size:9px;color:#b91c1c;font-weight:700;letter-spacing:.06em}}
</style></head><body>
<div class="banner"></div><div class="section">{section}</div><div class="rule"></div>
<div class="oval"></div><img class="logo" src="data:image/png;base64,{logo_b64}">
<div class="bar"></div>
<div class="title"><h1>{title}</h1><h2>{subtitle}</h2></div>
<div class="body">{body}</div>
<div class="source">{source}</div>
<div class="mock">MOCKUP</div>
</body></html>"""


# ── Slide 1: the regime ─────────────────────────────────────────────────────────
def quadrant_svg(rg: dict) -> str:
    """Growth trend (x) vs inflation trend (y): one point a quarter over two years, arrow to today."""
    W, H, L, R, T, B = 482, 345, 44, 462, 8, 312
    s, trail = rg["steady"], rg["trail"]
    q = [trail[k] for k in range(len(trail) - 1, -1, -3)][::-1]
    lim = 0.75
    X = lambda v: L + (v + lim) / (2 * lim) * (R - L)  # noqa: E731
    Y = lambda v: B - (v + lim) / (2 * lim) * (B - T)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="font-family:inherit">',
           f'<defs><marker id="ar" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{SITE_GOLD}"/></marker></defs>']
    # corner tints (identify regions only)
    tints = [("#f7f1e6", L, T, X(-s), Y(s)), ("#f3eee9", X(s), T, R, Y(s)),
             ("#eff2f3", L, Y(-s), X(-s), B), ("#f0f4ee", X(s), Y(-s), R, B),
             ("#f8f8f5", X(-s), T, X(s), B), ("#f8f8f5", L, Y(s), R, Y(-s)), ("#fdfdfb", X(-s), Y(s), X(s), Y(-s))]
    for col, x0, y0, x1, y1 in tints:
        out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" fill="{col}"/>')
    # today's zone
    zx = {"slowing": (L, X(-s)), "steady": (X(-s), X(s)), "improving": (X(s), R)}[rg["growth"]["word"]]
    zy = {"rising": (T, Y(s)), "steady": (Y(s), Y(-s)), "easing": (Y(-s), B)}[rg["inflation"]["word"]]
    out.append(f'<rect x="{zx[0]:.1f}" y="{zy[0]:.1f}" width="{zx[1] - zx[0]:.1f}" height="{zy[1] - zy[0]:.1f}" '
               f'fill="{SITE_GOLD}" fill-opacity=".10" stroke="{SITE_GOLD}" stroke-opacity=".6" stroke-width="1.2"/>')
    for v in [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]:
        col, w = ("#a9aea6", 1.2) if v == 0 else ("#ffffff", 1)
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{B}" stroke="{col}" stroke-width="{w}"/>'
                   f'<line x1="{L}" x2="{R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{col}" stroke-width="{w}"/>'
                   f'<text x="{X(v):.1f}" y="{B + 13}" text-anchor="middle" font-size="9.5" fill="#8a8f98">{signed(v)}</text>'
                   f'<text x="{L - 5}" y="{Y(v) + 3:.1f}" text-anchor="end" font-size="9.5" fill="#8a8f98">{signed(v)}</text>')
    for x in (X(-s), X(s)):
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{B}" stroke="#8b938c" stroke-dasharray="4 3" stroke-width=".8"/>')
    for y in (Y(s), Y(-s)):
        out.append(f'<line x1="{L}" x2="{R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#8b938c" stroke-dasharray="4 3" stroke-width=".8"/>')
    # corner names
    for txt, x, y, anc in [("Stagflation risk", L + 6, T + 14, "start"), ("Heating up", R - 6, T + 14, "end"),
                           ("Cooling down", L + 6, B - 7, "start"), ("Sweet spot", R - 6, B - 7, "end")]:
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anc}" font-size="10" font-weight="700" fill="#b9a27a">{txt}</text>')
    # trail: smooth path thickening towards today
    P = [(X(p["g"]), Y(p["i"])) for p in q]
    n = len(P)
    a, b = P[-2], P[-1]
    d = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
    cut = min(14, d * 0.45)
    P[-1] = (b[0] - (b[0] - a[0]) / d * cut, b[1] - (b[1] - a[1]) / d * cut)
    for k in range(n - 1):
        p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, n - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        out.append(f'<path d="M{p1[0]:.1f},{p1[1]:.1f} C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
                   f'fill="none" stroke="{SITE_GOLD}" stroke-linecap="round" stroke-width="{1.5 + 4 * k / (n - 1):.2f}" '
                   f'stroke-opacity="{0.3 + 0.7 * k / (n - 1):.2f}"{" marker-end=\"url(#ar)\"" if k == n - 2 else ""}/>')
    for k, (x, y) in enumerate(P[:-1]):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if k == 0 else 2.6}" fill="{"#fff" if k == 0 else SITE_GOLD}" '
                   f'stroke="{SITE_GOLD}" fill-opacity="{1 if k == 0 else 0.45 + 0.5 * k / (n - 1):.2f}"/>')
    out.append(f'<text x="{P[0][0] + 7:.1f}" y="{P[0][1] + 15:.1f}" font-size="10" fill="#7a6a45">{month(q[0]["m"])}</text>')
    lx, ly = X(q[-1]["g"]), Y(q[-1]["i"])
    out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="13" fill="{NAVY}" fill-opacity=".1"/>'
               f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="7" fill="{NAVY}"/>'
               f'<text x="{lx:.1f}" y="{ly - 18:.1f}" text-anchor="middle" font-size="11.5" font-weight="700" fill="{NAVY}">Now</text>')
    out.append(f'<text x="{(L + R) / 2:.1f}" y="{H - 4}" text-anchor="middle" font-size="10.5" fill="#55607a">'
               f'Growth trend → (12-month change, standard deviations)</text>'
               f'<text x="11" y="{(T + B) / 2:.1f}" text-anchor="middle" font-size="10.5" fill="#55607a" '
               f'transform="rotate(-90 11 {(T + B) / 2:.1f})">Inflation trend →</text></svg>')
    return "".join(out)


def regime_map(rg: dict) -> str:
    G, I = ["slowing", "steady", "improving"], ["rising", "steady", "easing"]
    cells = ['<div></div>'] + [f'<div class="mh">Growth<br>{g}</div>' for g in G]
    for i in I:
        cells.append(f'<div class="mv">Inflation {i}</div>')
        for g in G:
            on = g == rg["growth"]["word"] and i == rg["inflation"]["word"]
            cells.append(f'<div class="mc{" on" if on else ""}">{"<small>TODAY</small>" if on else ""}{RNAME[f"{g}|{i}"]}</div>')
    return f'<div class="map">{"".join(cells)}</div>'


def inputs_table(rg: dict) -> str:
    def v(x, u):
        return f"{x:.1f}" if u == "index" else (f"{signed(x)} pp" if u == "pp" else f"{x:.2f}%")
    short = ["ISM manufacturing", "Payrolls (6m ann.)", "Industrial production", "Unemployment (6m chg)", "CPI", "Core CPI"]
    rows = []
    for k, (x, nm) in enumerate(zip(rg["inputs"], short)):
        if k == 0:
            rows.append('<tr class="grp"><td colspan="3">Growth</td></tr>')
        if k == 4:
            rows.append('<tr class="grp"><td colspan="3">Inflation</td></tr>')
        better = x["value"] > x["year_ago"]
        arrow = "▲" if better else "▼"
        rows.append(f'<tr><th>{nm}</th><td><b>{v(x["value"], x["unit"])}</b> <i>{arrow}</i></td><td>{v(x["year_ago"], x["unit"])}</td></tr>')
    per = long_month(rg["inputs"][0]["period"])
    return (f'<table class="inp"><thead><tr><th></th><th>{per}</th><th>Year earlier</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def slide_regime(nav: dict, logo: str) -> str:
    rg = nav["regime"]
    g, i = rg["growth"], rg["inflation"]
    name = RNAME[f'{g["word"]}|{i["word"]}']
    body = f"""<style>
.wrap{{display:flex;gap:16px;height:100%}}
.right{{flex:1;display:flex;flex-direction:column;gap:8px;padding-top:2px}}
.map{{display:grid;grid-template-columns:62px repeat(3,1fr);gap:3px;font-size:10px}}
.mh{{text-align:center;color:#55607a;font-weight:700;line-height:1.1;padding-bottom:1px}}
.mv{{color:#55607a;font-weight:700;display:flex;align-items:center;line-height:1.1}}
.mc{{background:#f4f6f9;border-radius:3px;height:34px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#55607a;line-height:1.05}}
.mc.on{{background:{NAVY};color:#fff;font-weight:700;box-shadow:0 0 0 2px {GOLD}}}
.mc small{{font-size:7.5px;letter-spacing:.08em;color:{GOLD}}}
.inp{{border-collapse:collapse;font-size:10.5px;width:100%}}
.inp th,.inp td{{padding:2.4px 4px;text-align:right;border-bottom:1px solid #e6eaf0}}
.inp thead th{{color:#55607a;font-size:9.5px;border-bottom:1.5px solid {NAVY}}}
.inp tbody th{{text-align:left;font-weight:400}}
.inp td i{{font-style:normal;font-size:8px;color:#8a8f98}}
.inp .grp td{{text-align:left;font-weight:700;color:{NAVY};font-size:9.5px;border-bottom:none;padding-top:5px}}
.read{{font-size:10.5px;line-height:1.3;color:#334}}
.read b{{color:{NAVY}}}
</style>
<div class="wrap"><div>{quadrant_svg(rg)}</div>
<div class="right">{regime_map(rg)}
<div class="read"><b>Growth {g["word"]}</b>: trend stronger than in {g["pct"]:.0f}% of months since {rg["history_from"][:4]}. <b>Inflation {i["word"]}</b>: within ±{rg["steady"]:.2f} of zero.</div>
{inputs_table(rg)}</div></div>"""
    d = rg["as_of"]
    src = (f'Source: Herculis Macro Navigator (FRED / ALFRED, data as first published), Herculis Group. '
           f'Signal of {d[8:10]}/{d[5:7]}/{d[:4]}. Each point is one quarter; the arrow ends on today. Not a market forecast.')
    # months in a row in today's regime (same ±steady rule as navigator.py)
    word = lambda v, up, dn: up if v > rg["steady"] else dn if v < -rg["steady"] else "steady"  # noqa: E731
    run = 0
    for p in reversed(rg["trail"]):
        if (word(p["g"], "improving", "slowing"), word(p["i"], "rising", "easing")) != (g["word"], i["word"]):
            break
        run += 1
    nth = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(run, f"{run}th")
    sub = f"Growth {g['word']}, inflation {i['word']}: the {nth} month in a row in this regime."
    return frame("Macro", f"US Macro Regime: {name}", sub, body, src, logo)


# ── Slide 2: the recession dashboard ───────────────────────────────────────────
def big_gauge(rc: dict) -> str:
    """Semicircle, three zones (Distant / Near / Imminent), needle at the combined position."""
    g = rc["gauge"]
    cx, cy, r, w = 150, 140, 120, 24
    pt = lambda f, rad: (cx - rad * math.cos(math.pi * f), cy - rad * math.sin(math.pi * f))  # noqa: E731
    b1, b2 = g["bands"][0] / g["max"], g["bands"][1] / g["max"]
    out = ['<svg viewBox="-20 0 340 175" width="340" height="175">']
    for a, b, col in [(0, b1, "#cfe3d5"), (b1, b2, "#f0dcae"), (b2, 1, "#ebbcc6")]:
        x1, y1 = pt(a, r); x2, y2 = pt(b, r); x3, y3 = pt(b, r - w); x4, y4 = pt(a, r - w)
        out.append(f'<path d="M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f} L{x3:.1f},{y3:.1f} '
                   f'A{r - w},{r - w} 0 0 0 {x4:.1f},{y4:.1f} Z" fill="{col}" stroke="#fff" stroke-width="1.5"/>')
    for f, lab in [(b1 / 2, "Distant"), ((b1 + b2) / 2, "Near"), ((b2 + 1) / 2, "Imminent")]:
        x, y = pt(f, r + 8)
        anc = "end" if f < 0.4 else "start" if f > 0.6 else "middle"
        out.append(f'<text x="{x:.1f}" y="{y + 3:.1f}" text-anchor="{anc}" font-size="12" font-weight="700" fill="#55607a">{lab}</text>')
    f = g["position"] / g["max"]
    nx, ny = pt(f, r - w - 8)
    out.append(f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{NAVY}" stroke-width="4" stroke-linecap="round"/>'
               f'<circle cx="{cx}" cy="{cy}" r="8" fill="{NAVY}"/></svg>')
    return "".join(out)


def model_rows(rc: dict) -> str:
    rows = []
    for m in rc["models"] + [rc["yield_curve"]]:
        lo, hi = m["scale"]
        down = m.get("down", False)
        fired = (m["value"] <= m["trigger"]) if down else (m["value"] >= m["trigger"])
        pos = lambda v: max(0, min(1, (v - lo) / (hi - lo)))  # noqa: E731
        fv, ft = pos(m["value"]), pos(m["trigger"])
        if m["unit"] == "pp":
            val, trig = f'{signed(m["value"])} pp', f'{m["trigger"]:.2f} pp'
        else:
            val, trig = f'{m["value"]:.1f}%', f'{m["trigger"]:.0f}%'
        rule = {"sahm": f"signals at {trig}", "vci": f"signals at {trig}", "chauvet": f"3 months above {trig}",
                "hamilton": f"signals above {trig}", "curve": "signals below 0 (inverted)"}[m["key"]]
        # bar: safe side green, signal side red; for the curve the signal side is on the left
        safe, sig = "#d3e6d9", "#ebbcc6"
        grad = (f"linear-gradient(90deg,{sig} 0 {ft * 100:.1f}%,{safe} {ft * 100:.1f}% 100%)" if down
                else f"linear-gradient(90deg,{safe} 0 {ft * 100:.1f}%,{sig} {ft * 100:.1f}% 100%)")
        per = m["period"] if "Q" in m["period"] else month(m["period"])
        name = {"vci": "Participation-adj. Sahm"}.get(m["key"], m["name"])
        note = ' <span class="ref">reference, not in band</span>' if m["key"] == "curve" else ""
        rows.append(f"""<div class="row{' fired' if fired else ''}">
<div class="nm">{name}{note}<small>{per} · {rule}</small></div>
<div class="val">{val}</div>
<div class="trk" style="background:{grad}"><span class="tk" style="left:{ft * 100:.1f}%"></span><span class="dot" style="left:{fv * 100:.1f}%"></span></div>
<div class="st {'on' if fired else ''}">{'SIGNAL' if fired else 'QUIET'}</div></div>""")
    return "".join(rows)


def slide_recession(nav: dict, logo: str) -> str:
    rc = nav["recession"]
    band = rc["band"].capitalize()
    cal = rc["calibration"]
    n_fired = sum(1 for m in rc["models"] if m["value"] >= m["trigger"])
    curve_inv = rc["yield_curve"]["value"] < 0
    body = f"""<style>
.wrap{{display:flex;gap:22px;height:100%}}
.left{{width:330px;display:flex;flex-direction:column;align-items:center;padding-top:14px}}
.kick{{font-size:10.5px;color:#55607a;letter-spacing:.04em;margin-top:2px}}
.big{{font-size:40px;font-weight:700;color:#b7791f;line-height:1;margin:3px 0 7px}}
.cal{{font-size:11.5px;line-height:1.35;text-align:center;color:#334;max-width:285px}}
.cal b{{color:{NAVY};font-size:14px}}
.right{{flex:1;display:flex;flex-direction:column}}
.hd{{display:grid;grid-template-columns:170px 62px 1fr 56px;gap:10px;font-size:9.5px;color:#55607a;font-weight:700;border-bottom:1.5px solid {NAVY};padding:0 0 3px}}
.row{{display:grid;grid-template-columns:170px 62px 1fr 56px;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid #e6eaf0}}
.nm{{font-size:12px;font-weight:700;color:{NAVY};line-height:1.15}}
.nm small{{display:block;font-size:9.5px;font-weight:400;color:#6b7280;margin-top:1px}}
.ref{{font-size:8px;font-weight:400;color:#8a8f98;font-style:italic}}
.val{{font-size:15px;font-weight:700;text-align:right;color:{NAVY}}}
.trk{{position:relative;height:10px;border-radius:5px;margin:0 8px}}
.tk{{position:absolute;top:-4px;width:2px;height:18px;background:{NAVY};transform:translateX(-1px)}}
.dot{{position:absolute;top:-2px;width:14px;height:14px;border-radius:50%;background:{NAVY};border:2px solid #fff;transform:translateX(-7px);box-shadow:0 0 0 1px {NAVY}}}
.st{{font-size:10px;font-weight:700;text-align:center;padding:3px 0;border-radius:3px;background:#eef1f5;color:#55607a;letter-spacing:.05em}}
.st.on{{background:{RED};color:#fff}}
.row.fired .val{{color:#b91c1c}}
.foot{{font-size:9.5px;color:#6b7280;margin-top:7px;line-height:1.3}}
</style>
<div class="wrap"><div class="left">{big_gauge(rc)}
<div class="kick">ALL FOUR MODELS COMBINED · {long_month(rc["as_of"]).upper()}</div>
<div class="big">{band}</div>
<div class="cal">In the {cal["months"]} past months in this band, a recession began within 12 months in
<b>{cal["within_12m"]:.0f}%</b> of cases, against <b>{cal["base_12m"]:.0f}%</b> for all months.</div></div>
<div class="right"><div class="hd"><span>Model</span><span style="text-align:right">Latest</span><span>Position vs trigger (│)</span><span style="text-align:center">State</span></div>
{model_rows(rc)}
<div class="foot">The band uses the latest month in which all four models report ({long_month(rc["as_of"])}); single models are shown to their latest release.</div></div></div>"""
    src = ("Source: Herculis Macro Navigator (FRED: BLS, Chauvet–Piger, Hamilton; U.S. Treasury; NBER), Herculis Group. "
           f"Models updated to {long_month(rc['latest_partial'])}.")
    sub = (f"{n_fired} of 4 models signals; yield curve {'inverted' if curve_inv else 'not inverted'}. "
           f"Risk above normal, not imminent.")
    return frame("Macro", f"US Recession Dashboard: {band}", sub, body, src, logo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", type=Path, default=NAV)
    ap.add_argument("--logo", type=Path, default=LOGO)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    nav = json.loads(a.nav.read_text())
    logo = base64.b64encode(a.logo.read_bytes()).decode()
    for name, html in [("macro_backdrop_1_regime", slide_regime(nav, logo)),
                       ("macro_backdrop_2_recession", slide_recession(nav, logo))]:
        render_html_to_image(html, str(a.out / f"{name}.png"), size=(W_PX, H_PX), device_scale_factor=2)
        print("wrote", a.out / f"{name}.png")


if __name__ == "__main__":
    main()
