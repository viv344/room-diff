"""Self-contained HTML report: interactive 3D scene + one card per change with before/after crops."""
import base64
import html
import json

import numpy as np
import plotly.graph_objects as go

from .semantics import COLORS


def _sub(P, C, n):
    if len(P) > n:
        i = np.random.default_rng(0).choice(len(P), n, replace=False)
        P, C = P[i], C[i]
    return P, C


def _rgb(C, desat=0.45):
    """Background points are partly desaturated so the coloured change layers stand out."""
    C = np.clip(C, 0, 1)
    C = (1 - desat) * C + desat * C.mean(1, keepdims=True)
    c = (C * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def _box_lines(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    E = [((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),
         ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),
         ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1))]
    xs, ys, zs = [], [], []
    for a, b in E:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return xs, ys, zs


def figure(scene, changes):
    fig = go.Figure()
    for sess, vis in (("after", True), ("before", "legendonly")):
        P, C = _sub(*scene["clouds"][sess], 45000)
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name=f"{sess} scan",
                                   marker=dict(size=1.6, color=_rgb(C)), hoverinfo="skip", visible=vis))
    for sess, col in (("before", "#8e4ec6"), ("after", "#f76b15")):
        cs = np.array([v.center for v in scene["views"] if v.session == sess])
        fig.add_trace(go.Scatter3d(x=cs[:, 0], y=cs[:, 1], z=cs[:, 2], mode="lines+markers", name=f"{sess} camera path",
                                   line=dict(color=col, width=4), marker=dict(size=2.5, color=col), hoverinfo="name"))
    for i, c in enumerate(changes, 1):
        col = COLORS[c.type]
        tag = f"#{i} {c.type}: {c.label}"
        groups = [("now" if c.type == "moved" else c.type, c.points)]
        if c.type == "moved":
            groups.insert(0, ("was", c.old_points))
        for kind, P in groups:
            P = _sub(P, P, 4000)[0]
            faded = c.type == "moved" and kind == "was"
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", legendgroup=tag,
                                       name=tag, showlegend=kind in ("now", c.type), hovertext=f"{tag} ({kind})",
                                       hoverinfo="text", marker=dict(size=3, color=col, opacity=0.35 if faded else 0.95)))
            xs, ys, zs = _box_lines(P.min(0), P.max(0))
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", legendgroup=tag, showlegend=False,
                                       line=dict(color=col, width=5, dash="dash" if faded else "solid"), hoverinfo="skip"))
        top = c.points.max(0)
        fig.add_trace(go.Scatter3d(x=[c.points[:, 0].mean()], y=[c.points[:, 1].mean()], z=[top[2] + 0.12],
                                   mode="text", text=[f"<b>#{i} {c.label}</b>"], textfont=dict(color=col, size=13),
                                   legendgroup=tag, showlegend=False, hoverinfo="skip"))
        if c.type == "moved":
            a, b = c.old_points.mean(0), c.points.mean(0)
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines", legendgroup=tag,
                                       showlegend=False, line=dict(color=col, width=8), hoverinfo="skip"))
            d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            fig.add_trace(go.Cone(x=[b[0]], y=[b[1]], z=[b[2]], u=[d[0]], v=[d[1]], w=[d[2]], sizemode="absolute",
                                  sizeref=0.15, anchor="tip", colorscale=[[0, col], [1, col]], showscale=False,
                                  legendgroup=tag, hoverinfo="skip"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8a918b"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(128,128,128,0.12)", font=dict(size=11)),
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z up (m)",
                   **{ax: dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(128,128,128,0.25)",
                               zerolinecolor="rgba(128,128,128,0.4)", showbackground=True)
                      for ax in ("xaxis", "yaxis", "zaxis")},
                   camera=dict(eye=dict(x=-0.5, y=-1.3, z=1.7))))
    return fig


CSS = """
/* layout: 3D viewer left, change cards right; stacks to one column on narrow screens */
:root{--bg:#f5f6f4;--card:#ffffff;--ink:#1b1d1c;--muted:#626862;--line:#dfe3df;--chip:#e9ece8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151716;--card:#1e2120;--ink:#e8ebe8;--muted:#9aa29b;--line:#313532;--chip:#2a2e2b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#151716;--card:#1e2120;--ink:#e8ebe8;--muted:#9aa29b;--line:#313532;--chip:#2a2e2b;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:28px 32px 8px}h1{margin:0 0 6px;font-size:26px;text-wrap:balance}header p{margin:0;color:var(--muted);max-width:900px}
.summary{margin:14px 0 0;font-size:17px;color:var(--ink)!important}
.wrap{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);gap:20px;padding:18px 32px 32px}
@media(max-width:1000px){.wrap{grid-template-columns:1fr;padding:12px 16px}header{padding:20px 16px 4px}}
.viewer{background:var(--card);border:1px solid var(--line);border-radius:12px;height:72vh;min-height:420px;overflow:hidden}
.cards{display:flex;flex-direction:column;gap:14px;max-height:72vh;overflow:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h3{margin:0 0 4px;font-size:17px;display:flex;gap:8px;align-items:center}
.badge{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:#fff;border-radius:99px;padding:2px 8px}
.card p{margin:4px 0 10px}.meta{color:var(--muted);font-size:12.5px}
.crops{display:grid;grid-template-columns:1fr 1fr;gap:8px}.crops figure{margin:0}
.crops img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;border:1px solid var(--line)}
.crops figcaption{font-size:12px;color:var(--muted);text-align:center}
.note{display:inline-block;margin-top:10px;background:var(--chip);border-radius:6px;padding:4px 10px;font-size:13px;color:var(--muted)}
.stats{padding:0 32px 32px;color:var(--muted);font-size:13px}.stats code{background:var(--chip);padding:1px 5px;border-radius:4px}
"""


def card(i, c):
    col = COLORS[c.type]
    imgs = "".join(
        f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" alt="{k}"><figcaption>{k}</figcaption></figure>'
        for k, b in c.crops.items())
    meta = f"confidence {c.confidence:.0%} · centre ({', '.join(f'{x:.2f}' for x in c.centroid)}) m"
    if c.type == "moved":
        meta += f" · moved {c.extra['distance_m']:.2f} m"
    return (f'<div class="card"><h3><span class="badge" style="background:{col}">{c.type}</span>#{i} {html.escape(c.label)}</h3>'
            f'<p>{html.escape(c.description)}</p><div class="crops">{imgs}</div><div class="meta">{meta}</div></div>')


def write_report(path, scene, changes, summary, stats, note=None, fragment=False):
    """fragment=True omits the doctype/html/head/body wrapper (for hosts that add their own)."""
    fig = figure(scene, changes)
    plot = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["toImage"]},
                       default_height="100%", default_width="100%")
    cards = "".join(card(i, c) for i, c in enumerate(changes, 1)) or '<div class="card"><p>No changes detected.</p></div>'
    st = " · ".join(f"{k}: <code>{html.escape(str(v))}</code>" for k, v in stats.items())
    head = f"<title>What Changed in the Room?</title><style>{CSS}</style>"
    doc = f"""{head if fragment else '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + head + '</head><body>'}
<header><h1>What changed in the room?</h1>
<p>Both recordings were reconstructed into one shared, metric, gravity-aligned 3D model. A region counts as changed only
if the other recording looked straight through where it should be. Parts only one recording saw are not flagged.
Drag to orbit; click legend items to toggle layers (the <i>before scan</i> is hidden by default).</p>
<p class="summary">{html.escape(summary)}</p>{f'<span class="note">{html.escape(note)}</span>' if note else ''}</header>
<div class="wrap"><div class="viewer">{plot}</div><div class="cards">{cards}</div></div>
<div class="stats">{st}</div>{'' if fragment else '</body></html>'}"""
    with open(path, "w") as f:
        f.write(doc)


def write_json(path, changes, summary, stats):
    with open(path, "w") as f:
        json.dump({"summary": summary, "changes": [c.to_json() for c in changes], "stats": stats}, f, indent=2)
