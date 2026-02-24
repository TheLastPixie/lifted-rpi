"""
Interactive Plotly + HTML explorer for lifted-space 2-D / 3-D projections.

Precomputes all pairwise and triple-wise convex hulls, embeds them as
JSON inside a self-contained HTML snippet that can be displayed via
``IPython.display.HTML``.
"""
from __future__ import annotations

import itertools
import json
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


# ──── vertex helpers ────

def _parts_from_vset(vset, eng):
    V = vset.V
    n, m, w = eng.n, eng.m, eng.w
    X = V[:, :n]
    Vc = V[:, n : n + m]
    W = V[:, n + m : n + m + w]
    U = (eng.K @ X.T).T + Vc
    return X, Vc, U, W


def _token_columns(vset, eng):
    X, Vc, U, W = _parts_from_vset(vset, eng)
    cols = {"x1": X[:, 0], "v1": X[:, 1]}
    if eng.n >= 4:
        cols["x2"] = X[:, 2]
        cols["v2"] = X[:, 3]
    for j in range(eng.m):
        cols[f"u{j+1}"] = U[:, j]
    for j in range(eng.w):
        cols[f"w{j+1}"] = W[:, j]
    return cols


def _axis_label(tok: str) -> str:
    t = tok[0].lower()
    i = tok[1:]
    if t == "x":
        return f"x{i} [m]"
    if t == "v":
        return f"v{i} [m/s]"
    if t == "u":
        return f"u{i} [m/s\u00b2]"
    if t == "w":
        return f"w{i} [m/s\u00b2]"
    return tok


# ──── robust hull builders ────

def _hull2d_xy(x, y):
    P = np.unique(np.c_[x, y], axis=0)
    if P.shape[0] < 3:
        c = P.mean(axis=0) if P.size else np.zeros(2)
        eps = 1e-6 + 1e-3
        P = np.vstack([P, c + [eps, 0], c + [0, eps]])
    H = ConvexHull(P, qhull_options="QJ Pp")
    V = P[H.vertices]
    V = np.vstack([V, V[0]])
    return V[:, 0], V[:, 1]


def _hull3d_xyz(x, y, z):
    P = np.unique(np.c_[x, y, z], axis=0)
    if P.shape[0] < 4:
        c = P.mean(axis=0) if P.size else np.zeros(3)
        eps = 1e-6 + 1e-3
        pad = np.array([[eps, 0, 0], [0, eps, 0], [0, 0, eps], [eps, eps, eps]])
        P = np.vstack([P, c + pad])
    H = ConvexHull(P, qhull_options="QJ Pp Qt")
    tris = H.simplices
    return P[:, 0], P[:, 1], P[:, 2], tris[:, 0], tris[:, 1], tris[:, 2]


# ──── main builder ────

def build_interactive_explorer(
    history: list,
    G_vset,
    eng,
) -> str:
    """
    Build a self-contained HTML string with two Plotly figures (2-D + 3-D)
    and dropdown selectors for every token pair / triplet.

    Parameters
    ----------
    history : list[VSet]
        ``[Z_0, …, Z*]``; uses ``history[0]`` and ``history[-1]``.
    G_vset : VSet
        Constraint graph-set.
    eng : LiftedSetOpsGPU_NoHull

    Returns
    -------
    html : str
        Ready to pass to ``IPython.display.HTML(html)`` or write to a file.
    """
    Z0, Zs = history[0], history[-1]
    cols_G = _token_columns(G_vset, eng)
    cols_Z0 = _token_columns(Z0, eng)
    cols_Zs = _token_columns(Zs, eng)

    tokens = ["x1", "v1"]
    if eng.n >= 4:
        tokens += ["x2", "v2"]
    tokens += [f"u{i+1}" for i in range(eng.m)]
    tokens += [f"w{i+1}" for i in range(eng.w)]

    pairs = list(itertools.combinations(tokens, 2))
    trips = list(itertools.combinations(tokens, 3))

    # Precompute hulls
    H2 = {"G": {}, "Z0": {}, "Z*": {}}
    for a, b in pairs:
        k = f"{a}|{b}"
        gx, gy = _hull2d_xy(cols_G[a], cols_G[b])
        x0, y0 = _hull2d_xy(cols_Z0[a], cols_Z0[b])
        xs, ys = _hull2d_xy(cols_Zs[a], cols_Zs[b])
        H2["G"][k] = {"x": gx.tolist(), "y": gy.tolist()}
        H2["Z0"][k] = {"x": x0.tolist(), "y": y0.tolist()}
        H2["Z*"][k] = {"x": xs.tolist(), "y": ys.tolist()}

    H3 = {"G": {}, "Z0": {}, "Z*": {}}
    for a, b, c in trips:
        k = f"{a}|{b}|{c}"
        Xg, Yg, Zg, ig, jg, kg = _hull3d_xyz(cols_G[a], cols_G[b], cols_G[c])
        X0, Y0, Z0m, i0, j0, k0 = _hull3d_xyz(cols_Z0[a], cols_Z0[b], cols_Z0[c])
        Xs, Ys, Zsm, is_, js_, ks_ = _hull3d_xyz(cols_Zs[a], cols_Zs[b], cols_Zs[c])
        H3["G"][k] = {
            "x": Xg.tolist(), "y": Yg.tolist(), "z": Zg.tolist(),
            "i": ig.tolist(), "j": jg.tolist(), "k": kg.tolist(),
        }
        H3["Z0"][k] = {
            "x": X0.tolist(), "y": Y0.tolist(), "z": Z0m.tolist(),
            "i": i0.tolist(), "j": j0.tolist(), "k": k0.tolist(),
        }
        H3["Z*"][k] = {
            "x": Xs.tolist(), "y": Ys.tolist(), "z": Zsm.tolist(),
            "i": is_.tolist(), "j": js_.tolist(), "k": ks_.tolist(),
        }

    labels = {tok: _axis_label(tok) for tok in tokens}

    default_pair = ("x1", "v1") if ("x1", "v1") in pairs else pairs[0]
    default_trip = (
        ("x1", "u1", "w1")
        if set(["x1", "u1", "w1"]).issubset(tokens) and ("x1", "u1", "w1") in trips
        else trips[0]
    )
    key2 = "|".join(default_pair)
    key3 = "|".join(default_trip)

    # Build initial Plotly figures
    fig2d = go.Figure(
        [
            go.Scatter(
                x=H2["G"][key2]["x"], y=H2["G"][key2]["y"],
                fill="toself", mode="lines", line=dict(width=1),
                name="\U0001d4a2", fillcolor="rgba(46,134,171,0.20)",
            ),
            go.Scatter(
                x=H2["Z0"][key2]["x"], y=H2["Z0"][key2]["y"],
                fill="toself", mode="lines",
                line=dict(width=1.5, dash="dash"),
                name="Z\u2080", fillcolor="rgba(162,59,114,0.25)",
            ),
            go.Scatter(
                x=H2["Z*"][key2]["x"], y=H2["Z*"][key2]["y"],
                fill="toself", mode="lines", line=dict(width=2),
                name="Z*", fillcolor="rgba(241,143,1,0.35)",
            ),
        ]
    )
    fig2d.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, b=10, t=34),
        xaxis_title=labels[default_pair[0]],
        yaxis_title=labels[default_pair[1]],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.02),
    )

    b3G = H3["G"][key3]
    b3Z0 = H3["Z0"][key3]
    b3Zs = H3["Z*"][key3]
    fig3d = go.Figure(
        [
            go.Mesh3d(
                x=b3G["x"], y=b3G["y"], z=b3G["z"],
                i=b3G["i"], j=b3G["j"], k=b3G["k"],
                color="rgba(46,134,171,0.28)", name="\U0001d4a2",
                opacity=0.4, showscale=False,
            ),
            go.Mesh3d(
                x=b3Z0["x"], y=b3Z0["y"], z=b3Z0["z"],
                i=b3Z0["i"], j=b3Z0["j"], k=b3Z0["k"],
                color="rgba(162,59,114,0.35)", name="Z\u2080",
                opacity=0.5, showscale=False,
            ),
            go.Mesh3d(
                x=b3Zs["x"], y=b3Zs["y"], z=b3Zs["z"],
                i=b3Zs["i"], j=b3Zs["j"], k=b3Zs["k"],
                color="rgba(241,143,1,0.55)", name="Z*",
                opacity=0.55, showscale=False,
            ),
        ]
    )
    fig3d.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, b=10, t=34),
        scene=dict(
            xaxis_title=labels[default_trip[0]],
            yaxis_title=labels[default_trip[1]],
            zaxis_title=labels[default_trip[2]],
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.02),
    )

    payload = {
        "pairs": pairs,
        "trips": trips,
        "H2": H2,
        "H3": H3,
        "labels": labels,
        "defaults": {"pair": default_pair, "trip": default_trip},
        "fig2d": fig2d.to_dict(),
        "fig3d": fig3d.to_dict(),
    }

    # Assemble HTML (doubled braces for JS inside f-string)
    html = f"""
<div id="lifted-ui" style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif">
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:8px 0 12px 0">
    <div><b>2D pair:</b>
      <select id="pairSel" style="margin-left:6px;padding:2px 6px"></select>
      <button id="save2dPng" style="margin-left:8px">Save 2D PNG</button>
      <button id="save2dPdf" style="margin-left:6px">Save 2D PDF</button>
    </div>
    <div><b>3D triplet:</b>
      <select id="tripSel" style="margin-left:6px;padding:2px 6px"></select>
      <button id="save3dPng" style="margin-left:8px">Save 3D PNG</button>
      <button id="save3dPdf" style="margin-left:6px">Save 3D PDF</button>
    </div>
  </div>
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    <div id="fig2d" style="flex:1;min-width:420px;border:1px solid #eee;border-radius:10px;padding:6px"></div>
    <div id="fig3d" style="flex:1;min-width:420px;border:1px solid #eee;border-radius:10px;padding:6px"></div>
  </div>
  <div id="lifted-status" style="margin-top:8px;color:#555"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.33.0.min.js"></script>
<script>
const DATA = {json.dumps(payload)};
const pairSel = document.getElementById('pairSel');
const tripSel = document.getElementById('tripSel');
const statusEl = document.getElementById('lifted-status');

DATA.pairs.forEach(p => {{
  const opt = document.createElement('option');
  opt.value = JSON.stringify(p);
  opt.textContent = p[0] + ' , ' + p[1];
  const d = DATA.defaults.pair;
  if (p[0]===d[0] && p[1]===d[1]) opt.selected = true;
  pairSel.appendChild(opt);
}});
DATA.trips.forEach(t => {{
  const opt = document.createElement('option');
  opt.value = JSON.stringify(t);
  opt.textContent = t[0] + ' , ' + t[1] + ' , ' + t[2];
  const d = DATA.defaults.trip;
  if (t[0]===d[0] && t[1]===d[1] && t[2]===d[2]) opt.selected = true;
  tripSel.appendChild(opt);
}});

Plotly.newPlot('fig2d', DATA.fig2d.data, DATA.fig2d.layout, {{displaylogo:false,responsive:true}});
Plotly.newPlot('fig3d', DATA.fig3d.data, DATA.fig3d.layout, {{displaylogo:false,responsive:true}});

function update2D() {{
  const p = JSON.parse(pairSel.value), key = p.join('|');
  const g  = DATA.H2['G'][key], z0 = DATA.H2['Z0'][key], zs = DATA.H2['Z*'][key];
  Plotly.restyle('fig2d', {{x:[g.x, z0.x, zs.x], y:[g.y, z0.y, zs.y]}});
  Plotly.relayout('fig2d', {{
    'xaxis.title.text': DATA.labels[p[0]],
    'yaxis.title.text': DATA.labels[p[1]]
  }});
}}

function update3D() {{
  const t = JSON.parse(tripSel.value), key = t.join('|');
  const g  = DATA.H3['G'][key], z0 = DATA.H3['Z0'][key], zs = DATA.H3['Z*'][key];
  Plotly.restyle('fig3d', {{ x:[g.x], y:[g.y], z:[g.z], i:[g.i], j:[g.j], k:[g.k] }}, [0]);
  Plotly.restyle('fig3d', {{ x:[z0.x], y:[z0.y], z:[z0.z], i:[z0.i], j:[z0.j], k:[z0.k] }}, [1]);
  Plotly.restyle('fig3d', {{ x:[zs.x], y:[zs.y], z:[zs.z], i:[zs.i], j:[zs.j], k:[zs.k] }}, [2]);
  Plotly.relayout('fig3d', {{
    'scene.xaxis.title.text': DATA.labels[t[0]],
    'scene.yaxis.title.text': DATA.labels[t[1]],
    'scene.zaxis.title.text': DATA.labels[t[2]]
  }});
}}

pairSel.addEventListener('change', update2D);
tripSel.addEventListener('change', update3D);

const fname2d = () => {{ const p = JSON.parse(pairSel.value); return `lifted_${{p[0]}}_${{p[1]}}_2D`; }};
const fname3d = () => {{ const t = JSON.parse(tripSel.value); return `lifted_${{t[0]}}_${{t[1]}}_${{t[2]}}_3D`; }};

document.getElementById('save2dPng').onclick = () =>
  Plotly.downloadImage('fig2d', {{format:'png', width:1400, height:1100, filename: fname2d()}});
document.getElementById('save2dPdf').onclick = () =>
  Plotly.downloadImage('fig2d', {{format:'pdf', width:1400, height:1100, filename: fname2d()}});
document.getElementById('save3dPng').onclick = () =>
  Plotly.downloadImage('fig3d', {{format:'png', width:1600, height:1200, filename: fname3d()}});
document.getElementById('save3dPdf').onclick = () =>
  Plotly.downloadImage('fig3d', {{format:'pdf', width:1600, height:1200, filename: fname3d()}});

statusEl.textContent = 'Interactive view ready. Pick variables, rotate 3D, then Save.';
</script>
"""
    return html
