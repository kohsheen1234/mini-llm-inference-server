"""Generate docs/softmax.svg: an explanatory figure for numerically stable softmax.

Two panels:
  A) Why the naive version is unsafe: e^x overflows float64 for large logits.
  B) The stability trick: subtracting the max leaves the probabilities unchanged.

Pure Python (stdlib only) so it runs without matplotlib. Re-run to regenerate:

    python examples/plot_softmax.py
"""

import math
import os

# ---------------------------------------------------------------------------
# Small softmax helper (stdlib only, mirrors stable_softmax.py for 1-D input).
# ---------------------------------------------------------------------------


def stable_softmax_1d(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Tiny SVG builder.
# ---------------------------------------------------------------------------

INK = "#1f2933"
MUTE = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2563eb"
DANGER = "#dc2626"
SAFE = "#059669"
BARS = ["#2563eb", "#7c3aed", "#0891b2"]

W, H = 920, 440


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal", family="sans-serif"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{d} />'
    )


def rect(x, y, w, h, fill, rx=0, opacity=1.0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" rx="{rx}" opacity="{opacity}" />'
    )


def path(d, stroke, width=2.5, fill="none"):
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{width}" fill="{fill}" stroke-linecap="round" />'


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">',
    rect(0, 0, W, H, "#ffffff"),
    text(W / 2, 34, "Numerically stable softmax", size=22, anchor="middle", weight="700"),
]

# ---------------------------------------------------------------------------
# Panel A: exp() overflow (left).
# ---------------------------------------------------------------------------
ax0, ay0 = 70, 90          # top-left of plot area
aw, ah = 340, 250          # plot width / height
axb = ay0 + ah             # baseline y

FLOAT_MAX_EXP = 308.25     # log10 of largest finite float64 (~1.8e308)
OVERFLOW_X = math.log(10 ** FLOAT_MAX_EXP)  # logit where e^x overflows (~709.8)
XMAX = 1000.0
YMAX = 400.0               # log10 axis top

parts.append(text(ax0, ay0 - 16, "A.  Naive exp() overflows", size=15, weight="700"))
parts.append(text(ax0, ay0 - 1, "value of e^(logit), log scale", size=11, fill=MUTE))

# axes
parts.append(line(ax0, ay0, ax0, axb, stroke=INK, width=1.5))
parts.append(line(ax0, axb, ax0 + aw, axb, stroke=INK, width=1.5))

# y grid at powers of ten
for exp10 in range(0, int(YMAX) + 1, 100):
    gy = axb - (exp10 / YMAX) * ah
    parts.append(line(ax0, gy, ax0 + aw, gy, stroke=GRID, width=1))
    parts.append(text(ax0 - 8, gy + 4, f"1e{exp10}", size=10, fill=MUTE, anchor="end"))

# x ticks
for xv in (0, 250, 500, 750, 1000):
    gx = ax0 + (xv / XMAX) * aw
    parts.append(line(gx, axb, gx, axb + 5, stroke=INK, width=1))
    parts.append(text(gx, axb + 19, str(xv), size=10, fill=MUTE, anchor="middle"))
parts.append(text(ax0 + aw / 2, axb + 38, "logit value", size=12, fill=INK, anchor="middle"))

# float64 ceiling
ceil_y = axb - (FLOAT_MAX_EXP / YMAX) * ah
parts.append(line(ax0, ceil_y, ax0 + aw, ceil_y, stroke=DANGER, width=1.5, dash="6 4"))
parts.append(text(ax0 + 6, ceil_y - 6, "float64 max ≈ 1.8e308", size=10, fill=DANGER, anchor="start"))

# overflow region
ox = ax0 + (OVERFLOW_X / XMAX) * aw
parts.append(rect(ox, ay0, ax0 + aw - ox, ah, DANGER, opacity=0.08))
parts.append(line(ox, ay0, ox, axb, stroke=DANGER, width=1, dash="3 3"))

# e^x line (straight on log scale): log10(e^x) = x / ln(10)
safe_x_end = min(OVERFLOW_X, XMAX)
px0, py0 = ax0, axb
px1 = ax0 + (safe_x_end / XMAX) * aw
py1 = axb - ((safe_x_end / math.log(10)) / YMAX) * ah
parts.append(path(f"M {px0:.1f} {py0:.1f} L {px1:.1f} {py1:.1f}", ACCENT, width=3))
# dashed continuation into overflow
parts.append(
    f'<path d="M {px1:.1f} {py1:.1f} L {ox+ (ax0+aw-ox)*0.55:.1f} {ay0+6:.1f}" '
    f'stroke="{DANGER}" stroke-width="2.5" fill="none" stroke-dasharray="5 4" />'
)
parts.append(text(ox + (ax0 + aw - ox) / 2, ay0 + 30, "= inf", size=13, fill=DANGER, anchor="middle", weight="700"))
parts.append(text(ox + (ax0 + aw - ox) / 2, ay0 + 48, "→ NaN", size=12, fill=DANGER, anchor="middle"))

# ---------------------------------------------------------------------------
# Panel B: shift invariance (right).
# ---------------------------------------------------------------------------
bx0, by0 = 545, 90
bw, bh = 320, 250
byb = by0 + bh

raw = [2.0, 1.0, 0.1]
shifted = [v - max(raw) for v in raw]
probs = stable_softmax_1d(raw)   # identical for raw and shifted

parts.append(text(bx0, by0 - 16, "B.  Subtracting max is free", size=15, weight="700"))
parts.append(text(bx0, by0 - 1, "softmax(x) = softmax(x − max x)", size=11, fill=MUTE))

parts.append(line(bx0, by0, bx0, byb, stroke=INK, width=1.5))
parts.append(line(bx0, byb, bx0 + bw, byb, stroke=INK, width=1.5))
for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
    gy = byb - frac * bh
    parts.append(line(bx0, gy, bx0 + bw, gy, stroke=GRID, width=1))
    parts.append(text(bx0 - 8, gy + 4, f"{frac:.2f}", size=10, fill=MUTE, anchor="end"))
parts.append(
    f'<text x="{bx0 - 46:.1f}" y="{by0 + bh / 2:.1f}" font-family="sans-serif" font-size="12" '
    f'fill="{INK}" text-anchor="middle" transform="rotate(-90 {bx0 - 46:.1f} {by0 + bh / 2:.1f})">probability</text>'
)

groups = [("raw  [2, 1, 0.1]", raw), ("shifted  [0, -1, -1.9]", shifted)]
group_w = bw / len(groups)
bar_w = 26
gap = 8
labels = ["cat", "dog", "plane"]
for gi, (gname, _vals) in enumerate(groups):
    cluster_x = bx0 + gi * group_w + 30
    for bi, p in enumerate(probs):
        x = cluster_x + bi * (bar_w + gap)
        bh_px = p * bh
        parts.append(rect(x, byb - bh_px, bar_w, bh_px, BARS[bi], rx=3))
        parts.append(text(x + bar_w / 2, byb - bh_px - 6, f"{p:.3f}", size=10, fill=INK, anchor="middle", weight="600"))
        parts.append(text(x + bar_w / 2, byb + 15, labels[bi], size=10, fill=MUTE, anchor="middle"))
    parts.append(text(cluster_x + (3 * bar_w + 2 * gap) / 2 - 4, byb + 33, gname, size=11, fill=INK, anchor="middle", weight="600"))

# "identical" tie between clusters
tie_y = by0 + 26
c1 = bx0 + 0 * group_w + 30 + (3 * bar_w + 2 * gap) / 2 - 4
c2 = bx0 + 1 * group_w + 30 + (3 * bar_w + 2 * gap) / 2 - 4
parts.append(line(c1, tie_y, c2, tie_y, stroke=SAFE, width=1.5, dash="4 3"))
parts.append(text((c1 + c2) / 2, tie_y - 6, "identical output", size=12, fill=SAFE, anchor="middle", weight="700"))

parts.append("</svg>")

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "softmax.svg")
with open(out_path, "w") as f:
    f.write("\n".join(parts))

print(f"wrote {out_path}")
