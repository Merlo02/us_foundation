"""Schematic MAE diagram for UltrasonicMAE, in the clean vertical style of the
project's reference figure.

Left column (bottom -> top): raw signals -> Signal Patching -> Linear Embedding
-> Masking -> visible patches -> Encoder (8x) -> Decoder (4x) -> Linear Patch
Reconstruction Head -> reconstructed patches, with the reconstruction loss L
comparing input vs output patches. A single continuous A-mode waveform is drawn
across each patch row and segmented by the patch boxes (masked = empty box), so
the same signal can be followed from input -> masking -> reconstruction.

Right: a compact Transformer block (the real code block): LayerNorm -> MHSA with
CT-RoPE on Q/K -> residual -> LayerNorm -> MLP -> residual.

Output: utility/architecture_schematic.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# ----------------------------------------------------------------------
# Palette (matched to the reference figure)
# ----------------------------------------------------------------------
YEL = "#fdf6b2"
ORANGE = "#f08a24"
LAV = "#c9b8e8"
MAG = "#e3198f"
LN = "#f3ea9e"
MLP = "#edb8d6"
GREY = "#ececec"
EDGE = "#3a3a3a"
RED = "#d6453d"
BLUE = "#3f6fd1"

fig, ax = plt.subplots(figsize=(11.5, 12.8), dpi=200)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 13)
ax.axis("off")

CX = 3.0
CW = 4.0
xL = CX - CW / 2
xR = CX + CW / 2


def box(x, y, w, h, text, fc, fs=11, weight="bold", ec=EDGE, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color="#16202b", zorder=4)


def up_arrow(x, y0, y1, lw=2.0, color=EDGE):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=color, zorder=2))


def rf_burst(n=400, seed=0):
    """Synthetic damped RF burst in [-1, 1] with a few echoes (A-mode look)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    s = np.zeros(n)
    for c, w, d, sg in [(0.95, 60, 0.13, 0.045), (0.7, 85, 0.42, 0.05),
                        (0.5, 45, 0.68, 0.06), (0.35, 70, 0.88, 0.04)]:
        s += c * np.exp(-((t - d) ** 2) / (2 * sg ** 2)) * np.sin(w * (t - d))
    s += 0.03 * rng.standard_normal(n)
    return s / (np.abs(s).max() + 1e-9)


def draw_signal(x0, x1, yc, amp, color, seed=0, lw=1.1):
    s = rf_burst(seed=seed)
    x = np.linspace(x0, x1, s.size)
    ax.plot(x, yc + amp * s, color=color, lw=lw, zorder=4)


def patch_row(y, h, color, masked, seed, n=6):
    """One continuous waveform segmented by n dashed patch boxes.

    The signal is only drawn inside *visible* (non-masked) patches, so the
    masked positions show as empty boxes — exactly the MAE reconstruction view.
    """
    gap = 0.12
    pw = (CW - (n - 1) * gap) / n
    x0, x1 = xL + 0.05, xR - 0.05
    s = rf_burst(seed=seed)
    xs = np.linspace(x0, x1, s.size)
    ys = y + h / 2 + h * 0.36 * s
    for i in range(n):
        px = xL + i * (pw + gap)
        ax.add_patch(Rectangle((px, y), pw, h, facecolor="white",
                     edgecolor="#555", lw=1.1, linestyle=(0, (3, 2)), zorder=3))
        if i not in masked:
            seg = (xs >= px + 0.04) & (xs <= px + pw - 0.04)
            ax.plot(xs[seg], ys[seg], color=color, lw=0.95, zorder=4)


# ======================================================================
# LEFT COLUMN  (bottom -> top)  — red seed=1, blue seed=9 everywhere so the
# same waveform is followed through patching / masking / reconstruction.
# ======================================================================
RSEED, BSEED = 1, 9

# 1. raw input signals (continuous, no boxes)
draw_signal(xL + 0.1, xR - 0.1, 1.05, 0.5, RED, seed=RSEED)
draw_signal(xL + 0.1, xR - 0.1, 0.2, 0.45, BLUE, seed=BSEED)
ax.text(CX, 0.62, ". . .", ha="center", va="center", fontsize=14, color="#555")
ax.text(xR + 0.15, 0.6, "multi-dataset\nA-mode signals", ha="left",
        va="center", fontsize=8.5, style="italic", color="#555")
up_arrow(CX, 1.7, 2.25)

# 2. front-end stack  (CT-RoPE removed — it acts at attention time, see inset)
box(xL, 2.25, CW, 0.6, "Signal Patching", YEL, fs=11)
up_arrow(CX, 2.85, 3.0)
box(xL, 3.0, CW, 0.6, "Linear Embedding of Patches", YEL, fs=10.5)
up_arrow(CX, 3.6, 3.75)
box(xL, 3.75, CW, 0.6, "Masking", YEL, fs=11)
up_arrow(CX, 4.35, 4.6)

# 3. visible (post-masking) patch tokens — red upper, blue lower
patch_row(5.42, 0.78, RED, masked={2, 3, 5}, seed=RSEED)
patch_row(4.6, 0.78, BLUE, masked={1, 4}, seed=BSEED)
ax.text(xR + 0.15, 5.4, "visible patches\n(masked = empty)", ha="left",
        va="center", fontsize=8.5, style="italic", color="#555")
up_arrow(CX, 6.2, 6.5)

# 4. Encoder
box(xL, 6.5, CW, 1.4, "Encoder", ORANGE, fs=17)
ax.text(xL - 0.35, 7.2, r"8 $\times$", ha="right", va="center",
        fontsize=13, weight="bold", color="#333")
up_arrow(CX, 7.9, 8.15)

# 5. Decoder
box(xL, 8.15, CW, 0.95, "Decoder", LAV, fs=14)
ax.text(xL - 0.35, 8.62, r"4 $\times$", ha="right", va="center",
        fontsize=13, weight="bold", color="#333")
up_arrow(CX, 9.1, 9.35)

# 6. Reconstruction head
box(xL, 9.35, CW, 0.68, "Linear Patch Reconstruction Head", MAG, fs=10,
    ec="#9a0e60")
up_arrow(CX, 10.03, 10.3)

# 7. reconstructed patch tokens — fully reconstructed (no empty boxes)
patch_row(11.12, 0.78, RED, masked=set(), seed=RSEED)
patch_row(10.3, 0.78, BLUE, masked=set(), seed=BSEED)
ax.text(xR + 0.15, 11.1, "reconstructed\npatches", ha="left",
        va="center", fontsize=8.5, style="italic", color="#555")

# ----------------------------------------------------------------------
# Reconstruction loss L
# ----------------------------------------------------------------------
lx = xR + 1.4
for (ya, side) in [(11.5, "top"), (2.55, "bot")]:
    ax.add_patch(FancyArrowPatch((xR + 0.02, ya), (lx, ya), arrowstyle="-",
                 lw=1.6, color=EDGE))
ax.add_patch(FancyArrowPatch((lx, 11.5), (lx, 7.5), arrowstyle="-", lw=1.6,
             color=EDGE))
ax.add_patch(FancyArrowPatch((lx, 2.55), (lx, 6.6), arrowstyle="-", lw=1.6,
             color=EDGE))
ax.text(lx, 7.05, r"$\mathcal{L}$", ha="center", va="center", fontsize=28,
        color="#16202b")
ax.text(lx + 0.25, 6.35, "Smooth-L1 on\nmasked patches", ha="left",
        va="center", fontsize=8.3, style="italic", color="#555")

# ======================================================================
# RIGHT INSET  —  Transformer block (the real code block)
# ======================================================================
ix, iy, iw, ih = 8.3, 4.8, 4.7, 7.0
ax.add_patch(FancyBboxPatch((ix, iy), iw, ih,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             facecolor=GREY, edgecolor="#9aa3ac", lw=1.4, zorder=1))
ax.text(ix + iw / 2, iy + ih - 0.4, "Transformer block",
        ha="center", va="center", fontsize=12.5, weight="bold", color="#2f3a45")

# connect Encoder/Decoder to the inset
ax.add_patch(FancyArrowPatch((xR + 0.02, 7.2), (ix, 8.0), arrowstyle="-",
             lw=1.3, color="#9aa3ac", linestyle=(0, (4, 3))))

bx = ix + 1.45
bw = 1.9
bcx = bx + bw / 2

box(bx, 5.2, bw, 0.5, "Input", "#ffffff", fs=9.5, weight="normal")
box(bx, 6.0, bw, 0.5, "LayerNorm", LN, fs=9.0, weight="normal")
box(bx, 7.1, bw, 0.62, "Multi-Head Self-Attention\n(CT-RoPE on Q, K)*", LAV,
    fs=8.7, weight="bold")
box(bx, 8.5, bw, 0.5, "LayerNorm", LN, fs=9.0, weight="normal")
box(bx, 9.5, bw, 0.5, "MLP  (GELU, ratio 4)", MLP, fs=8.8, weight="bold")
box(bx, 10.7, bw, 0.5, "Output", "#ffffff", fs=9.5, weight="normal")

for a, b in [(5.7, 6.0), (6.5, 7.1), (7.72, 8.5), (9.0, 9.5), (10.0, 10.7)]:
    up_arrow(bcx, a, b, lw=1.5)


def plus(x, y):
    ax.add_patch(plt.Circle((x, y), 0.13, facecolor="white", edgecolor=EDGE,
                 lw=1.4, zorder=5))
    ax.text(x, y, "+", ha="center", va="center", fontsize=11, zorder=6)


sk_x = bx - 0.7
# residual 1: input -> around attention
plus(bcx, 8.25)
ax.plot([bcx, sk_x, sk_x, bcx], [5.7, 5.7, 8.25, 8.25], color=EDGE, lw=1.3,
        zorder=2)
# residual 2: after add1 -> around mlp
plus(bcx, 10.35)
ax.plot([bcx, sk_x, sk_x, bcx], [8.4, 8.4, 10.35, 10.35], color=EDGE, lw=1.3,
        zorder=2)

# ----------------------------------------------------------------------
# Title + asterisk footnote
# ----------------------------------------------------------------------
ax.text(0.1, 12.65, "UltrasonicMAE", ha="left", va="center", fontsize=16,
        weight="bold", color="#16202b")
ax.text(0.1, 12.25,
        "ViT-style Masked Autoencoder for 1-D ultrasound A-mode signals "
        "(He et al. 2021)", ha="left", va="center", fontsize=10.5,
        style="italic", color="#4a5a68")
ax.text(8.3, 4.35,
        "* CT-RoPE: positional encoding in continuous time in order to\n"
        "  handle signals acquired at different frequencies.",
        ha="left", va="top", fontsize=8.6, color="#555")

plt.tight_layout(pad=0.3)
out = "utility/architecture_schematic.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
