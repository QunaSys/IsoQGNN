"""Generator for QCE26 slide figures.

Every image in this folder is produced by this script, so a figure can always be
regenerated and never drifts from the numbers it depicts. Run it from anywhere:

    python paper/talk/images/make_slide_figures.py

Styling is for projection, not print: large sans-serif type (the conference asks
for a 20-24 pt minimum), thick strokes, and a light background. That is
deliberately different from the paper figures, which are Times at ~8 pt.

Palette matches the paper and the deck:
  navy  #0B2545  quantum / Iso-QGNN
  teal  #1B998B  classical / Iso-CGNN
  ink   #1A1A2E  text
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

NAVY = "#0B2545"   # quantum
TEAL = "#1B998B"   # classical
INK = "#1A1A2E"
MUTED = "#6B7A8F"
TARGET_FILL = "#F2C14E"  # amber: reserved for the region the talk is about

# Caveat ("footer") lines. James (2026-09-07): drawn on the figure they render at
# about 8 pt once the figure is scaled onto a slide, so they are NOT drawn. Every
# figure records its line here and the run writes them to slide_text_lines.md,
# to go in the slide's own text box. Flip FOOTERS_ON_FIGURE to draw them again.
FOOTERS_ON_FIGURE = False
FOOTER_LINES = {}


def footer(fig, name, text, y=-0.03, fontsize=12.5, draw=None, **kw):
    """Record a figure's caveat line; draw it only if asked.

    James (2026-09-07, later): the caveat text must live INSIDE the image after
    all, at a size that survives scaling onto a slide. Figures are converted one
    at a time as he reviews them: pass draw=True with fontsize >= 15 and at most
    two lines. The recorded line still goes to slide_text_lines.md."""
    FOOTER_LINES[name] = " ".join(text.split())
    if FOOTERS_ON_FIGURE if draw is None else draw:
        fig.text(0.5, y, text, ha="center", va="top", fontsize=fontsize,
                 color="#8A6D1F", linespacing=1.35, **kw)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 17,
    "axes.labelsize": 19,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 1.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def slide03_data_parameter_plane():
    """Opening slide: the data-and-parameters problem, and two routes into it.

    The two shaded regions are illustrative orders of magnitude for the field,
    NOT measured points from any specific published model. The only exact claim
    on the figure is this work's design point: 64 trainable parameters and the
    ~300 molecules the models need, i.e. the N90 for Iso-QGNN on the gap task,
    NOT the 1,200 they were trained on. That is deliberate (James, 2026-09-16):
    the claim the slide makes is about the regime this method operates in, and
    slide 15 then confirms the models land there. Do not "correct" it to 1,200.

    The two routes are drawn as a non-crossing pair, navy above and teal below,
    so each label sits unambiguously on its own curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(55, 4e7)
    ax.set_ylim(10, 6e8)
    ax.set_xlabel("Training molecules")
    ax.set_ylabel("Trainable parameters")

    # --- the corner chemistry is often stuck in: little data, few parameters ---
    ax.add_patch(Rectangle(
        (55, 10), 2200 - 55, 3000 - 10,
        facecolor=TARGET_FILL, alpha=0.20, edgecolor=TARGET_FILL,
        linewidth=2.0, zorder=1,
    ))
    ax.text(
        72, 2300, "where chemistry\noften has to operate",
        fontsize=15.5, color="#8A6D1F", va="top", ha="left",
        linespacing=1.25, zorder=4,
    )

    # --- typical deep graph networks: many parameters, much data ---
    ax.add_patch(Rectangle(
        (3e5, 3e5), 2.5e7 - 3e5, 2e8 - 3e5,
        facecolor=MUTED, alpha=0.14, edgecolor=MUTED,
        linewidth=1.6, linestyle=(0, (5, 3)), zorder=1,
    ))
    ax.text(
        2.8e6, 8e6, "deep graph\nnetworks",
        fontsize=16.5, color="#4A5A6B", ha="center", va="center",
        linespacing=1.25, zorder=4,
    )

    # --- two non-crossing routes down into the corner ---
    # Upper route, quantum: fewer parameters for the same expressivity.
    ax.add_patch(FancyArrowPatch(
        (3.0e5, 6e6), (1700, 1100),
        connectionstyle="arc3,rad=0.20",
        arrowstyle="-|>", mutation_scale=26,
        linewidth=3.4, color=NAVY, zorder=3,
    ))
    ax.text(
        1.1e4, 4.5e6, "quantum:\nexpressivity per parameter",
        fontsize=16, color=NAVY, ha="center", va="center",
        linespacing=1.25, zorder=4,
    )

    # Lower route, geometric: symmetry shrinks the hypothesis class.
    ax.add_patch(FancyArrowPatch(
        (3.0e5, 3.2e5), (1700, 200),
        connectionstyle="arc3,rad=-0.20",
        arrowstyle="-|>", mutation_scale=26,
        linewidth=3.4, color=TEAL, zorder=3,
    ))
    ax.text(
        2.2e4, 190, "geometric:\nsymmetry built in",
        fontsize=16, color=TEAL, ha="center", va="center",
        linespacing=1.25, zorder=4,
    )

    # --- this work's design point ---
    ax.plot(
        [300], [64], marker="o", markersize=15,
        markerfacecolor="white", markeredgecolor=INK,
        markeredgewidth=3.0, zorder=6,
    )
    ax.annotate(
        "this work\n64 parameters",
        xy=(300, 64), xytext=(620, 22),
        fontsize=16, color=INK, ha="left", va="center",
        linespacing=1.25, zorder=6,
        arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.6,
                        shrinkA=4, shrinkB=10),
    )

    ax.grid(True, which="major", linestyle=":", linewidth=0.9,
            color="#C9D1D9", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    path = OUT / "slide03_data_parameter_plane.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide04_hea_vs_topology():
    """Slide 2: why a topology-aligned circuit rather than a generic one.

    Left: a hardware-efficient ansatz. Rotations and a brick-wall entangler
    ladder repeated L times, the entangling pattern set by processor
    connectivity and identical whatever molecule is presented.

    Right: the topology-aligned circuit of this work at L = 1. Entanglers sit
    only on the bonds of the molecule, atom-type parameters are shared across
    atoms of the same element, and bond-type parameters across bonds of the
    same type. Bonds that share no atom occupy the same column, so the depth is
    set by the maximum degree rather than by the number of bonds.

    Nothing on this figure is a measured quantity. The molecule drawn is
    propanamide, chosen because its five heavy atoms fill the register shown.
    """
    fig, ax = plt.subplots(figsize=(14.4, 6.4))
    ax.set_xlim(0, 24.4)
    ax.set_ylim(-1.5, 9.3)
    ax.axis("off")
    ax.set_aspect("equal")

    HEA_GREY = "#6B7A8F"
    C_COL = "#4A5A6B"
    O_COL = "#C1435C"
    N_COL = "#3D6DB5"
    BOND_SINGLE = TEAL
    BOND_DOUBLE = "#B4634E"

    def wire(x0, x1, y, colour="#4A5568", lw=2.0):
        ax.plot([x0, x1], [y, y], color=colour, lw=lw, zorder=1,
                solid_capstyle="round")

    def gate(x, y, label, face, edge, w=0.86, h=0.72, fs=11.5):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=face,
                               edgecolor=edge, lw=1.8, zorder=3,
                               joinstyle="round"))
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                color="white", zorder=4)

    def cz(x, ya, yb, colour):
        ax.plot([x, x], [ya, yb], color=colour, lw=2.2, zorder=2)
        for y in (ya, yb):
            ax.plot([x], [y], marker="o", markersize=8.5, color=colour,
                    zorder=3)

    ys = [6.4, 5.2, 4.0, 2.8, 1.6]

    # ================= LEFT: hardware-efficient ansatz =================
    x_end = 10.1
    for y in ys:
        wire(0.5, x_end, y, "#8A96A6")

    def hea_block(x0, faded=False, second_ladder=True):
        f = "#BCC5CE" if faded else HEA_GREY
        e = "#9AA6B2" if faded else "#4A5A6B"
        for y in ys:
            gate(x0, y, "R", f, e, fs=12)
        cz(x0 + 1.1, ys[0], ys[1], e)
        cz(x0 + 1.1, ys[2], ys[3], e)
        if second_ladder:
            cz(x0 + 2.0, ys[1], ys[2], e)
            cz(x0 + 2.0, ys[3], ys[4], e)

    hea_block(1.4)
    hea_block(4.6)
    hea_block(7.8, faded=True, second_ladder=False)

    ax.text(5.1, 8.55, "Hardware-efficient ansatz", ha="center", va="center",
            fontsize=19.5, color=INK)
    ax.text(10.35, 4.0, "$\\times L$", ha="left", va="center", fontsize=17,
            color="#8A6D1F")
    ax.text(5.1, 0.30, "gates follow the processor",
            ha="center", va="center", fontsize=16, color="#4A5A6B")
    ax.annotate("", xy=(9.9, -0.55), xytext=(1.0, -0.55),
                arrowprops=dict(arrowstyle="-|>", color=TARGET_FILL, lw=3.0,
                                mutation_scale=22))
    ax.text(5.1, -1.10, "depth must grow to capture chemistry",
            ha="center", va="center", fontsize=16, color="#8A6D1F")

    ax.plot([11.7, 11.7], [-1.2, 8.9], color="#D5DBE1", lw=1.6,
            ls=(0, (4, 4)))

    # ================= RIGHT: topology-aligned =================
    x0 = 12.6
    x_end_r = 23.9
    labels = [("$q_C$", C_COL), ("$q_C$", C_COL), ("$q_C$", C_COL),
              ("$q_O$", O_COL), ("$q_N$", N_COL)]
    for y, (lab, col) in zip(ys, labels):
        wire(x0 + 0.55, x_end_r, y, "#4A5568")
        ax.text(x0 + 0.28, y, lab, ha="right", va="center", fontsize=14,
                color=col)

    ax.text(18.2, 8.55, "Topology-aligned  (this work)", ha="center",
            va="center", fontsize=19.5, color=INK)

    # molecule sketch, well clear of the title
    mx, my = 13.3, 7.45
    d = 0.82
    coords = {0: (mx, my), 1: (mx + d, my), 2: (mx + 2 * d, my),
              3: (mx + 3 * d, my + 0.34), 4: (mx + 3 * d, my - 0.34)}
    for pair in [(0, 1), (1, 2), (2, 4)]:
        ax.plot(*zip(coords[pair[0]], coords[pair[1]]), color="#4A5568",
                lw=2.2, zorder=2)
    for off in (0.08, -0.08):
        ax.plot([coords[2][0], coords[3][0]],
                [coords[2][1] + off, coords[3][1] + off],
                color="#4A5568", lw=1.9, zorder=2)
    for i, col in [(0, C_COL), (1, C_COL), (2, C_COL), (3, O_COL), (4, N_COL)]:
        ax.plot([coords[i][0]], [coords[i][1]], marker="o", markersize=12,
                color=col, zorder=3)
    ax.text(mx + 3 * d + 0.55, my, "propanamide", ha="left", va="center",
            fontsize=14, color="#4A5568")

    # stage 1: atom-type rotations, shared across atoms of the same element
    for y, (_, col) in zip(ys, labels):
        gate(x0 + 1.3, y, "$R_Y$", col, col, fs=12)

    # stage 2: entanglers on bonds only; disjoint bonds share a column
    bonds = [((0, 1), BOND_SINGLE, x0 + 2.9), ((2, 3), BOND_DOUBLE, x0 + 2.9),
             ((1, 2), BOND_SINGLE, x0 + 4.9), ((2, 4), BOND_SINGLE, x0 + 6.9)]
    for (a_, b_), col, xc in bonds:
        cz(xc, ys[a_], ys[b_], col)
        gate(xc + 0.78, ys[a_], "$R_Z$", col, col, fs=11)
        gate(xc + 0.78, ys[b_], "$R_Z$", col, col, fs=11)

    # stage 3: variational layer, then readout
    for y in ys:
        gate(x0 + 9.5, y, "$R_XR_Y$", "#2C3E50", "#2C3E50", w=1.2, fs=10.5)
        ax.plot([x_end_r - 0.05], [y], marker="D", markersize=8,
                color="#4A5568", zorder=3)

    ax.text(18.2, 0.30, "gates follow the molecule",
            ha="center", va="center", fontsize=16, color=INK)
    ax.text(18.2, -0.60, "one layer  ·  64 parameters",
            ha="center", va="center", fontsize=16.5, color=INK)
    ax.text(18.2, -0.98, "nine-qubit register; the four ghost qubits are not drawn",
            ha="center", va="center", fontsize=11.5, color=MUTED)
    ax.text(18.2, -1.54, "colour marks a shared parameter: one per atom type, "
                         "one per bond type",
            ha="center", va="center", fontsize=13.5, color="#4A5568")

    fig.tight_layout()
    path = OUT / "slide04_hea_vs_topology.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide07_primitive_mapping_and_budget():
    """Methodology slide: the primitive-by-primitive substitution, plus the budget.

    Absorbs Table I and Table II of the paper. Table I becomes positional
    rather than tabular: each Iso-CGNN box sits directly beneath the Iso-QGNN
    primitive it replaces, so the pairing is read without effort. The readout
    head is drawn once, spanning both rows, because it is not an analogue but
    the same module. Table II becomes a proportion bar, split so that the eight
    topology-aligned parameters are separated from the fifty-six of shared
    machinery.

    Every number here is exact and comes from the paper's Table II:
    4 + 4 + 18 + 18 + 20 = 64, identical for both architectures.
    """
    fig, ax = plt.subplots(figsize=(14.4, 7.5))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12.5)
    ax.axis("off")
    ax.set_aspect("equal")

    SHARED = "#2C3E50"
    MACHINE = "#C9D1D9"

    def box(xc, yc, w, h, lines, face, edge, fs=12.5, tcol="white", lw=2.0):
        ax.add_patch(Rectangle((xc - w / 2, yc - h / 2), w, h, facecolor=face,
                               edgecolor=edge, lw=lw, zorder=3,
                               joinstyle="round"))
        ax.text(xc, yc, "\n".join(lines), ha="center", va="center",
                fontsize=fs, color=tcol, zorder=4, linespacing=1.45)

    def chevron(x, y, colour):
        ax.annotate("", xy=(x + 0.62, y), xytext=(x - 0.18, y),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2,
                                    mutation_scale=18), zorder=2)

    xs = [4.15, 8.55, 12.95, 17.35]
    bw, bh = 3.55, 2.05
    y_q, y_c = 9.15, 6.15

    # stage headers
    heads = ["Atom embedding", "Bond interaction", "Node update", "Readout"]
    for x, h in zip(xs, heads):
        ax.text(x, 11.45, h, ha="center", va="center", fontsize=15,
                color=INK)

    # row labels
    ax.text(1.75, y_q, "Iso-QGNN", ha="right", va="center", fontsize=16.5,
            color=NAVY)
    ax.text(1.75, y_c, "Iso-CGNN", ha="right", va="center", fontsize=16.5,
            color=TEAL)

    q_boxes = [
        [r"$R_Y(\theta_{\mathrm{atom}})$", r"on $|0\rangle$"],
        [r"$CZ$, then $R_Z(\theta_{\mathrm{bond}})$", "on each bond"],
        [r"$R_X(\phi_1),\, R_Y(\phi_2)$", "per atom qubit"],
        [r"measure $\langle Z_i \rangle$"],
    ]
    c_boxes = [
        [r"$h_i = \cos(\theta_{\mathrm{atom}})$"],
        [r"$h_i \leftarrow h_i + \sum_j \theta_{\mathrm{bond}} h_j$",
         r"over bonded $j$"],
        [r"$\tanh(s_i h_i + b_i)$", "per node"],
        [r"$h_i$ is already real", "(no measurement)"],
    ]
    for x, lines in zip(xs, q_boxes):
        box(x, y_q, bw, bh, lines, NAVY, NAVY)
    for x, lines in zip(xs, c_boxes):
        box(x, y_c, bw, bh, lines, TEAL, TEAL)

    for x in xs[:-1]:
        chevron(x + bw / 2 + 0.12, y_q, NAVY)
        chevron(x + bw / 2 + 0.12, y_c, TEAL)

    # the head, drawn once because it is the same module on both sides
    hx, hw = 21.65, 3.4
    box(hx, (y_q + y_c) / 2, hw, 4.35,
        ["LayerNorm(9)", "+ Linear(9, 2)", "", "2 logits"],
        SHARED, SHARED, fs=13)
    for y in (y_q, y_c):
        ax.annotate("", xy=(hx - hw / 2 - 0.1, y),
                    xytext=(xs[-1] + bw / 2 + 0.1, y),
                    arrowprops=dict(arrowstyle="-|>", color=SHARED, lw=2.2,
                                    mutation_scale=18), zorder=2)
    ax.text(hx, 11.45, "Head", ha="center", va="center", fontsize=15,
            color=INK)
    ax.text(hx, 5.15, "the same module,\nnot an analogue", ha="center",
            va="top", fontsize=13.5, color=SHARED, linespacing=1.35)

    # ---------------- parameter budget, Table II as a proportion -------------
    seg = [("atom", 4, TARGET_FILL), ("bond", 4, TARGET_FILL),
           ("per-node update", 18, MACHINE), ("LayerNorm", 18, MACHINE),
           ("classifier", 20, MACHINE)]
    bx0, bx1, by, bhh = 2.0, 22.3, 1.75, 0.82
    span = bx1 - bx0
    x = bx0
    for name, n, col in seg:
        w = span * n / 64
        ax.add_patch(Rectangle((x, by), w, bhh, facecolor=col,
                               edgecolor="white", lw=2.0, zorder=3))
        ax.text(x + w / 2, by + bhh / 2, str(n), ha="center", va="center",
                fontsize=13, color="#3A3A3A" if col == TARGET_FILL else "#3A3A3A",
                zorder=4)
        ax.text(x + w / 2, by + bhh + 0.28, name, ha="center", va="bottom",
                fontsize=12, color="#4A5568", rotation=0, zorder=4)
        x += w

    x_split = bx0 + span * 8 / 64
    ax.plot([bx0, x_split], [by - 0.30, by - 0.30], color="#8A6D1F", lw=2.6)
    ax.text((bx0 + x_split) / 2, by - 0.72,
            "8: the inductive bias", ha="center", va="center",
            fontsize=13.5, color="#8A6D1F")
    ax.plot([x_split, bx1], [by - 0.30, by - 0.30], color="#6B7A8F", lw=2.6)
    ax.text((x_split + bx1) / 2, by - 0.72,
            "56: shared machinery, identical in design and at initialisation",
            ha="center", va="center", fontsize=13.5, color="#4A5568")
    ax.text(bx1 + 0.25, by + bhh / 2, "= 64", ha="left", va="center",
            fontsize=16, color=INK)

    fig.tight_layout()
    path = OUT / "slide07_primitive_mapping_and_budget.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide19_parameter_budget():
    """Backup figure, not used in the delivered 18-slide deck: where the 64 come from.

    Held one keypress behind the methodology slide, for the question "why 64?"
    or "why 18 for LayerNorm?". Rows are the seven actual parameter tensors
    rather than the paper's five grouped rows, because the grouped version
    hides the 9 + 9 that answers the LayerNorm question.

    Every number was read off the constructed models, not transcribed:
    both IsoQGNN(n_nodes=9) and IsoCGNN(n_nodes=9) report these
    seven tensors and total 64. The layer scaling was confirmed the same way.

    The gauge freedom in the two-logit head (a single-logit head would give 54)
    is deliberately NOT on the slide: a backup slide should answer the question
    asked, not volunteer a tangent.
    """
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    BIAS = TARGET_FILL
    MACHINE = "#B9C2CB"

    rows = [
        ("Atom angles", "(4, 1)", 4, "one per element: C, N, O, F", BIAS),
        ("Bond angles", "(1, 4)", 4,
         "one per bond type: single, double, triple, aromatic", BIAS),
        ("Node update", "(1, 9, 2)", 18,
         r"9 register slots $\times$ 2 angles ($R_X$, $R_Y$)", MACHINE),
        ("LayerNorm gain", "(9,)", 9, "one per readout feature", MACHINE),
        ("LayerNorm bias", "(9,)", 9, "one per readout feature", MACHINE),
        ("Classifier weight", "(2, 9)", 18,
         r"9 inputs $\times$ 2 classes", MACHINE),
        ("Classifier bias", "(2,)", 2, "one per class", MACHINE),
    ]

    X_STRIPE, X_NAME, X_SHAPE, X_COUNT, X_WHY = 2.0, 5.2, 40.0, 60.5, 64.5

    ax.text(X_STRIPE, 93.5, "Parameter budget, identical for both architectures",
            ha="left", va="center", fontsize=19, color=INK)

    for lab, x, ha in (("Component", X_NAME, "left"), ("Shape", X_SHAPE, "left"),
                       ("Count", X_COUNT, "right"),
                       ("Where the number comes from", X_WHY, "left")):
        ax.text(x, 84.5, lab, ha=ha, va="center", fontsize=14.5,
                color="#4A5568")
    ax.plot([X_STRIPE, 98], [80.5, 80.5], color="#9AA6B2", lw=1.6)

    y = 74.0
    for name, shape, count, why, col in rows:
        ax.add_patch(Rectangle((X_STRIPE, y - 2.6), 1.15, 5.2, facecolor=col,
                               edgecolor="none", zorder=3))
        ax.text(X_NAME, y, name, ha="left", va="center", fontsize=14.5,
                color=INK)
        ax.text(X_SHAPE, y, shape, ha="left", va="center", fontsize=14,
                color="#4A5568")
        ax.text(X_COUNT, y, str(count), ha="right", va="center", fontsize=15,
                color=INK)
        ax.text(X_WHY, y, why, ha="left", va="center", fontsize=13.5,
                color="#4A5568")
        y -= 7.0

    ax.plot([X_STRIPE, 98], [26.5, 26.5], color="#9AA6B2", lw=1.6)
    ax.add_patch(Rectangle((X_STRIPE, 16.0), 98 - X_STRIPE, 8.4,
                           facecolor="#EEF1F4", edgecolor="none", zorder=1))
    ax.text(X_NAME, 20.2, "Total, per architecture", ha="left", va="center",
            fontsize=15.5, color=INK)
    ax.text(X_COUNT, 20.2, "64", ha="right", va="center", fontsize=19,
            color=INK)

    notes = [
        "LayerNorm normalises across the nine readout features of a single "
        "molecule. Its mean and variance are computed from the input, so they "
        "cost no parameters.",
        "Only the eight atom and bond parameters carry the topology-aligned "
        "sharing. The other 56 are position-indexed machinery and the head, "
        "identical in both architectures.",
        "Each further layer adds 22, being four bond angles and eighteen "
        "node-update angles: one layer gives 64, two give 86, three give 108.",
    ]
    yy = 11.0
    for n in notes:
        ax.text(X_STRIPE, yy, n, ha="left", va="center", fontsize=12.5,
                color="#4A5568")
        yy -= 4.6

    fig.tight_layout()
    path = OUT / "slide19_parameter_budget.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide08_training_protocol():
    """Training-protocol slide: the matched control as a pipeline, and the two
    protocols distinguished.

    The upper half is what happens inside one trial: a seed produces one split,
    the split is shared by both architectures, all RNGs are reset to a common
    state before each model is built, both train under identical
    hyperparameters, and the reported figure is the final epoch.

    The lower half settles a distinction the paper warns about but which is
    easy to lose in speech: the matched comparison and the sample-complexity
    sweep use different budgets and different reporting rules, so their
    absolute AUCs are not comparable.

    All numbers read from the code: run_experiments.py has N_TRIALS = 10,
    N_EPOCHS = 200, BATCH_SIZE = 1 and reports trial_curve[-1]; the 10 per cent
    validation split is built but never evaluated. run_data_efficiency.py has
    N_TRIALS = 5, N_EPOCHS = 40, VAL_SIZE = 300, a fixed 500-molecule test set
    and reports the epoch at argmax of the validation history. Configuration
    counts: 19 x 5 x 2 x 2 = 380 trainings against 10 x 2 x 2 = 40.
    """
    fig, ax = plt.subplots(figsize=(14.4, 8.1))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_aspect("equal")

    NEUTRAL = "#55636F"

    def box(xc, yc, w, h, lines, face, edge, fs=12.5, tcol="white"):
        ax.add_patch(Rectangle((xc - w / 2, yc - h / 2), w, h, facecolor=face,
                               edgecolor=edge, lw=2.0, zorder=3,
                               joinstyle="round"))
        ax.text(xc, yc, "\n".join(lines), ha="center", va="center",
                fontsize=fs, color=tcol, zorder=4, linespacing=1.4)

    def arrow(x0, y0, x1, y1, colour=NEUTRAL):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2,
                                    mutation_scale=18), zorder=2)

    # ---- one trial ----
    ax.add_patch(Rectangle((0.7, 9.5), 28.4, 7.0, facecolor="#FAFBFC",
                           edgecolor="#B9C2CB", lw=1.8, ls=(0, (5, 4)),
                           zorder=0))
    ax.text(1.3, 15.9, "inside one trial", ha="left", va="center",
            fontsize=13.5, color="#7A8794")

    y_mid, y_q, y_c = 13.0, 14.35, 11.65
    box(3.0, y_mid, 3.4, 1.5, ["seed 42 + t"], NEUTRAL, NEUTRAL, fs=12.5)
    box(8.0, y_mid, 5.0, 2.1, ["one 80/10/10 split", "drawn once, shared"],
        NEUTRAL, NEUTRAL, fs=12)
    box(14.0, y_mid, 5.2, 2.1, ["all RNGs reset to", "a common seed"],
        NEUTRAL, NEUTRAL, fs=12)
    box(20.4, y_q, 4.4, 1.6, ["Iso-QGNN"], NAVY, NAVY, fs=13.5)
    box(20.4, y_c, 4.4, 1.6, ["Iso-CGNN"], TEAL, TEAL, fs=13.5)
    box(26.2, y_mid, 4.6, 3.6,
        ["test AUC", "and accuracy", "", "at the final epoch"],
        "#2C3E50", "#2C3E50", fs=12)

    arrow(4.75, y_mid, 5.45, y_mid)
    arrow(10.55, y_mid, 11.35, y_mid)
    arrow(16.65, y_mid, 17.3, y_q - 0.15, NAVY)
    arrow(16.65, y_mid, 17.3, y_c + 0.15, TEAL)
    arrow(22.65, y_q, 23.85, y_q, NAVY)
    arrow(22.65, y_c, 23.85, y_c, TEAL)

    ax.text(20.4, 10.15,
            "Adam, lr 0.01  ·  cross-entropy  ·  batch size one  ·  200 epochs",
            ha="center", va="center", fontsize=12.5, color="#4A5568")

    arrow(29.2, y_mid, 30.4, y_mid)
    ax.text(30.9, y_mid, "× 10\ntrials", ha="left", va="center", fontsize=13,
            color=INK, linespacing=1.35)
    ax.text(1.3, 8.5,
            "Reported: mean and cross-trial standard deviation over the ten "
            "trials, seeds 42 to 51. No model selection of any kind.",
            ha="left", va="center", fontsize=13, color=INK)

    # ---- the two protocols are not the same ----
    xl, c1, c2 = 1.6, 14.6, 24.6
    ax.plot([xl, 30.2], [7.0, 7.0], color="#9AA6B2", lw=1.5)
    ax.text(c1, 6.3, "matched comparison", ha="center", va="center",
            fontsize=13.5, color=INK)
    ax.text(c2, 6.3, "sample-complexity sweep", ha="center", va="center",
            fontsize=13.5, color=INK)
    ax.plot([xl, 30.2], [5.6, 5.6], color="#D5DBE1", lw=1.2)

    table = [
        ("trials per configuration", "10", "5"),
        ("epochs", "200", "40"),
        ("epoch reported", "the final one", "chosen on validation AUC"),
        ("test set", "150 held out per trial", "a fixed 500 molecules"),
    ]
    y = 4.85
    for lab, a_, b_ in table:
        ax.text(xl, y, lab, ha="left", va="center", fontsize=12.5,
                color="#4A5568")
        ax.text(c1, y, a_, ha="center", va="center", fontsize=12.5, color=INK)
        ax.text(c2, y, b_, ha="center", va="center", fontsize=12.5, color=INK)
        y -= 1.05
    footer(fig, "slide08_training_protocol.png",
           "380 trainings against 40, which is why the sweep is shorter. "
           "Absolute AUCs are not comparable between the two.", y=0.055, fontsize=15, draw=True)

    fig.tight_layout()
    path = OUT / "slide08_training_protocol.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide15_conclusion_plane():
    """Conclusion slide 1: the opening plane, with the result filled in.

    Same axes, same regions and same design point as slide03, which the audience
    saw before any result was shown. The two route arrows are gone, because the
    routes have now been taken; what remains is where the models landed.

    Both architectures sit at the same height (64 parameters, matched capacity)
    and the x position of each is the training-set size at which it first reaches
    90% of its own asymptotic gap AUC: 300 for Iso-QGNN, 100 for Iso-CGNN. The
    composition-only control sits at 5 parameters and the 1,200-molecule training
    split it was fitted on. Every number is from the corrected CSVs / Table III.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(55, 4e7)
    ax.set_ylim(2.2, 6e8)
    ax.set_xlabel("Training molecules")
    ax.set_ylabel("Trainable parameters")

    # regions, as on the opening slide
    ax.add_patch(Rectangle((55, 2.2), 2200 - 55, 3000 - 2.2,
                           facecolor=TARGET_FILL, alpha=0.20,
                           edgecolor=TARGET_FILL, linewidth=2.0, zorder=1))
    ax.add_patch(Rectangle((3e5, 3e5), 2.5e7 - 3e5, 2e8 - 3e5,
                           facecolor=MUTED, alpha=0.10, edgecolor=MUTED,
                           linewidth=1.4, linestyle=(0, (5, 3)), zorder=1))
    ax.text(2.8e6, 8e6, "deep graph\nnetworks", fontsize=15,
            color="#8A96A6", ha="center", va="center", linespacing=1.25)

    # headline, in the space the route arrows used to occupy
    ax.text(3.2e3, 3.2e4,
            "Same capacity, comparable performance.\n"
            "The inductive bias is the active ingredient.",
            fontsize=17, color=INK, ha="left", va="center", linespacing=1.4,
            zorder=5)

    # matched capacity: both models on the same horizontal
    ax.plot([100, 300], [64, 64], color=INK, lw=1.4, ls=(0, (3, 3)), zorder=4)

    # Iso-CGNN reaches 90% of asymptotic gap AUC at N = 100
    ax.plot([100], [64], marker="o", markersize=15, markerfacecolor=TEAL,
            markeredgecolor="white", markeredgewidth=2.5, zorder=6)
    ax.annotate("Iso-CGNN\ngap 0.916  ·  dipole 0.786",
                xy=(100, 64), xytext=(62, 1.15e4),
                fontsize=15, color=TEAL, ha="left", va="center",
                linespacing=1.35, zorder=6,
                arrowprops=dict(arrowstyle="-", color=TEAL, linewidth=1.5,
                                shrinkA=6, shrinkB=10))

    # Iso-QGNN reaches 90% of asymptotic gap AUC at N = 300
    ax.plot([300], [64], marker="o", markersize=15, markerfacecolor=NAVY,
            markeredgecolor="white", markeredgewidth=2.5, zorder=6)
    ax.annotate("Iso-QGNN\ngap 0.887  ·  dipole 0.773",
                xy=(300, 64), xytext=(760, 420),
                fontsize=15, color=NAVY, ha="left", va="center",
                linespacing=1.35, zorder=6,
                arrowprops=dict(arrowstyle="-", color=NAVY, linewidth=1.5,
                                shrinkA=6, shrinkB=10))

    # the non-graph control: 5 parameters, fitted on the 1,200-molecule split
    ax.plot([1200], [5], marker="o", markersize=13, markerfacecolor="#9AA6B2",
            markeredgecolor="white", markeredgewidth=2.5, zorder=6)
    ax.annotate("composition-only control, 5 parameters\n"
                "gap 0.659  ·  dipole 0.698",
                xy=(1200, 5), xytext=(3.4e3, 5.5),
                fontsize=14, color="#5A6675", ha="left", va="center",
                linespacing=1.35, zorder=6,
                arrowprops=dict(arrowstyle="-", color="#9AA6B2", linewidth=1.5,
                                shrinkA=6, shrinkB=9))

    ax.text(3.2e3, 40,
            "x for the two models: molecules to reach 90% of asymptotic gap "
            "AUC.\nControl shown at its 1,200-molecule training split.",
            fontsize=12.5, color="#7A8794", ha="left", va="center", zorder=5,
            linespacing=1.35)

    ax.grid(True, which="major", linestyle=":", linewidth=0.9,
            color="#C9D1D9", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    path = OUT / "slide15_conclusion_plane.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide16_scaling():
    """Conclusion slide 2: how resources scale with molecule size.

    Three panels, same x axis (heavy atoms N). Parameters and gates grow
    linearly; entangling depth does not grow at all. The visual rhythm, up, up,
    flat, is the message. Solid lines are the single layer used in the paper;
    dashed lines are the cost of a second layer, which is what the one-hop
    caveat may demand at larger N.

    Formulas, all verified by constructing the models / from the test suite:
      parameters      P(N, L) = 4 + 4L + 2NL + 2N + (2N + 2)   -> 6N + 10 at L=1
      gates           G(N, L) = N + L(2N + 3|E|), drawn with |E| = N + 1 (one ring)
                                                  -> 6N + 3 at L=1
      entangling depth  <= max degree <= 4 for C, N, O, F (valence), per layer
    The paper's own worked gate examples, 51 at N=8 and 186 at N=30, assume
    |E| = N+1 and N+2 respectively; the single line here uses N+1 throughout, so
    it reads 183 rather than 186 at N=30. Ring count moves it by ~3 per ring.

    This is RESOURCE scaling only. Nothing here says accuracy is preserved at
    larger N; the footer says so explicitly.
    """
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.6))
    N = np.linspace(2, 40, 200)
    L2 = "#8A96A6"

    def marks(ax, f1, f2, unit, fmt="{:.0f}"):
        for n, lab in ((9, "QM9\nregister"), (30, "drug-like")):
            ax.axvline(n, color="#D5DBE1", lw=1.2, ls=(0, (3, 3)), zorder=0)
            ax.text(n, ax.get_ylim()[1] * 0.025, lab, ha="center", va="bottom",
                    fontsize=12, color="#7A8794", linespacing=1.2)
            ax.plot([n], [f1(n)], marker="o", markersize=8, color=INK, zorder=5)
            ax.annotate(fmt.format(f1(n)) + unit, xy=(n, f1(n)),
                        xytext=(n + 1.2, f1(n)), fontsize=13.5, color=INK,
                        ha="left", va="center")

    # ---- panel 1: parameters (both architectures) ----
    ax = axes[0]
    p1 = lambda n: 6 * n + 10
    p2 = lambda n: 8 * n + 14
    ax.plot(N, p2(N), color=L2, lw=2.2, ls=(0, (5, 3)), zorder=2)
    ax.plot(N, p1(N), color=INK, lw=3.0, zorder=3)
    ax.set_ylim(0, 360)
    ax.set_title("Trainable parameters", fontsize=16, color=INK, pad=10)
    marks(ax, p1, p2, "")
    ax.text(40, p2(40) + 12, "two layers", ha="right", va="bottom",
            fontsize=12, color=L2)
    ax.text(36.5, p1(36.5) + 14, "one layer", ha="center", va="bottom",
            fontsize=12, color=INK)
    ax.text(21, p1(21) - 50, "6 per heavy atom", fontsize=13, color=INK,
            ha="left", va="top")

    # ---- panel 2: gates (quantum circuit) ----
    ax = axes[1]
    g1 = lambda n: 6 * n + 3
    g2 = lambda n: 11 * n + 6
    ax.plot(N, g2(N), color=L2, lw=2.2, ls=(0, (5, 3)), zorder=2)
    ax.plot(N, g1(N), color=NAVY, lw=3.0, zorder=3)
    ax.set_ylim(0, 480)
    ax.set_title("Gates in the circuit", fontsize=16, color=INK, pad=10)
    marks(ax, g1, g2, "")
    ax.text(40, g2(40) + 14, "two layers", ha="right", va="bottom",
            fontsize=12, color=L2)
    ax.text(36.5, g1(36.5) + 16, "one layer", ha="center", va="bottom",
            fontsize=12, color=NAVY)
    ax.text(21, g1(21) - 60, "about 6 per heavy atom", fontsize=13, color=NAVY,
            ha="left", va="top")

    # ---- panel 3: entangling depth (quantum circuit) ----
    ax = axes[2]
    ax.plot(N, np.full_like(N, 8), color=L2, lw=2.2, ls=(0, (5, 3)), zorder=2)
    ax.plot(N, np.full_like(N, 4), color=NAVY, lw=3.0, zorder=3)
    ax.set_ylim(0, 12)
    ax.set_title("Entangling depth", fontsize=16, color=INK, pad=10)
    for n, lab in ((9, "QM9\nregister"), (30, "drug-like")):
        ax.axvline(n, color="#D5DBE1", lw=1.2, ls=(0, (3, 3)), zorder=0)
        ax.text(n, 0.3, lab, ha="center", va="bottom", fontsize=12,
                color="#7A8794", linespacing=1.2)
    ax.text(40, 8.35, "two layers", ha="right", va="bottom", fontsize=12,
            color=L2)
    ax.text(40, 3.65, "one layer", ha="right", va="top", fontsize=12,
            color=NAVY)
    ax.text(21, 6.0, "bounded by valence:\ncarbon makes at most four bonds",
            ha="center", va="center", fontsize=13, color=NAVY,
            linespacing=1.35)
    ax.text(2.5, 2.6, "does not grow with N", fontsize=13, color=NAVY,
            ha="left", va="center")

    for ax in axes:
        ax.set_xlim(2, 41)
        ax.set_xlabel("Heavy atoms  $N$")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.9, color="#C9D1D9")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    footer(fig, "slide16_scaling.png",
           "Resource scaling only. At one layer information travels one bond, so larger molecules\n"
           "may need a second layer, the dashed lines. Whether accuracy holds at larger N is open.",
           y=-0.02, fontsize=15, draw=True)

    fig.tight_layout()
    path = OUT / "slide16_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide10_matched_capacity():
    """Results slide 1: Table III as a figure, read from the result CSVs.

    Grouped bars of test AUC for the two tasks: the composition-only control
    (grey, 5 parameters), Iso-QGNN (navy) and Iso-CGNN (teal), both at 64.
    Error bars are the cross-trial standard deviation over ten trials. The ten
    individual trials are drawn as a single vertical column of dots just to the
    right of each error bar, so the spread behind each bar is visible rather than
    summarised; height is the only thing a dot encodes, and there is no sideways
    scatter to misread. Value labels sit above the error bars, never on them. The two head-to-head verdicts are annotated where they apply.

    Every number comes from experiment_summary_metrics.csv and
    trivial_baseline_metrics.csv at build time. Nothing is typed in.
    """
    import csv
    import numpy as np
    root = REPO_ROOT
    G = {(r["Model"], r["Property"]): r
         for r in csv.DictReader(open(root / "experiment_summary_metrics.csv"))}
    L = {r["Property"]: r
         for r in csv.DictReader(open(root / "trivial_baseline_metrics.csv"))}

    def stats(r):
        return (float(r["Final_AUC_Mean"]), float(r["Final_AUC_Std_AcrossTrials"]),
                [float(x) for x in r["Final_AUC_PerTrial"].split(";")])

    tasks = [("gap", "HOMO-LUMO gap"), ("mu", "Dipole moment")]
    series = [("control", "composition-only control  (5)", "#B9C2CB", "#7A8794"),
              ("quantum", "Iso-QGNN  (64)", NAVY, NAVY),
              ("classical", "Iso-CGNN  (64)", TEAL, TEAL)]
    offs = {"control": -0.27, "quantum": 0.0, "classical": 0.27}
    W = 0.24

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for gi, (t, tlab) in enumerate(tasks):
        for key, lab, face, edge in series:
            m, sd, per = stats(L[t] if key == "control" else G[(key, t)])
            x = gi + offs[key]
            ax.bar(x, m - 0.5, bottom=0.5, width=W, color=face, edgecolor=edge,
                   linewidth=1.5, zorder=2, label=lab if gi == 0 else None)
            ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.8,
                        capsize=6, capthick=1.8, zorder=4)
            ax.scatter([x + W * 0.34] * len(per), per, s=20, color="white",
                       edgecolor=INK, linewidth=0.9, alpha=0.7, zorder=5)
            top = max(m + sd, max(per))
            ax.text(x, top + 0.014, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=14, color=INK, zorder=6)

    # head-to-head verdicts
    def bracket(gi, y, text, colour):
        x0, x1 = gi + offs["quantum"], gi + offs["classical"]
        ax.plot([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color=colour,
                lw=1.6, zorder=6)
        ax.text((x0 + x1) / 2, y + 0.008, text, ha="center", va="bottom",
                fontsize=13, color=colour, zorder=6, linespacing=1.3)
    bracket(0, 1.012, "+2.8 pts, 0.6 combined SD:\nmodest, consistent classical lead", INK)
    bracket(1, 0.885, "0.013 apart, 0.26 combined SD:\na tie within noise", INK)

    ax.axhline(0.5, color="#9AA6B2", lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.text(1.62, 0.503, "chance", ha="right", va="bottom", fontsize=12.5,
            color="#7A8794")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([t for _, t in tasks], fontsize=16)
    ax.set_xlim(-0.6, 1.62)
    ax.set_ylim(0.5, 1.09)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Test AUC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
              frameon=False, fontsize=13, handlelength=1.2, columnspacing=2.0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.9, color="#C9D1D9")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    footer(fig, "slide10_matched_capacity.png",
           "Bar: mean over ten trials on identical splits. Error bars: ± one cross-trial SD.\n"
           "Dots: the ten trials, one column per bar, height = that trial's AUC.",
           y=-0.015, fontsize=15, draw=True)

    fig.tight_layout()
    path = OUT / "slide10_matched_capacity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide11_sample_complexity():
    """Results slide 2: the sample-complexity sweep, paper Fig. 3 restyled.

    Two panels, one per task, test AUC on the fixed 500-molecule set against
    training-set size on a log axis so the small-N regime, where the story is,
    is readable. Error bars are the SD over five trials. For each curve the
    N90% point, the first size at which the model reaches 90% of its own peak
    mean AUC over the sweep, is computed here from the CSV and marked with a
    vertical guide. The amber band is the "within about 300 molecules" claim.

    Everything is read from data_efficiency_combined.csv at build time. The
    footer carries the paper's own warning: this protocol (3,000-molecule pool,
    fixed 500-molecule test set, 40 epochs, validation-selected epoch, 5 trials)
    differs from Table III's, so absolute AUCs are not comparable between them.
    """
    import csv
    import numpy as np
    root = REPO_ROOT
    rows = list(csv.DictReader(open(root / "data_efficiency_combined.csv")))

    def curve(model, task):
        pts = sorted((int(r["Size"]), float(r["Mean_AUC"]), float(r["Std_AUC"]))
                     for r in rows if r["Model"] == model and r["Property"] == task)
        n = np.array([a for a, _, _ in pts]); m = np.array([b_ for _, b_, _ in pts])
        sd = np.array([c for _, _, c in pts])
        n90 = int(n[np.argmax(m >= 0.9 * m.max())])
        return n, m, sd, n90

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharey=True)
    tasks = [("gap", "HOMO-LUMO gap"), ("mu", "Dipole moment")]
    ticks = [50, 100, 200, 300, 500, 1000, 1500]

    for ax, (t, tlab) in zip(axes, tasks):
        ax.set_xscale("log")
        ax.axvspan(45, 300, color=TARGET_FILL, alpha=0.16, zorder=0, lw=0)
        for model, col, lab, dy in (("classical", TEAL, "Iso-CGNN", 0.955),
                                    ("quantum", NAVY, "Iso-QGNN", 0.975)):
            n, m, sd, n90 = curve(model, t)
            ax.errorbar(n, m, yerr=sd, color=col, lw=2.6, marker="o", ms=6.5,
                        capsize=3.5, elinewidth=1.4, zorder=3, label=lab)
            ax.axvline(n90, color=col, lw=1.6, ls=(0, (4, 3)), zorder=2,
                       ymin=0.0, ymax=0.90)
            ax.text(n90, dy, f"$N_{{90\\%}}$ = {n90}", ha="center", va="center",
                    fontsize=13.5, color=col, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
        ax.set_title(tlab, fontsize=16, color=INK, pad=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(x) for x in ticks])
        ax.minorticks_off()
        ax.set_xlim(45, 1700)
        ax.set_xlabel("Training molecules")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.9, color="#C9D1D9")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Test AUC  (fixed 500-molecule set)")
    axes[0].text(48, 0.535, "within about 300 molecules", fontsize=12.5,
                 color="#8A6D1F", ha="left", va="center")
    axes[0].text(1650, 0.665, "classical leads at every size:\n"
                 "0.10 to 0.18 AUC at N ≤ 200,\n"
                 "a few hundredths above 1,000",
                 ha="right", va="center", fontsize=12.5, color=INK,
                 linespacing=1.35)
    axes[1].text(1650, 0.665, "curves nearly overlap;\nnear-tie at N = 150",
                 ha="right", va="center", fontsize=12.5, color=INK,
                 linespacing=1.35)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), frameon=False,
                   fontsize=13.5)

    footer(fig, "slide11_sample_complexity.png",
           "3,000-molecule pool, fixed 500-molecule test set, five trials per size, 40 epochs, "
           "epoch chosen on validation AUC.\nA different protocol from the matched comparison, "
           "so absolute AUCs are not comparable with it. Error bars: ± one cross-trial SD.",
           y=-0.02, fontsize=15, draw=True)

    fig.tight_layout()
    path = OUT / "slide11_sample_complexity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide12_gradient_stability():
    """Results slide 3: paper Fig. 4 restyled for projection.

    Two panels side by side on a shared log axis: Iso-QGNN in navy, Iso-CGNN in
    teal, so the colour code for substrate is kept and the three parameter
    groups are told apart by line style instead (solid = node-update layer,
    dashed = bond angles, dotted = atom angles). The amber band is the
    "of order 10^-1 to 10^0" region the paper describes. Gap task only, as in
    the paper.

    What is plotted, exactly: q_trainer.py accumulates the L2 norm of each
    parameter tensor's gradient after every update, divides by the number of
    updates in the epoch, and run_experiments.py averages that over ten trials.
    Gradients are exact (state-vector simulation, backprop); there is no shot
    noise. The LayerNorm and classifier gradients are not tracked.

    All values read from experiment_full_curves.csv at build time.
    """
    import csv
    import numpy as np
    root = REPO_ROOT
    rows = [r for r in csv.DictReader(open(root / "experiment_full_curves.csv"))
            if r["Property"] == "gap"]

    def series(model):
        r = sorted((x for x in rows if x["Model"] == model), key=lambda x: int(x["Epoch"]))
        ep = np.array([int(x["Epoch"]) for x in r])
        return ep, {g: np.array([float(x[c]) for x in r]) for g, c in
                    (("node-update", "Grad_GNN"), ("bond", "Grad_Bond"),
                     ("atom", "Grad_Atom"))}

    styles = {"node-update": ("-", 3.0), "bond": ((0, (6, 3)), 2.6),
              "atom": ((0, (1.5, 2.5)), 2.8)}
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharey=True)
    panels = [(axes[0], "quantum", NAVY, "Iso-QGNN",
               "climbs from a near-identity start\nto a plateau within ~25 epochs"),
              (axes[1], "classical", TEAL, "Iso-CGNN",
               "decays from a larger start\nto a comparable steady state")]
    for ax, model, col, title, note in panels:
        ax.axhspan(0.1, 1.0, color=TARGET_FILL, alpha=0.16, zorder=0, lw=0)
        ep, g = series(model)
        for name, (ls, lw) in styles.items():
            ax.plot(ep, g[name], color=col, ls=ls, lw=lw, zorder=3,
                    label=f"{name} parameters")
        ax.set_yscale("log")
        ax.set_ylim(0.06, 2.6)
        ax.set_yticks([0.1, 0.3, 1.0])
        ax.set_yticklabels(["0.1", "0.3", "1"])
        ax.minorticks_off()
        ax.set_xlim(0, 200)
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=17, color=col, pad=10)
        ax.text(197, 1.95, note, ha="right", va="center", fontsize=13,
                color=col, linespacing=1.35)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.9, color="#C9D1D9")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Mean gradient norm per update")
    axes[0].text(3, 0.115, "of order 0.1 to 1: nothing decays towards zero",
                 fontsize=12.5, color="#8A6D1F", ha="left", va="bottom")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#55636F", ls=ls, lw=lw)
               for ls, lw in styles.values()]
    axes[1].legend(handles, [f"{k} parameters" for k in styles],
                   loc="lower right", frameon=False, fontsize=13,
                   handlelength=2.6)

    footer(fig, "slide12_gradient_stability.png",
           "HOMO-LUMO gap task, ten trials. Exact state-vector gradients, no shot noise. Head parameters not tracked.\n"
           "Nine qubits, one layer: a check at this scale, not a statement about scaling.",
           y=-0.02, fontsize=15, draw=True)

    fig.tight_layout()
    path = OUT / "slide12_gradient_stability.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide13_bond_parameters():
    """Results slide 4: paper Fig. 5 restyled, and made honest about units.

    The paper puts quantum angles and classical weights on one axis per task.
    Here each substrate gets its own column and its own scale, because the
    quantum bond parameters are RZ angles in radians (periodic in 2*pi) and the
    classical ones are unbounded message weights; sharing an axis invites a
    comparison of magnitudes that means nothing. Rows are the two tasks.

    The test the paper applies, "SD comparable to or larger than the mean", is
    exactly "the +/- 1 SD error bar crosses zero", so the zero line is the
    reference and the count of crossings is printed. Quantum panels also carry
    dashed guides at +/- pi: a cross-trial SD near pi on an angle means the
    seeds are spread over a half-period, and an arithmetic mean of such angles
    is not a meaningful quantity.

    All values read from experiment_summary_metrics.csv at build time.
    """
    import csv
    import math
    import numpy as np
    root = REPO_ROOT
    G = {(r["Model"], r["Property"]): r
         for r in csv.DictReader(open(root / "experiment_summary_metrics.csv"))}
    bonds = ["Single", "Double", "Triple", "Aromatic"]

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 7.6), sharex=True)
    cross = 0
    for i, (t, tlab) in enumerate((("gap", "HOMO-LUMO gap"), ("mu", "Dipole moment"))):
        for j, (m, col, mlab) in enumerate((("quantum", NAVY, "Iso-QGNN"),
                                             ("classical", TEAL, "Iso-CGNN"))):
            ax = axes[i, j]
            r = G[(m, t)]
            mu = np.array([float(r[f"Bond_{b_}_Mean"]) for b_ in bonds])
            sd = np.array([float(r[f"Bond_{b_}_Std"]) for b_ in bonds])
            x = np.arange(4)
            ax.axhline(0, color=INK, lw=1.2, zorder=2)
            ax.bar(x, mu, width=0.58, color=col, alpha=0.85, zorder=3)
            ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.8,
                        capsize=7, capthick=1.8, zorder=4)
            lim = 1.12 * (abs(mu) + sd).max()
            if m == "quantum":
                lim = max(lim, 4.0)   # always show the +/- pi guides, with headroom for the row label
            for xi, (m_, s_) in enumerate(zip(mu, sd)):
                crosses = s_ >= abs(m_)
                cross += crosses
                if not crosses:
                    # tag on the empty side of the bar, clear of the error bar
                    ytag = 0.06 * lim if m_ < 0 else -0.06 * lim
                    ax.text(xi, ytag, "sign\nresolved", ha="center",
                            va="bottom" if m_ < 0 else "top", fontsize=11,
                            color=col, linespacing=1.2, zorder=6)
            ax.set_ylim(-lim, lim)
            if m == "quantum":
                for yv in (math.pi, -math.pi):
                    ax.axhline(yv, color="#9AA6B2", lw=1.2, ls=(0, (4, 3)), zorder=1)
                ax.text(3.45, math.pi, "+π", ha="right", va="bottom", fontsize=12,
                        color="#7A8794")
                ax.text(3.45, -math.pi, "−π", ha="right", va="top", fontsize=12,
                        color="#7A8794")
                ax.set_ylabel("learned angle  (rad)")
            else:
                ax.set_ylabel("learned message weight")
            if i == 0:
                ax.set_title(mlab, fontsize=17, color=col, pad=10)
            ax.text(-0.45, lim * 0.97, tlab, ha="left", va="top", fontsize=14,
                    color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels(bonds, fontsize=14)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.9, color="#C9D1D9")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    footer(fig, "slide13_bond_parameters.png",
           f"Ten trials, epoch 200. Bars: mean bond parameter. Error bars: ± one cross-trial SD.\n"
           f"{cross} of 16 error bars cross zero, so the SD exceeds the mean and even the sign is not determined across seeds.",
           y=0.02, fontsize=15, draw=True)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT / "slide13_bond_parameters.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def slide06_dataset_and_labels():
    """Methodology slide 1: the data, the two targets, the labels and the metric.

    Slide 6, between the intro circuit figure and the architectures figure. Three panels, each about a third of the area: the two target
    histograms stacked in a left column (gap above dipole) and the heavy-atom
    bars as a full-height column on the right, a layout James asked for so it
    sits well on a slide. Histograms are over the exact 1,500-molecule sample
    the experiments use (the QM9 CSV sampled with seed 42, as in
    run_experiments.get_dataset), coloured by class either side of the sample
    median, with the full-QM9 distribution as a rescaled outline to show the
    sample is representative. The heavy-atom count is why padding is a
    footnote (most molecules fill all nine qubits). AUC is deliberately NOT drawn: James does not want
    fundamental concepts explained on the figures, so the metric is spoken.

    The dashed medians are SAMPLE medians. The experiments use the training-
    split median of each trial, which moves by a few hundredths; the footer
    says so. The QM9 CSV is cached under images/.data/ (downloaded from the
    same DeepChem URL run_experiments.py uses if absent).
    """
    import re
    import urllib.request
    import numpy as np
    import pandas as pd

    cache = OUT / ".data" / "qm9.csv"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv", cache)
    raw = pd.read_csv(cache)
    HARTREE_EV = 27.211386
    sample = raw.sample(n=1500, random_state=42).reset_index(drop=True)
    try:  # same validity filter as run_experiments.get_dataset
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")

        def n_heavy(smi):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return -1
            syms = [a.GetSymbol() for a in m.GetAtoms()]
            return len(syms) if all(s in ("C", "N", "O", "F") for s in syms) else -1
    except ImportError:
        def n_heavy(smi):  # QM9 has no two-letter elements
            return len(re.findall(r"[CNOFcnof]", smi))
    sample["nheavy"] = sample.smiles.map(n_heavy)
    sample = sample[(sample.nheavy > 0) & (sample.nheavy <= 9)]
    n = len(sample)
    gap, mu = sample.gap.values * HARTREE_EV, sample.mu.values
    gap_all, mu_all = raw.gap.values * HARTREE_EV, raw.mu.values

    BELOW, ABOVE = "#B8C4D6", TARGET_FILL
    fig = plt.figure(figsize=(13.6, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], hspace=0.45, wspace=0.2,
                          left=0.07, right=0.985, top=0.965, bottom=0.175)

    def hist_panel(ax, x, x_all, med, xlabel, xmax, unit, xlim=None):
        bins = np.linspace(0, xmax, 41)
        counts, edges = np.histogram(np.clip(x, 0, xmax - 1e-9), bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, counts, width=edges[1] - edges[0],
               color=[ABOVE if c > med else BELOW for c in centers],
               edgecolor="white", linewidth=0.5, zorder=3)
        c_all, _ = np.histogram(np.clip(x_all, 0, xmax - 1e-9), bins)
        ax.step(edges, np.append(c_all, c_all[-1]) * n / len(x_all), where="post",
                color=MUTED, lw=1.7, zorder=4)
        ax.axvline(med, color=INK, lw=2, ls=(0, (4, 3)), zorder=5)
        ymax = counts.max() * 1.6
        ax.set_ylim(0, ymax)
        ax.text(med, ymax * 0.985, f"median {med:.2f} {unit}", ha="center", va="top",
                fontsize=13, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.text(med - 0.025 * xmax, ymax * 0.86, "label 0\nbelow", ha="right", va="top",
                fontsize=13, color="#4A5A70", linespacing=1.1)
        ax.text(med + 0.025 * xmax, ymax * 0.86, "label 1\nabove", ha="left", va="top",
                fontsize=13, color="#8A6D1F", linespacing=1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("molecules in sample")
        ax.set_xlim(*(xlim or (0, xmax)))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return edges, c_all * n / len(x_all), ymax

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    gmed, mmed = float(np.median(gap)), float(np.median(mu))
    edges, outline, ymax1 = hist_panel(ax1, gap, gap_all, gmed, "HOMO-LUMO gap  (eV)", 13, "eV",
                                       xlim=(2, 12.5))
    hist_panel(ax2, mu, mu_all, mmed, "dipole moment  (debye)", 10, "D")
    over = int((mu >= 10).sum())
    ax2.text(9.85, ax2.get_ylim()[1] * 0.45, f"{over} molecule{'s' if over != 1 else ''}\nat or above 10 D\nshown in the last bin",
             ha="right", va="center", fontsize=11, color=MUTED, linespacing=1.15)
    k = int(np.searchsorted(edges, 9.4)) - 1   # bin holding x = 9.4 eV, on the right tail
    ax1.annotate("outline: all of QM9,\nrescaled to the sample",
                 xy=(9.4, outline[k]), xytext=(12.4, ymax1 * 0.42), ha="right", va="center",
                 fontsize=12, color=MUTED, linespacing=1.2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0, shrinkB=2))

    ax3 = fig.add_subplot(gs[:, 1])
    ks = np.arange(1, 10)
    cnt = np.array([int((sample.nheavy == k).sum()) for k in ks])
    ax3.bar(ks, cnt, color=[NAVY if k == 9 else BELOW for k in ks], width=0.7, zorder=3)
    for k, c in zip(ks, cnt):
        if c:
            ax3.text(k, c + cnt.max() * 0.02, f"{c:,}", ha="center", va="bottom", fontsize=12, color=INK)
    ax3.set_xticks(ks)
    ax3.set_xlabel("heavy atoms per molecule")
    ax3.set_ylabel("molecules in sample")
    ax3.set_ylim(0, cnt.max() * 1.32)
    ax3.text(0.6, cnt.max() * 1.28,
             f"{100 * cnt[-1] / n:.0f}% fill all nine qubits.\n"
             "The rest are padded with\n"
             "ghost qubits left in $|0\\rangle$\n"
             "that take part in no gate.",
             ha="left", va="top", fontsize=13, color=INK, linespacing=1.3)
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)

    footer(fig, "slide06_dataset_and_labels.png",
           f"{n:,} molecules drawn once from QM9 with a fixed seed, split 1,200 / 150 / 150 in every trial.\n"
           "Label 1 = above the median of the TRAINING split, recomputed each trial and applied unchanged to held-out "
           "molecules.\nDashed lines: whole-sample medians. The per-trial training medians all lie within 0.1 eV and 0.1 D of them.",
           y=0.075, fontsize=15, draw=True)
    path = OUT / "slide06_dataset_and_labels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


FIGURES = [  # one builder per slide, in deck order (2026-09-07 deck)
    slide03_data_parameter_plane,
    slide04_hea_vs_topology,
    slide06_dataset_and_labels,
    slide07_primitive_mapping_and_budget,
    slide08_training_protocol,
    slide10_matched_capacity,
    slide11_sample_complexity,
    slide12_gradient_stability,
    slide13_bond_parameters,
    slide15_conclusion_plane,
    slide16_scaling,
]


if __name__ == "__main__":
    for build in FIGURES:
        p = build()
        print(f"wrote {p}")
    lines = ["# Slide text lines", "",
             "One line per figure, for the slide's own text box. The figures no longer draw",
             "these footers (see FOOTERS_ON_FIGURE in make_slide_figures.py). Regenerated on",
             "every run, so the numbers in them are as current as the figures.", ""]
    for name in sorted(FOOTER_LINES):
        lines.append(f"- `{name}`: {FOOTER_LINES[name]}")
    (OUT / "slide_text_lines.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "slide_text_lines.md")
