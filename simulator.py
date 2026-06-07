"""

Simulateur de Modulation Numérique — BPSK, QPSK, 16-QAM (CORRIGÉ)

Canal AWGN | Constellations | Courbes BER vs Eb/N0

"""

 

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy.special import erfc

 

# ══════════════════════════════════════════════════════════

#  PARAMÈTRES

# ══════════════════════════════════════════════════════════

NUM_BITS    = 500_000

MIN_ERRORS  = 100

MAX_BITS    = 20_000_000

EB_NO_RANGE = np.arange(0, 14, 1)

EB_NO_FINE  = np.arange(0, 14, 0.1)

EB_NO_VISU  = 8

N_VISU      = 4000

SEED        = 42

 

rng = np.random.default_rng(SEED)

 

def gray_enc(n): return n ^ (n >> 1)

def gray_dec(g):

    mask = g

    while mask:

        mask >>= 1

        g ^= mask

    return g

 

def awgn(symbols, ebn0_db, bits_per_sym):

    ebn0_lin = 10 ** (ebn0_db / 10.0)

    sigma    = np.sqrt(1.0 / (2.0 * bits_per_sym * ebn0_lin))

    noise    = sigma * (rng.standard_normal(len(symbols))

                      + 1j * rng.standard_normal(len(symbols)))

    return symbols + noise

 

# ── BPSK ──────────────────────────────────────────────────

def bpsk_mod(bits):   return (1 - 2 * bits).astype(complex)

def bpsk_demod(rx):   return (rx.real < 0).astype(int)

def bpsk_theory(ebn0_db):

    return 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10.0)))

 

# ── QPSK ──────────────────────────────────────────────────

_QPSK_SYM = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)

 

def qpsk_mod(bits):

    bits  = bits[:len(bits) - len(bits) % 2]

    pairs = bits.reshape(-1, 2)

    idx   = pairs[:, 0] * 2 + pairs[:, 1]

    g_idx = np.array([gray_enc(i) for i in idx])

    return _QPSK_SYM[g_idx]

 

def qpsk_demod(rx):

    dists = np.abs(rx[:, None] - _QPSK_SYM[None, :])

    g_idx = np.argmin(dists, axis=1)

    idx   = np.array([gray_dec(int(g)) for g in g_idx])

    bits  = np.empty(len(rx) * 2, dtype=int)

    bits[0::2] = (idx >> 1) & 1

    bits[1::2] =  idx & 1

    return bits

 

def qpsk_theory(ebn0_db):

    return 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10.0)))

 

# ── 16-QAM (CORRIGÉ) ──────────────────────────────────────

_PAM4          = np.array([-3, -1, 1, 3])

_QAM16_NORM    = np.sqrt(10)

 

def _pam4_map(bits2):

    idx  = bits2[:, 0] * 2 + bits2[:, 1]

    gidx = np.array([gray_enc(i) for i in idx])

    return _PAM4[gidx].astype(float)

 

def _pam4_decide(vals_scaled):

    dists = np.abs(vals_scaled[:, None] - _PAM4[None, :])

    gidx  = np.argmin(dists, axis=1)

    idx   = np.array([gray_dec(int(g)) for g in gidx])

    return (idx >> 1) & 1, idx & 1

 

def qam16_mod(bits):

    bits = bits[:len(bits) - len(bits) % 4]

    g    = bits.reshape(-1, 4)

    I    = _pam4_map(g[:, 0:2])

    Q    = _pam4_map(g[:, 2:4])

    return (I + 1j * Q) / _QAM16_NORM

 

def qam16_demod(rx):

    scaled = rx * _QAM16_NORM

    b0I, b1I = _pam4_decide(scaled.real)

    b0Q, b1Q = _pam4_decide(scaled.imag)

    bits = np.empty(len(rx) * 4, dtype=int)

    bits[0::4] = b0I; bits[1::4] = b1I

    bits[2::4] = b0Q; bits[3::4] = b1Q

    return bits

 

def qam16_theory(ebn0_db):

    ebn0 = 10 ** (ebn0_db / 10.0)

    return (3/8) * erfc(np.sqrt(ebn0 * 4 / 10))

 

# ── Simulation BER adaptative ─────────────────────────────

def simulate_ber(mod_fn, demod_fn, bps, ebn0_db):

    errors = 0

    total  = 0

    while errors < MIN_ERRORS and total < MAX_BITS:

        n    = NUM_BITS - NUM_BITS % bps

        bits = rng.integers(0, 2, n).astype(int)

        syms = mod_fn(bits)

        rx   = awgn(syms, ebn0_db, bps)

        dec  = demod_fn(rx)

        nb   = min(len(bits), len(dec))

        errors += int(np.sum(bits[:nb] != dec[:nb]))

        total  += nb

    return errors / total if total > 0 else np.nan

 

# ══════════════════════════════════════════════════════════

#  MAIN

# ══════════════════════════════════════════════════════════

def main():

    print("=" * 58)

    print("  Simulateur BPSK | QPSK | 16-QAM  —  Canal AWGN")

    print("=" * 58)

 

    configs = [

        ("BPSK",   bpsk_mod,  bpsk_demod,  1, bpsk_theory,  "#2196F3"),

        ("QPSK",   qpsk_mod,  qpsk_demod,  2, qpsk_theory,  "#4CAF50"),

        ("16-QAM", qam16_mod, qam16_demod, 4, qam16_theory, "#FF5722"),

    ]

 

    # ── BER ───────────────────────────────────────────────

    print("\n[1/2]  Simulation BER…")

    ber_sim = {}

    for name, mf, df, bps, tf, color in configs:

        print(f"       → {name}", end="", flush=True)

        bers = []

        for ebn0 in EB_NO_RANGE:

            bers.append(simulate_ber(mf, df, bps, ebn0))

            print(".", end="", flush=True)

        ber_sim[name] = (np.array(bers), tf(EB_NO_FINE), color)

        print(" ✓")

 

    # ── Constellations ────────────────────────────────────

    print(f"\n       Constellations à Eb/N0 = {EB_NO_VISU} dB…")

    consts = {}

    for name, mf, df, bps, tf, color in configs:

        n    = N_VISU * bps

        bits = rng.integers(0, 2, n).astype(int)

        rx   = awgn(mf(bits), EB_NO_VISU, bps)

        consts[name] = (rx, color)

 

    ideal_pts = {

        "BPSK":   np.array([1+0j, -1+0j]),

        "QPSK":   _QPSK_SYM,

        "16-QAM": np.array([(i + 1j*q) / _QAM16_NORM

                             for i in _PAM4 for q in _PAM4]),

    }

    ideal_labels = {

        "BPSK":   ["0", "1"],

        "QPSK":   ["00", "01", "11", "10"],

        "16-QAM": [],

    }

 

    # ══════════════════════════════════════════════════════

    #  FIGURE  —  layout 2×3

    # ══════════════════════════════════════════════════════

    print("\n[2/2]  Génération figure…")

 

    BG    = "#0d1117"

    PANEL = "#161b22"

    GRID  = "#21262d"

    TEXT  = "#e6edf3"

    MUTED = "#8b949e"

 

    fig = plt.figure(figsize=(18, 10), facecolor=BG)

 

    gs = gridspec.GridSpec(2, 3, figure=fig,

                           hspace=0.45, wspace=0.30,

                           left=0.05, right=0.97,

                           top=0.93, bottom=0.12)

 

    def style(ax):

        ax.set_facecolor(PANEL)

        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

        ax.tick_params(colors=MUTED, labelsize=8)

        ax.xaxis.label.set_color(MUTED)

        ax.yaxis.label.set_color(MUTED)

        ax.title.set_color(TEXT)

        ax.grid(True, color=GRID, linewidth=0.6)

 

    fig.text(0.5, 0.965,

             "Simulateur de Modulation Numérique  —  BPSK  |  QPSK  |  16-QAM  |  Canal AWGN",

             ha="center", fontsize=13, color=TEXT, fontweight="bold")

 

    # ── Ligne 0 : 3 constellations ────────────────────────

    for col_idx, name in enumerate(["BPSK", "QPSK", "16-QAM"]):

        ax = fig.add_subplot(gs[0, col_idx])

        style(ax)

        rx, color = consts[name]

        ax.scatter(rx.real, rx.imag, s=3, alpha=0.25, color=color)

        pts = ideal_pts[name]

        ax.scatter(pts.real, pts.imag, s=90, color="white",

                   edgecolors=color, linewidths=1.2, zorder=5, marker="o")

        for pt, lbl in zip(pts, ideal_labels.get(name, [])):

            ax.annotate(lbl, (pt.real, pt.imag),

                        xytext=(5, 3), textcoords="offset points",

                        fontsize=6.5, color="white", fontweight="bold")

        ax.axhline(0, color=GRID, lw=0.8, ls="--")

        ax.axvline(0, color=GRID, lw=0.8, ls="--")

        ax.set_title(f"Constellation {name}  (Eb/N0 = {EB_NO_VISU} dB)", fontsize=10)

        ax.set_xlabel("I (en-phase)", fontsize=9)

        ax.set_ylabel("Q (quadrature)", fontsize=9)

        ax.set_aspect("equal", adjustable="box")

 

    # ── Ligne 1 : courbe BER (2 colonnes) ─────────────────

    ax_ber = fig.add_subplot(gs[1, :2])

    style(ax_ber)

    ax_ber.grid(True, which="both", color=GRID, linewidth=0.5)

 

    line_styles = {"BPSK": (2.0, "-"),  "QPSK": (2.0, "--"), "16-QAM": (2.0, "-")}

    markers     = {"BPSK": "o",         "QPSK": "s",          "16-QAM": "^"}

 

    for name, (ber_s, ber_t, color) in ber_sim.items():

        lw, ls = line_styles[name]

        mk     = markers[name]

 

        # Courbe théorique

        ax_ber.semilogy(EB_NO_FINE, ber_t,

                        linestyle=ls, color=color, linewidth=lw,

                        label=f"{name} — théorie")

 

        # Points simulés

        mask_valid = (ber_s > 0) & (ber_s < 1.0)

        if np.any(mask_valid):

            ax_ber.semilogy(EB_NO_RANGE[mask_valid], ber_s[mask_valid],

                            linestyle="none", marker=mk, color=color,

                            ms=7, alpha=0.9,

                            markeredgecolor="white", markeredgewidth=0.6,

                            label=f"{name} — simulation")

 

    ax_ber.set_xlabel("Eb/N₀ (dB)", fontsize=10)

    ax_ber.set_ylabel("BER", fontsize=10)

    ax_ber.set_title(

        "Taux d'Erreur Binaire vs Eb/N₀\nLigne = théorie   |   Symboles = simulation Monte Carlo",

        fontsize=10)

    ax_ber.set_xlim(0, 13)

    ax_ber.set_ylim(1e-6, 1)

 

    ax_ber.legend(

        fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, ncol=2,

        loc="lower left", bbox_to_anchor=(0.01, 0.01)

    )

 

    # Annotations adaptées aux nouvelles courbes valides

    annot_cfg = [

        ("BPSK",   "#2196F3",  6.8, 1e-3,  5.5, 3e-6, "arc3,rad=0.3"),

        ("QPSK",   "#4CAF50",  6.8, 1e-3,  7.5, 3e-6, "arc3,rad=0.2"),

        ("16-QAM", "#FF5722", 10.5, 1e-3, 10.0, 3e-6, "arc3,rad=-0.2"),

    ]

    for name, color, xarr, yarr, xtxt, ytxt, conn in annot_cfg:

        ax_ber.annotate(

            f"{name} : Eb/N₀ = {xarr} dB\npour BER = 10⁻³",

            xy=(xarr, yarr), xytext=(xtxt, ytxt),

            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, connectionstyle=conn),

            fontsize=8, color=color,

            bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=color, lw=0.9), ha="center"

        )

 

    # ── Ligne 1 : zoom BPSK vs QPSK (col 2) ──────────────

    ax_zoom = fig.add_subplot(gs[1, 2])

    style(ax_zoom)

    ax_zoom.grid(True, which="both", color=GRID, linewidth=0.5)

 

    for name, ls, mk in [("BPSK", "-", "o"), ("QPSK", "--", "s")]:

        ber_s, ber_t, color = ber_sim[name]

        mask = (ber_s > 0) & (ber_s < 1.0)

        ax_zoom.semilogy(EB_NO_FINE, ber_t, linestyle=ls, color=color, lw=2, label=name)

        ax_zoom.semilogy(EB_NO_RANGE[mask], ber_s[mask],

                         linestyle="none", marker=mk, color=color,

                         ms=7, alpha=0.9, markeredgecolor="white", markeredgewidth=0.6)

 

    ax_zoom.set_xlabel("Eb/N₀ (dB)", fontsize=9)

    ax_zoom.set_ylabel("BER", fontsize=9)

    ax_zoom.set_title("Zoom : BPSK vs QPSK\n(BER identique — ligne vs tirets)", fontsize=9)

    ax_zoom.set_xlim(0, 12)

    ax_zoom.set_ylim(1e-5, 1)

    ax_zoom.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, loc="upper right")

 

    # ── Sauvegarde ────────────────────────────────────────

    plt.savefig("simulateur_modulation.png", dpi=150, bbox_inches="tight", facecolor=BG)

    plt.show()

    print("       ✓  simulateur_modulation.png  sauvegardé")

 

    print("\n" + "=" * 58)

    print(f"  {'Modulation':<10} {'BER @ 4dB':>11} {'BER @ 8dB':>11} {'Eb/N0@10⁻³':>11}")

    print("  " + "-" * 46)

    for name, (ber_s, ber_t, _) in ber_sim.items():

        b4  = ber_t[np.argmin(np.abs(EB_NO_FINE - 4))]

        b8  = ber_t[np.argmin(np.abs(EB_NO_FINE - 8))]

        req = EB_NO_FINE[np.argmin(np.abs(ber_t - 1e-3))]

        print(f"  {name:<10} {b4:>11.2e} {b8:>11.2e} {req:>10.1f} dB")

    print("=" * 58)

 

if __name__ == "__main__":

    main()