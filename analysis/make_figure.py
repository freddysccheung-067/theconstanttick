"""Figure WP0: the rolling claim, falsified three ways."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CYC, INV = "#C0492B", "#1B6FB5"      # validated: CVD dE 20.5 protan, 28.2 normal
INK, MUTE, GRID = "#1a1a1a", "#5d5d5d", "#d8d6d0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
d = np.load(os.path.join(RESULTS, "run.npz"))
R = json.load(open(os.path.join(RESULTS, "results.json")))
R1, R2, r = 3.0, 0.4, 0.2
W1, W2 = 1.0, R1 / R2

fig, ax = plt.subplots(2, 2, figsize=(13.6, 10.8))
fig.suptitle("Neither profile rolls: three independent falsifications of the rolling claim",
             fontsize=16.5, fontweight="bold", y=0.978)

# ---------------- (a) where the teeth actually touch ----------------
a = ax[0, 0]
th = np.linspace(-0.13, 0.13, 400)
a.plot(R1*np.cos(th), R1*np.sin(th), color=MUTE, lw=1.0, ls=":", zorder=1)
a.plot((R1+R2) - R2*np.cos(th), R2*np.sin(th), color=MUTE, lw=1.0, ls=":", zorder=1)
a.text(2.855, -0.255, "pitch\ncircles", color=MUTE, fontsize=10, ha="center", linespacing=1.25)
b = np.linspace(np.pi, np.pi*1.42, 300)
a.plot(R1+r+r*np.cos(b), r*np.sin(b), color=CYC, lw=1.0, ls="--", alpha=.40, zorder=2)
a.text(3.245, -0.183, "generating\ncircle", color=CYC, fontsize=9.5, alpha=.85,
       ha="center", linespacing=1.25)
a.plot(d["cyc_Px"], d["cyc_Py"], color=CYC, lw=3.4, solid_capstyle="round",
       label="cycloidal — arc of the generating circle", zorder=4)
a.plot(d["inv_Px"], d["inv_Py"], color=INV, lw=3.4, ls=(0, (5.5, 2.4)),
       solid_capstyle="round", label="involute — straight line of action", zorder=4)
a.plot([R1], [0], "o", ms=11, mfc="white", mec=INK, mew=2.2, zorder=6)
a.annotate("pitch point C\nthe one place where\nsliding is zero", (R1, 0), (2.90, 0.115),
           fontsize=10.5, color=INK, ha="center", linespacing=1.3,
           arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
xe, ye = d["cyc_Px"][-1], d["cyc_Py"][-1]
a.annotate("", (xe, ye), (R1, 0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
a.text((R1+xe)/2 + 0.012, (ye)/2 + 0.028, "|PC|", fontsize=13, style="italic",
       color=INK, fontweight="bold")
a.set_xlim(2.80, 3.42); a.set_ylim(-0.30, 0.20); a.set_aspect("equal")
a.set_title("(a)  Where the teeth actually touch", loc="left", fontsize=13.5, fontweight="bold")
a.set_xlabel("x  [mm]"); a.set_ylabel("y  [mm]")
a.legend(loc="upper right", fontsize=9.6, framealpha=.96)

# ---------------- (b) the collapse: sliding depends on |PC| alone ----------------
a = ax[0, 1]
pcmax = max(d["cyc_PC"].max(), d["inv_PC"].max())
xs = np.linspace(0, pcmax*1.06, 50)
a.plot(xs, (W1+W2)*xs, color=INK, lw=1.3, zorder=2,
       label="$(\\omega_1+\\omega_2)\\,|PC|$")
sl = slice(None, None, 26)
a.plot(d["inv_PC"][sl], d["inv_vsA"][sl], "s", ms=8.5, mfc="none", mec=INV, mew=2.0,
       label="involute — computed", zorder=4)
a.plot(d["cyc_PC"][sl], d["cyc_vsA"][sl], "o", ms=7.5, mfc=CYC, mec="white", mew=1.2,
       label="cycloidal — computed", zorder=5)
for pc, c, nm, dx in ((d["cyc_PC"].max(), CYC, "cycloidal\nstops here", -0.021),
                      (d["inv_PC"].max(), INV, "involute\nruns to here", -0.021)):
    a.axvline(pc, color=c, lw=1.4, ls=":", zorder=1)
    a.text(pc + dx, 2.62, nm, color=c, fontsize=10, ha="right",
           fontweight="bold", linespacing=1.25)
a.plot([0], [0], "o", ms=13, mfc="white", mec=INK, mew=2.4, zorder=6)
a.annotate("“rolling” means the whole mesh\nlives at this single point —\nthe teeth never leave C",
           (0, 0), (0.108, 0.33), fontsize=10.5, color=INK, linespacing=1.35,
           arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
a.set_xlim(-0.010, pcmax*1.10); a.set_ylim(-0.09, 2.95)
a.set_title("(b)  Sliding is set by $|PC|$ alone — the profile cancels",
            loc="left", fontsize=13.5, fontweight="bold")
a.set_xlabel("distance of contact point from the pitch point,  $|PC|$  [mm]")
a.set_ylabel("sliding speed  $|v_s|$  [mm/s]  at $\\omega_1=1$ rad/s")
a.legend(loc="upper left", fontsize=10, framealpha=.96)

# ---------------- (c) the wheel-and-road test ----------------
a = ax[1, 0]
a.grid(axis="y", visible=False)
lab = ["gear 1 face\n(epicycloid)", "gear 2 flank\n(hypocycloid)",
       "gear 1 face\n(involute)", "gear 2 flank\n(involute)"]
val = [R["cyc_s1_um"], R["cyc_s2_um"], R["inv_s1_um"], R["inv_s2_um"]]
col = [CYC, CYC, INV, INV]
y = [4.05, 3.35, 2.15, 1.45]
for yi, v, c, i in zip(y, val, col, range(4)):
    a.barh(yi, v, height=.54, color=c, alpha=1.0 if i % 2 == 0 else .45,
           edgecolor="white", linewidth=2)
    a.text(v + 2.5, yi, f"{v:.1f} µm", va="center", fontsize=11.5, color=INK)
a.set_yticks(y); a.set_yticklabels(lab, fontsize=10.5, linespacing=1.25)
a.set_xlim(0, 150); a.set_ylim(0.30, 4.65)
a.set_xlabel("arc length of tooth surface in contact  [µm]")
a.set_title("(c)  In contact throughout — but different lengths",
            loc="left", fontsize=13.5, fontweight="bold")
for yv, v1, v2 in ((3.70, R["cyc_s1_um"], R["cyc_s2_um"]),
                   (1.80, R["inv_s1_um"], R["inv_s2_um"])):
    a.annotate("", (v1, yv), (v2, yv), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
    a.text((v1+v2)/2, yv+.10, f"slip {v1-v2:.1f} µm", ha="center", fontsize=11,
           fontweight="bold", color=INK)
a.text(75, 0.60, "A rolling wheel lays down exactly as much road as it has rim.\n"
                 "These do not. Pure geometry — no velocities used anywhere.",
       ha="center", fontsize=10.5, color=MUTE, style="italic", linespacing=1.35)

# ---------------- (d) verification ----------------
a = ax[1, 1]
s2 = slice(None, None, 4)
for pre, c, ls, nm in (("cyc", CYC, "-", "cycloidal"), ("inv", INV, (0, (5.5, 2.4)), "involute")):
    A, B, C = d[f"{pre}_vsA"], d[f"{pre}_vsB"], d[f"{pre}_vsC"]
    x = d[f"{pre}_PC"] / d[f"{pre}_PC"].max()
    a.semilogy(x[s2], np.maximum(np.abs(A-B)/A.max(), 1e-17)[s2], color=c, lw=1.9, ls=ls,
               label=f"{nm}:  (A) vs (B)")
    a.semilogy(x[s2], np.maximum(np.abs(A-C)/A.max(), 1e-17)[s2], color=c, lw=1.2, ls=":",
               alpha=.8, label=f"{nm}:  (A) vs (C)")
a.axhline(2.2e-16, color=INK, lw=1.4, ls="-.")
a.text(0.30, 3.2e-16, "double-precision epsilon", fontsize=10, color=INK)
a.axvspan(0, 0.012, color="#c9c6bd", alpha=.9, zorder=0)
a.text(0.030, 2.5e-8, "cusp: the epicycloid\nparametrisation is\nsingular at C",
       fontsize=9.5, color=MUTE, linespacing=1.3)
a.set_xlim(0, 1); a.set_ylim(1e-17, 2e-4)
a.set_xlabel("position through the mesh,   $|PC| \\, / \\, |PC|_{\\mathrm{max}}$")
a.set_ylabel("relative difference between methods")
a.set_title("(d)  Three ways of computing $|v_s|$, all agreeing",
            loc="left", fontsize=13.5, fontweight="bold")
a.legend(loc="upper right", fontsize=9.2, framealpha=.96, ncol=2)
a.text(0.33, 1.6e-6, "(A) rigid body    (B) $(\\omega_1{{+}}\\omega_2)|PC|$    (C) surface scrubbing",
       fontsize=9.4, color=MUTE, ha="left")

fig.tight_layout(rect=[0, 0.016, 1, 0.952])
fig.text(0.5, 0.005, "60→8 mesh, module 0.10 mm, generating circle r = R₂/2, addendum = 1 module.   "
         "Contact located by solving tangency; the velocity ratio and the path of contact are outputs, not inputs.",
         ha="center", fontsize=9.5, color=MUTE)
fig.savefig(os.path.join(RESULTS, "fig0_no_rolling.png"), dpi=200)
fig.savefig(os.path.join(RESULTS, "fig0_no_rolling.pdf"))
print("wrote results/fig0_no_rolling.png / .pdf")
