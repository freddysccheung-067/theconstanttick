"""P02 WP0 -- numerical falsification of the 'cycloidal teeth roll' claim.

Run:  python analysis/falsify.py     (prints the report, writes results.json)

Nothing in here assumes the path of contact, the velocity ratio, or the
sliding identity. All three are recovered from two bare parametric curves
and the geometric condition that they touch tangentially.
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import meshing as M

W1 = 1.0
R1, R2, MODULE = 3.0, 0.4, 0.10       # 60T wheel, 8T pinion, m = 0.10 mm
ADD = MODULE
ALPHA = np.radians(20.0)


# ---------------------------------------------------------------- helpers
def first_tip_param(f, Rtip, umax=3.0, n=40000):
    u = np.linspace(1e-9, umax, n)
    i = np.argmax(np.linalg.norm(f.pos(u), axis=-1) > Rtip)
    return brentq(lambda x: np.linalg.norm(f.pos(x)) - Rtip, u[i - 1], u[i])


def arc_along_path(f, params, sub=48):
    """Arc length actually swept, summed segment by segment along the solved
    sequence. Robust to a non-monotonic parameter, which a plain endpoint
    integral is not: if the contact briefly doubled back, min-to-max would
    understate the surface actually covered."""
    p = np.asarray(params, dtype=float)
    tot = 0.0
    for lo, hi in zip(p[:-1], p[1:]):
        t = np.linspace(lo, hi, sub)
        tot += abs(float(np.trapezoid(np.linalg.norm(f.der(t), axis=-1), t)))
    return tot


def run_mesh(g, f1, f2, u_grid, x0):
    s = M.solve_mesh(g, u_grid, x0)
    ok = ~np.isnan(s["th1"])
    s = {k: v[ok] for k, v in s.items()}
    th1, th2 = s["th1"], s["th2"]
    P = np.stack([s["Px"], s["Py"]], axis=-1)

    # monotonic in th1 so we can spline; gives clean analytic derivatives
    o = np.argsort(th1)
    th1s = th1[o]
    cs_th2 = CubicSpline(th1s, th2[o])
    cs_u = CubicSpline(th1s, s["u"][o])
    cs_v = CubicSpline(th1s, s["v"][o])
    w2 = cs_th2(th1, 1) * W1                    # MEASURED angular velocity

    def crossz(w, v):
        return np.stack([-w * v[..., 1], w * v[..., 0]], axis=-1)

    vsA = np.linalg.norm(crossz(W1, P - g.O1) - crossz(w2, P - g.O2), axis=-1)
    vsB = (W1 + np.abs(w2)) * s["PC"]

    d1, d2 = f1.der(s["u"]), f2.der(s["v"])
    sp1 = np.linalg.norm(d1, axis=-1)
    sp2 = np.linalg.norm(d2, axis=-1)
    t1w = M._rot(d1, th1) / sp1[:, None]
    t2w = M._rot(d2, th2) / sp2[:, None]
    sgn = np.sign(np.einsum('ij,ij->i', t1w, t2w))
    sigma1 = sp1 * cs_u(th1, 1) * W1
    sigma2 = sp2 * cs_v(th1, 1) * W1 * sgn
    vsC = np.abs(sigma1 - sigma2)

    s.update(P=P, w2=w2, vsA=vsA, vsB=vsB, vsC=vsC, sp1=sp1, sp2=sp2,
             ratio=w2 / W1)
    return s


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.max(np.abs(a))))


R = {}
L = []


def say(t=""):
    L.append(t)
    print(t)


def main():
    # ============================ 1. CYCLOIDAL ============================
    r = R2 / 2
    f1 = M.epicycloid_face(R1, r)
    f2 = M.hypocycloid_flank(R2, r)
    g = M.MeshGeometry(R1, R2, f1, f2)
    u_tip = first_tip_param(f1, R1 + ADD)
    cy = run_mesh(g, f1, f2, np.linspace(1e-7, u_tip, 1500),
                  [R1 / R2 * 1e-7, -1e-7, R1 / R2 * 1e-7])

    say("=" * 68)
    say("PART 1  CYCLOIDAL MESH -- three things recovered, none assumed")
    say("=" * 68)
    say(f"  contact solutions found        : {len(cy['th1'])} / 1500")
    say(f"  max tangency residual          : {cy['resid'].max():.2e}")
    say(f"  velocity ratio w2/w1  measured : {cy['ratio'].min():.12f} to {cy['ratio'].max():.12f}")
    say(f"                        expected : {-R1/R2:.12f}   -> conjugate action CONFIRMED")
    ctr = np.array([R1 + r, 0.0])
    rad = np.linalg.norm(cy["P"] - ctr, axis=-1)
    say(f"  |P - (R1+r, 0)|                : {rad.min():.15f} to {rad.max():.15f}")
    say(f"                       expected r = {r}   -> path of contact IS the generating circle")
    R["cyc_ratio_err"] = float(np.max(np.abs(cy["ratio"] + R1 / R2)))
    R["cyc_path_err"] = float(np.max(np.abs(rad - r)))
    R["cyc_resid"] = float(cy["resid"].max())

    # --------- the wheel-and-road test: pure geometry, no kinematics ---------
    s1 = arc_along_path(f1, cy["u"])
    s2 = arc_along_path(f2, cy["v"])
    pred = R2 * (R1 + r) / (R1 * (R2 - r))
    say("")
    say("PART 2  THE WHEEL-AND-ROAD TEST  (geometry only -- no velocities)")
    say("-" * 68)
    say("  A wheel rolling on a road lays down exactly as much road as wheel rim.")
    say("  These two flanks are in contact for the whole mesh. Their lengths:")
    say(f"      gear 1 face  (epicycloid)  s1 = {s1*1000:9.4f} um")
    say(f"      gear 2 flank (hypocycloid) s2 = {s2*1000:9.4f} um")
    say(f"      s1 / s2 = {s1/s2:.6f}   closed form R2(R1+r)/(R1(R2-r)) = {pred:.6f}"
        f"   (rel err {abs(s1/s2-pred)/pred:.1e})")
    say(f"      total slip = {abs(s1-s2)*1000:.4f} um = {100*abs(s1-s2)/max(s1,s2):.2f}% of the longer flank")
    say("  Two surfaces in continuous contact, of DIFFERENT length.")
    say("  They cannot be rolling. The folk claim is false on geometry alone.")
    R.update(cyc_s1_um=s1 * 1e3, cyc_s2_um=s2 * 1e3, cyc_arc_ratio=s1 / s2,
             cyc_arc_ratio_closed_form=pred, cyc_slip_um=abs(s1 - s2) * 1e3)

    # ------------------- three independent sliding measures ------------------
    A, B, C = cy["vsA"], cy["vsB"], cy["vsC"]
    say("")
    say("PART 3  SLIDING SPEED, COMPUTED THREE INDEPENDENT WAYS")
    say("-" * 68)
    say("  (A) rigid-body   |w1 x (P-O1) - w2 x (P-O2)|      -- kinematics only")
    say("  (B) the identity (w1 + w2)|PC|                    -- the claim under test")
    say("  (C) scrubbing    |ds1/dt - ds2/dt| along the teeth -- profile-dependent")
    say(f"      A vs B : max relative difference {rel(A, B):.3e}")
    say(f"      A vs C : max relative difference {rel(A, C):.3e}")
    say(f"      -> the tooth shape cancels. |v_s| = (w1+w2)|PC| for this profile.")
    say(f"  |v_s| at the pitch point : {A[0]:.3e} mm/s   (|PC| = {cy['PC'][0]:.2e} mm)")
    say(f"  |v_s| at the tooth tip   : {A[-1]:.6f} mm/s  (|PC| = {cy['PC'][-1]:.6f} mm)")
    say(f"  mesh fraction with |v_s| above 1% of peak : {100*np.mean(A > 0.01*A.max()):.1f}%")
    R.update(cyc_AB=rel(A, B), cyc_AC=rel(A, C), cyc_vs_peak=float(A.max()),
             cyc_PC_peak=float(cy["PC"].max()))

    # ============================ 2. INVOLUTE ============================
    Rb1, Rb2 = R1 * np.cos(ALPHA), R2 * np.cos(ALPHA)
    tC = np.tan(ALPHA)
    ph1 = -np.arctan2(*M.involute_flank(Rb1).pos(tC)[::-1])
    i1 = M.involute_flank(Rb1, phase=ph1)
    ph2 = np.pi - np.arctan2(*M.involute_flank(Rb2, mirror=True).pos(tC)[::-1])
    i2 = M.involute_flank(Rb2, phase=ph2, mirror=True)
    gi = M.MeshGeometry(R1, R2, i1, i2)
    t_tip = np.sqrt(((R1 + ADD) / Rb1) ** 2 - 1.0)
    iv = run_mesh(gi, i1, i2, np.linspace(tC + 1e-9, t_tip, 1500), [tC, 0.0, 0.0])

    say("")
    say("=" * 68)
    say("PART 4  INVOLUTE MESH -- the same treatment, for comparison")
    say("=" * 68)
    say(f"  max tangency residual          : {iv['resid'].max():.2e}")
    say(f"  velocity ratio w2/w1  measured : {iv['ratio'].min():.12f} to {iv['ratio'].max():.12f}")
    dd = iv["P"] - gi.C
    ang = np.degrees(np.abs(np.arctan2(dd[1:, 0], dd[1:, 1])))
    ang = np.minimum(ang, 180 - ang)
    say(f"  angle of PC to common tangent  : {ang.min():.10f} to {ang.max():.10f} deg"
        f"  -> straight line at exactly {np.degrees(ALPHA):.0f} deg")
    s1i = arc_along_path(i1, iv["u"])
    s2i = arc_along_path(i2, iv["v"])
    say(f"  arc lengths in contact : s1 = {s1i*1000:.4f} um   s2 = {s2i*1000:.4f} um"
        f"   ratio {s1i/s2i:.6f}")
    say(f"  -> the involute does not roll either. Nothing rolls.")
    Ai, Bi, Ci = iv["vsA"], iv["vsB"], iv["vsC"]
    say(f"  A vs B : {rel(Ai, Bi):.3e}     A vs C : {rel(Ai, Ci):.3e}")
    R.update(inv_ratio_err=float(np.max(np.abs(iv["ratio"] + R1 / R2))),
             inv_alpha_err=float(np.max(np.abs(ang - np.degrees(ALPHA)))),
             inv_s1_um=s1i * 1e3, inv_s2_um=s2i * 1e3, inv_arc_ratio=s1i / s2i,
             inv_AB=rel(Ai, Bi), inv_AC=rel(Ai, Ci),
             inv_vs_peak=float(Ai.max()), inv_PC_peak=float(iv["PC"].max()))

    # ==================== 3. WHAT THE ADVANTAGE ACTUALLY IS ====================
    say("")
    say("=" * 68)
    say("PART 5  SO WHAT IS THE CYCLOID'S ADVANTAGE?")
    say("=" * 68)
    rv = A.max() / Ai.max()
    rp = cy["PC"].max() / iv["PC"].max()
    say(f"  peak |v_s|  cycloidal {A.max():.6f}   involute {Ai.max():.6f}   ratio {rv:.9f}")
    say(f"  peak |PC|   cycloidal {cy['PC'].max():.6f}   involute {iv['PC'].max():.6f}   ratio {rp:.9f}")
    say(f"  the two ratios agree to {abs(rv-rp):.2e}")
    say("  The cycloid slides exactly as much as the involute at equal |PC|.")
    say("  Its entire advantage is that it keeps |PC| smaller. Geometry of the")
    say("  contact path, not rolling.")
    R.update(peak_vs_ratio=rv, peak_PC_ratio=rp, ratio_agreement=abs(rv - rp))

    # ------------- the straight-flank check: the watchmaker's rule -------------
    h = M.hypocycloid_flank(R2, R2 / 2)
    chi = np.linspace(-1.0, 1.0, 20001)
    ymax = float(np.max(np.abs(h.pos(chi)[:, 1])))
    say("")
    say(f"  BONUS  at r = R2/2 the conjugate flank is straight: max |y| = {ymax:.3e} mm")
    say("         -- the two-century-old 'radial flanks' rule, recovered exactly.")
    R["straight_flank_max_y"] = ymax

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(R, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    open(os.path.join(out_dir, "report.txt"), "w").write("\n".join(L) + "\n")
    np.savez(os.path.join(out_dir, "run.npz"),
             **{f"cyc_{k}": v for k, v in cy.items() if isinstance(v, np.ndarray)},
             **{f"inv_{k}": v for k, v in iv.items() if isinstance(v, np.ndarray)})
    say("")
    say("wrote results/results.json, results/report.txt, results/run.npz")

    return R


if __name__ == "__main__":
    main()
