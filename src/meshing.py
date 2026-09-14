"""Numerical falsification of the 'cycloidal teeth roll' claim.

The point of this module is INDEPENDENCE. Nothing here assumes:
  * that the cycloidal path of contact is an arc of the generating circle,
  * that the velocity ratio is constant,
  * that |v_s| = (w1 + w2)|PC|,
  * or any other result from gear theory.

All three are *outputs*, recovered numerically from nothing but two
parametric curves and the geometric condition that they touch.

Method
------
Each tooth flank is built once, in its own body frame, from the classical
generating construction -- and then that construction is FORGOTTEN. What is
passed to the contact solver is a bare parametric curve c(t) with a
derivative, nothing more.

For a given parameter u on flank 1, we solve for the three unknowns
(v, th1, th2) that put the two flanks in genuine tangential contact:

    P1(u; th1) - P2(v; th2) = 0        (2 eqs: the points coincide)
    t1(u; th1) x t2(v; th2) = 0        (1 eq: the tangents are parallel)

Three equations, three unknowns, non-degenerate. Two things follow that are
NOT inputs:

  1. th2 as a function of th1 -- so constant velocity ratio is MEASURED.
  2. the locus of the contact point -- so the path of contact is MEASURED.

Geometry convention (world frame)
---------------------------------
    O1 = (0, 0)          gear 1, pitch radius R1, rotates +th1 (CCW)
    O2 = (a, 0)          gear 2, pitch radius R2, rotates +th2 (th2 < 0, CW)
    C  = (R1, 0)         pitch point,  a = R1 + R2
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve

# --------------------------------------------------------------------------
# parametric flanks: position and derivative, in the gear's own body frame
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Flank:
    """A bare parametric curve. The solver is given nothing else."""
    name: str
    pos: callable          # t -> (x, y)   in body frame
    der: callable          # t -> (dx/dt, dy/dt)


def epicycloid_face(R: float, r: float) -> Flank:
    """Addendum face: circle radius r rolled on the OUTSIDE of pitch circle R.

    Cusp (the pitch-circle crossing) sits at (R, 0), reached at psi = 0.
    """
    k = (R + r) / r

    def pos(psi):
        psi = np.asarray(psi, dtype=float)
        return np.stack([(R + r) * np.cos(psi) - r * np.cos(k * psi),
                         (R + r) * np.sin(psi) - r * np.sin(k * psi)], axis=-1)

    def der(psi):
        psi = np.asarray(psi, dtype=float)
        # r*k = R + r exactly, so both components carry the factor (R+r)
        return np.stack([(R + r) * (np.sin(k * psi) - np.sin(psi)),
                         (R + r) * (np.cos(psi) - np.cos(k * psi))], axis=-1)

    return Flank(f"epicycloid(R={R},r={r})", pos, der)


def hypocycloid_flank(R: float, r: float, rotate_pi: bool = True) -> Flank:
    """Dedendum flank: circle radius r rolled on the INSIDE of pitch circle R.

    With rotate_pi the cusp is moved to (-R, 0), which is where gear 2's own
    pitch point sits in gear 2's body frame.
    """
    j = (R - r) / r
    s = -1.0 if rotate_pi else 1.0

    def pos(chi):
        chi = np.asarray(chi, dtype=float)
        return s * np.stack([(R - r) * np.cos(chi) + r * np.cos(j * chi),
                             (R - r) * np.sin(chi) - r * np.sin(j * chi)], axis=-1)

    def der(chi):
        chi = np.asarray(chi, dtype=float)
        # r*j = R - r exactly
        return s * np.stack([(R - r) * (-np.sin(chi) - np.sin(j * chi)),
                             (R - r) * (np.cos(chi) - np.cos(j * chi))], axis=-1)

    return Flank(f"hypocycloid(R={R},r={r})", pos, der)


def involute_flank(Rb: float, phase: float = 0.0, mirror: bool = False) -> Flank:
    """Involute of a base circle of radius Rb, starting on the base circle."""
    s = -1.0 if mirror else 1.0

    def pos(t):
        t = np.asarray(t, dtype=float)
        x = Rb * (np.cos(t) + t * np.sin(t))
        y = s * Rb * (np.sin(t) - t * np.cos(t))
        c, sn = np.cos(phase), np.sin(phase)
        return np.stack([c * x - sn * y, sn * x + c * y], axis=-1)

    def der(t):
        t = np.asarray(t, dtype=float)
        dx = Rb * t * np.cos(t)
        dy = s * Rb * t * np.sin(t)
        c, sn = np.cos(phase), np.sin(phase)
        return np.stack([c * dx - sn * dy, sn * dx + c * dy], axis=-1)

    return Flank(f"involute(Rb={Rb})", pos, der)


# --------------------------------------------------------------------------
# rigid-body placement
# --------------------------------------------------------------------------

def _rot(v, ang):
    v = np.asarray(v, dtype=float)
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] - s * v[..., 1],
                     s * v[..., 0] + c * v[..., 1]], axis=-1)


def place(flank: Flank, t, theta, origin):
    """Position and tangent of the flank in the world frame."""
    p = _rot(flank.pos(t), theta) + np.asarray(origin, dtype=float)
    d = _rot(flank.der(t), theta)
    return p, d


# --------------------------------------------------------------------------
# the contact solver
# --------------------------------------------------------------------------

@dataclass
class MeshGeometry:
    R1: float
    R2: float
    flank1: Flank
    flank2: Flank

    @property
    def a(self):
        return self.R1 + self.R2

    @property
    def O1(self):
        return np.array([0.0, 0.0])

    @property
    def O2(self):
        return np.array([self.a, 0.0])

    @property
    def C(self):
        return np.array([self.R1, 0.0])


def contact_residual(x, u, g: MeshGeometry):
    """[dx, dy, tangent cross product] for unknowns x = (v, th1, th2)."""
    v, th1, th2 = x
    p1, d1 = place(g.flank1, u, th1, g.O1)
    p2, d2 = place(g.flank2, v, th2, g.O2)
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    cross = (d1[0] * d2[1] - d1[1] * d2[0]) / max(n1 * n2, 1e-300)
    return np.array([p1[0] - p2[0], p1[1] - p2[1], cross])


def solve_mesh(g: MeshGeometry, u_values, x0=None, jump_factor=6.0,
               max_depth=12, internal=2000, bootstrap=40, anchor_frac=0.30):
    """Continuation sweep over u. Returns a dict of arrays, one entry per u.

    Why this is more careful than a plain continuation
    --------------------------------------------------
    The tangency system has MORE THAN ONE root. The hypocycloid meets the same
    world point with the same tangent on several laps, and there are spurious
    contacts at other rotations. Every one satisfies the system to machine
    zero, so a wrong branch looks perfectly healthy -- residual 1e-16, a smooth
    sweep -- while the arc length is wrong by a factor of two or fifty. It is a
    silent-corruption failure, and in a naive continuation it is also
    grid-dependent: right at one sampling density, wrong at another.

    Three defences, none of which assumes a result from gear theory:

    1. **Anchor in the middle of the mesh, not at the cusp.** At u = 0 both
       flanks sit on their own cusp, where the derivatives vanish, the tangency
       condition is singular and fsolve simply stops wherever it happens to be:
       every "root" there agrees to eight decimals and is numerical noise.
       Anchoring on that noise was the original bug. Instead the anchor is at
       `anchor_frac` of the way through, where the roots are cleanly separated,
       and the sweep runs outward in BOTH directions from it. The root kept is
       the one whose contact point is nearest the pitch point, breaking ties on
       the smallest |v| so a lap-shifted duplicate of the same physical contact
       cannot be chosen.
    2. **A dense internal grid**, so branch selection does not depend on the
       output sampling.
    3. **Continuity of the contact POINT** -- a step is rejected, and the
       interval bisected and retried, if the contact point jumps more than
       `jump_factor` times the running median step. That is continuity of
       contact: a physical assumption about the mesh, not a theorem about it.

    A step still outside the bound at max_depth is REJECTED, not accepted: a
    gap in the sweep is recoverable, a silently wrong branch is not.
    """
    u_values = np.asarray(u_values, dtype=float)
    n = len(u_values)
    out = {k: np.full(n, np.nan) for k in
           ("u", "v", "th1", "th2", "Px", "Py", "PC", "resid")}
    rate = g.R1 / g.R2

    def solve_at(u, seed):
        sol, info, ier, msg = fsolve(contact_residual, seed, args=(u, g),
                                     full_output=True, xtol=1e-14)
        return sol, float(np.abs(contact_residual(sol, u, g)).max())

    def point(u, sol):
        p, _ = place(g.flank1, u, sol[1], g.O1)
        return p

    lo, hi = float(u_values[0]), float(u_values[-1])
    ua = lo + anchor_frac * (hi - lo)

    # ---- 1. anchor, in the well-conditioned interior ---------------------
    cands = []
    seeds = [np.array([sv * rate * ua * mag, sth * ua, rate * ua])
             for mag in np.linspace(0.1, 6.0, 24)
             for sv in (1.0, -1.0) for sth in (-1.0, 1.0)]
    if x0 is not None:
        seeds.append(np.array(x0, dtype=float))
    for seed in seeds:
        sol, res = solve_at(ua, seed)
        if res < 1e-11 and np.isfinite(sol).all():
            pc = float(np.linalg.norm(point(ua, sol) - g.C))
            cands.append((round(pc, 12), abs(sol[0]), sol, res))
    if not cands:
        return out
    _, _, sol_a, res_a = min(cands, key=lambda c: (c[0], c[1]))

    # ---- 2 & 3. sweep outward from the anchor, both directions -----------
    def run(grid, sol_start, res_start):
        us, xs, rs, steps = [ua], [sol_start], [res_start], []

        def advance(u_from, u_to, seed, p_ref, depth):
            sol, res = solve_at(u_to, seed)
            if res >= 1e-10 or not np.isfinite(sol).all():
                good = False
            else:
                d = float(np.linalg.norm(point(u_to, sol) - p_ref))
                lim = (jump_factor * np.median(steps)
                       if len(steps) >= bootstrap else np.inf)
                good = d <= lim
            if good:
                return sol, res
            if depth >= max_depth:
                return None, res
            mid = 0.5 * (u_from + u_to)
            s_mid, _ = advance(u_from, mid, seed, p_ref, depth + 1)
            if s_mid is None:
                return None, res
            return advance(mid, u_to, s_mid, point(mid, s_mid), depth + 1)

        for u in grid:
            seed = 2.0 * xs[-1] - xs[-2] if len(xs) >= 2 else xs[-1]
            p_ref = point(us[-1], xs[-1])
            sol, res = advance(us[-1], u, seed, p_ref, 0)
            if sol is None:
                continue
            steps.append(float(np.linalg.norm(point(u, sol) - p_ref)))
            steps = steps[-400:]
            us.append(u); xs.append(sol); rs.append(res)
        return us[1:], xs[1:], rs[1:]

    nf = max(int(internal * (hi - ua) / (hi - lo)), 2)
    nb = max(int(internal * (ua - lo) / (hi - lo)), 2)
    uf, xf, rf = run(np.linspace(ua, hi, nf)[1:], sol_a, res_a)
    ub, xb, rb = run(np.linspace(ua, lo, nb)[1:], sol_a, res_a)

    us = np.array(ub[::-1] + [ua] + uf)
    xs = np.array(list(xb[::-1]) + [sol_a] + list(xf))
    rs = np.array(rb[::-1] + [res_a] + rf)
    tol = 2.0 * (hi - lo) / internal + 1e-12

    for i, u in enumerate(u_values):
        j = int(np.argmin(np.abs(us - u)))
        if abs(us[j] - u) > tol:
            continue
        v, th1, th2 = xs[j]
        p1 = point(us[j], xs[j])
        out["u"][i], out["v"][i] = us[j], v
        out["th1"][i], out["th2"][i] = th1, th2
        out["Px"][i], out["Py"][i] = p1
        out["PC"][i] = np.linalg.norm(p1 - g.C)
        out["resid"][i] = rs[j]
    return out
