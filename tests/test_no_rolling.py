"""WP0: the rolling claim, as assertions.

Every test here fails if the folk claim is right. Run: pytest tests/ -q
"""
import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import meshing as M
import falsify as F

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

if not os.path.exists(os.path.join(RESULTS, "results.json")):
    F.main()
R = json.load(open(os.path.join(RESULTS, "results.json")))
R1, R2, r = F.R1, F.R2, F.R2 / 2


# ---------------------------------------------------- the geometry is real
def test_conjugate_action_is_recovered_not_assumed():
    """th2 was a free unknown. It must come out at exactly -R1/R2."""
    assert R["cyc_ratio_err"] < 1e-9
    assert R["inv_ratio_err"] < 1e-9


def test_contact_solver_actually_found_tangency():
    assert R["cyc_resid"] < 1e-10


def test_cycloidal_path_of_contact_is_the_generating_circle():
    """Never assumed anywhere: the solver was given two bare curves."""
    assert R["cyc_path_err"] < 1e-12


def test_involute_path_of_contact_is_a_straight_line_at_the_pressure_angle():
    assert R["inv_alpha_err"] < 1e-8


def test_traditional_generating_circle_gives_straight_radial_flanks():
    """r = R2/2 is the watchmakers' rule. The conjugate flank goes straight."""
    assert R["straight_flank_max_y"] < 1e-15


# ------------------------------------------- the claim itself, falsified
def test_cycloidal_teeth_do_not_roll_on_geometry_alone():
    """Two surfaces in contact throughout the mesh, of different length.

    Rolling contact requires equal arc lengths -- a wheel lays down exactly
    as much road as it has rim. THIS TEST IS THE PAPER.
    """
    assert R["cyc_arc_ratio"] == pytest.approx(2.133333, rel=1e-5)
    assert abs(R["cyc_arc_ratio"] - 1.0) > 1.0
    assert R["cyc_slip_um"] > 50.0


def test_involute_teeth_do_not_roll_either():
    assert abs(R["inv_arc_ratio"] - 1.0) > 1.0


def test_arc_length_ratio_matches_the_closed_form():
    """s1/s2 = R2(R1+r) / (R1(R2-r)), constant through the mesh."""
    assert R["cyc_arc_ratio"] == pytest.approx(R["cyc_arc_ratio_closed_form"], rel=1e-9)


# Physically valid band only: below r/R2 ~ 0.29 the contact ratio drops under
# 1 and the drive is discontinuous, and above r = R2/2 the flank undercuts.
# The solver does lose the branch at r = 0.06 and 0.07 -- logged as a known
# limitation rather than hidden, but both are infeasible geometries.
@pytest.mark.parametrize("rr", [0.12, 0.14, 0.16, 0.18, 0.20])
def test_arc_length_ratio_closed_form_across_generating_radii(rr):
    f1, f2 = M.epicycloid_face(R1, rr), M.hypocycloid_flank(R2, rr)
    g = M.MeshGeometry(R1, R2, f1, f2)
    ut = F.first_tip_param(f1, R1 + F.ADD)
    s = M.solve_mesh(g, np.linspace(1e-7, ut, 400))
    ok = ~np.isnan(s["th1"])
    s1 = F.arc_along_path(f1, s["u"][ok])
    s2 = F.arc_along_path(f2, s["v"][ok])
    assert s1 / s2 == pytest.approx(R2 * (R1 + rr) / (R1 * (R2 - rr)), rel=1e-8)


# --------------------------------------- the identity, three ways
def test_rigid_body_equals_the_identity():
    assert R["cyc_AB"] < 1e-10
    assert R["inv_AB"] < 1e-10


def test_rigid_body_equals_surface_scrubbing_away_from_the_cusp():
    """(C) goes through the tooth shape; (A) does not. Agreement is the point."""
    d = np.load(os.path.join(RESULTS, "run.npz"))
    for pre in ("cyc", "inv"):
        A, C, PC = d[f"{pre}_vsA"], d[f"{pre}_vsC"], d[f"{pre}_PC"]
        m = PC > 0.01 * PC.max()
        assert np.max(np.abs(A[m] - C[m]) / A.max()) < 1e-9


def test_sliding_vanishes_only_at_the_pitch_point():
    d = np.load(os.path.join(RESULTS, "run.npz"))
    A, PC = d["cyc_vsA"], d["cyc_PC"]
    assert A[0] < 1e-5                      # at C
    assert np.mean(A > 0.01 * A.max()) > 0.95   # everywhere else


def test_the_advantage_is_pure_contact_path_geometry():
    """Peak sliding ratio == peak |PC| ratio. The profile contributes nothing."""
    assert R["ratio_agreement"] < 1e-10
    assert R["peak_vs_ratio"] == pytest.approx(R["peak_PC_ratio"], rel=1e-10)


def test_anchoring_at_the_cusp_silently_picks_the_wrong_branch():
    """Regression test for the bug that made this whole module untrustworthy.

    At u = 0 both flanks sit on their own cusp, the derivatives vanish and the
    tangency condition is singular: the solver stops wherever it happens to be
    and every "root" there agrees to eight decimals. Anchoring the sweep on
    that noise picks a wrong lap of the hypocycloid, which then propagates
    smoothly through the entire mesh at a machine-zero residual -- nothing
    looks wrong, and the arc length is out by 62%. Anchoring in the
    well-conditioned interior fixes it.
    """
    rr = 0.18
    f1, f2 = M.epicycloid_face(R1, rr), M.hypocycloid_flank(R2, rr)
    g = M.MeshGeometry(R1, R2, f1, f2)
    ut = F.first_tip_param(f1, R1 + F.ADD)
    grid = np.linspace(1e-7, ut, 400)
    cf = R2 * (R1 + rr) / (R1 * (R2 - rr))

    def ratio(s):
        ok = ~np.isnan(s["th1"])
        return F.arc_along_path(f1, s["u"][ok]) / F.arc_along_path(f2, s["v"][ok])

    good = ratio(M.solve_mesh(g, grid))                       # anchor_frac = 0.30
    cusp = ratio(M.solve_mesh(g, grid, anchor_frac=1e-6))     # anchor at the cusp

    assert good == pytest.approx(cf, rel=1e-8)
    assert abs(cusp - cf) / cf > 1e-3, (
        "the cusp anchor no longer reproduces the failure; if the solver has "
        "been changed so this is genuinely safe, delete the test and say so")


def test_result_is_independent_of_output_grid_density():
    """A correct branch must not depend on how finely the caller samples."""
    rr = 0.18
    f1, f2 = M.epicycloid_face(R1, rr), M.hypocycloid_flank(R2, rr)
    g = M.MeshGeometry(R1, R2, f1, f2)
    ut = F.first_tip_param(f1, R1 + F.ADD)
    cf = R2 * (R1 + rr) / (R1 * (R2 - rr))
    for n in (200, 350, 700):
        s = M.solve_mesh(g, np.linspace(1e-7, ut, n))
        ok = ~np.isnan(s["th1"])
        got = F.arc_along_path(f1, s["u"][ok]) / F.arc_along_path(f2, s["v"][ok])
        assert got == pytest.approx(cf, rel=1e-7), f"grid density {n} changed the answer"

