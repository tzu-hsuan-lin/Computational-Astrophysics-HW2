"""Problem 4: Few-Body Application — Sun, Jupiter, and a Trojan Asteroid (Option A).

This script simulates the restricted three-body problem consisting of the Sun,
Jupiter, and a massless test particle in the inertial barycentric frame using
the leapfrog integrator from orbit.py and the direct-summation gravity routine
from Appendix B of hw2_orbit_integrators.pdf.

Physical setup (natural units G = 1, M_tot = 1, separation = 1):
  - Jupiter orbital period: P_J = 2*pi, Omega = 1
  - Mass ratio: mu = m_J / (M_Sun + m_J) = 9.537e-4
  - Sun: mass = 1 - mu, r0 = (-mu, 0), v0 = (0, -mu)
  - Jupiter: mass = mu, r0 = (1 - mu, 0), v0 = (0, 1 - mu)
  - Hill radius of Jupiter: r_H = (mu / 3)**(1/3) approx 0.06830

Test Particle Cases (each >= 200 Jupiter orbits at h <= P_J / 400):
  1. L4 Lagrange point (apex of equilateral triangle)
  2. Tadpole orbit (theta = 40 deg ahead of Jupiter)
  3. Horseshoe orbit (theta = 25 deg ahead of Jupiter)

Outputs:
  - figures/trojan_orbits.png: Trajectories in the frame co-rotating with Jupiter
  - figures/relative_longitude.png: Relative longitude ahead of Jupiter vs time
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from orbit import orbit


# =============================================================================
# 1. Physical Parameters and Acceleration
# =============================================================================

MU = 9.537e-4
M_SUN = 1.0 - MU
M_JUP = MU
P_J = 2.0 * np.pi
OMEGA = 1.0
R_H = (MU / 3.0) ** (1.0 / 3.0)  # approx 0.068301


def nbody_acc(x, m, eps=0.0):
    """Direct-summation gravitational acceleration from Appendix B.

    Parameters
    ----------
    x : ndarray of shape (N, 2)
        Positions of the N bodies.
    m : ndarray of shape (N,)
        Masses of the N bodies.
    eps : float, optional
        Plummer softening length (default is 0.0 for unsoftened point masses).

    Returns
    -------
    acc : ndarray of shape (N, 2)
        Gravitational acceleration for each body.
    """
    dx = x[None, :, :] - x[:, None, :]  # dx[i, j] = x[j] - x[i]
    r2 = np.sum(dx * dx, axis=-1) + eps * eps  # (N, N)
    np.fill_diagonal(r2, np.inf)  # exclude self-interaction
    return np.sum(dx * m[None, :, None] * (r2[:, :, None] ** -1.5), axis=1)


def compute_jacobi_energy(x_tp, v_tp, x_sun, x_jup):
    """Calculate Jacobi energy E_J of the test particle from inertial-frame quantities.

    E_J = 0.5 * (vx^2 + vy^2) - (1 - mu) / r1 - mu / r2 - (x * vy - y * vx)

    Parameters
    ----------
    x_tp : ndarray of shape (..., 2)
        Inertial positions of test particle.
    v_tp : ndarray of shape (..., 2)
        Inertial velocities of test particle.
    x_sun : ndarray of shape (..., 2)
        Inertial positions of Sun.
    x_jup : ndarray of shape (..., 2)
        Inertial positions of Jupiter.

    Returns
    -------
    E_J : ndarray of shape (...)
        Jacobi energy at each point in time.
    """
    v_sq = x_tp[..., 0] * 0.0 + np.sum(v_tp * v_tp, axis=-1)
    r1 = np.hypot(x_tp[..., 0] - x_sun[..., 0], x_tp[..., 1] - x_sun[..., 1])
    r2 = np.hypot(x_tp[..., 0] - x_jup[..., 0], x_tp[..., 1] - x_jup[..., 1])
    ang_mom = x_tp[..., 0] * v_tp[..., 1] - x_tp[..., 1] * v_tp[..., 0]
    return 0.5 * v_sq - (1.0 - MU) / r1 - MU / r2 - ang_mom


def compute_sun_jupiter_energy(x_sun, v_sun, x_jup, v_jup):
    """Calculate the total orbital energy of the Sun-Jupiter pair.

    E_pair = 0.5 * m_sun * v_sun^2 + 0.5 * m_jup * v_jup^2 - (m_sun * m_jup) / r_sj
    """
    v_sun_sq = np.sum(v_sun * v_sun, axis=-1)
    v_jup_sq = np.sum(v_jup * v_jup, axis=-1)
    r_sj = np.hypot(x_jup[..., 0] - x_sun[..., 0], x_jup[..., 1] - x_sun[..., 1])
    return 0.5 * M_SUN * v_sun_sq + 0.5 * M_JUP * v_jup_sq - (M_SUN * M_JUP) / r_sj


def to_rotating_frame(x_tp, x_jup):
    """Transform test particle positions into the frame co-rotating with Jupiter.

    Follows Jupiter's actual barycentric angle phi_J(t) = atan2(y_J, x_J).
    Jupiter is fixed along the +x axis at (1 - mu, 0).
    """
    phi_j = np.arctan2(x_jup[:, 1], x_jup[:, 0])
    c = np.cos(phi_j)
    s = np.sin(phi_j)
    xr = c * x_tp[:, 0] + s * x_tp[:, 1]
    yr = -s * x_tp[:, 0] + c * x_tp[:, 1]
    return xr, yr, phi_j


# =============================================================================
# 2. Simulation Driver
# =============================================================================

def run_test_particle_simulation(case_type, theta_deg=None, n_orbits=200, steps_per_orbit=400):
    """Run a 3-body simulation (Sun, Jupiter, Test Particle) with leapfrog.

    Parameters
    ----------
    case_type : {'l4', 'tadpole', 'horseshoe'}
        Test particle initial condition case.
    theta_deg : float, optional
        Initial angle ahead of Jupiter (for tadpole and horseshoe).
    n_orbits : int, default 200
        Number of Jupiter orbital periods.
    steps_per_orbit : int, default 400
        Number of integration steps per orbit.

    Returns
    -------
    dict with full trajectory and diagnostic quantities.
    """
    # Sun (index 0) and Jupiter (index 1) initial conditions
    r_sun = np.array([-MU, 0.0], dtype=float)
    v_sun = np.array([0.0, -MU], dtype=float)

    r_jup = np.array([1.0 - MU, 0.0], dtype=float)
    v_jup = np.array([0.0, 1.0 - MU], dtype=float)

    # Test particle (index 2) initial conditions
    if case_type == "l4":
        r_tp = np.array([0.5 - MU, np.sqrt(3.0) / 2.0], dtype=float)
        v_tp = np.array([-np.sqrt(3.0) / 2.0, 0.5 - MU], dtype=float)
    else:
        th = np.radians(theta_deg)
        r_tp = np.array([np.cos(th), np.sin(th)], dtype=float)
        v_tp = np.array([-np.sin(th), np.cos(th)], dtype=float)

    masses = np.array([M_SUN, M_JUP, 0.0], dtype=float)
    x0 = np.array([r_sun, r_jup, r_tp], dtype=float)
    v0 = np.array([v_sun, v_jup, v_tp], dtype=float)

    t = np.linspace(0.0, n_orbits * P_J, n_orbits * steps_per_orbit + 1)

    # Call the existing leapfrog integrator via orbit()
    def acc_fn(pos):
        return nbody_acc(pos, masses)

    x, v = orbit(acc_fn, x0, v0, t, method="leapfrog")

    # Extract body components
    x_sun, v_sun = x[:, 0, :], v[:, 0, :]
    x_jup, v_jup = x[:, 1, :], v[:, 1, :]
    x_tp, v_tp = x[:, 2, :], v[:, 2, :]

    # Transform to Jupiter co-rotating frame
    xr, yr, phi_j = to_rotating_frame(x_tp, x_jup)

    # Relative longitude ahead of Jupiter (degrees in [0, 360))
    lon_ahead = (np.degrees(np.arctan2(yr, xr))) % 360.0

    # Barycentric radius
    r_bary = np.hypot(x_tp[:, 0], x_tp[:, 1])

    # Distance to Jupiter
    dist_jup = np.hypot(x_tp[:, 0] - x_jup[:, 0], x_tp[:, 1] - x_jup[:, 1])
    closest_approach_rH = np.min(dist_jup) / R_H

    # Jacobi energy and relative error
    E_J = compute_jacobi_energy(x_tp, v_tp, x_sun, x_jup)
    delta_EJ_rel = np.abs((E_J - E_J[0]) / E_J[0])

    # Sun-Jupiter pair energy and relative error
    E_pair = compute_sun_jupiter_energy(x_sun, v_sun, x_jup, v_jup)
    E_pair_0 = -0.5 * M_SUN * M_JUP
    delta_E_pair_rel = np.abs((E_pair - E_pair_0) / E_pair_0)

    # Distance to L4 in rotating frame: L4 is fixed at (0.5 - mu, sqrt(3)/2)
    l4_xr = 0.5 - MU
    l4_yr = np.sqrt(3.0) / 2.0
    dist_l4 = np.hypot(xr - l4_xr, yr - l4_yr)

    return {
        "case_type": case_type,
        "t": t,
        "orbits": t / P_J,
        "x_tp": x_tp,
        "v_tp": v_tp,
        "x_sun": x_sun,
        "x_jup": x_jup,
        "xr": xr,
        "yr": yr,
        "phi_j": phi_j,
        "lon_ahead": lon_ahead,
        "r_bary": r_bary,
        "dist_jup": dist_jup,
        "closest_approach_rH": closest_approach_rH,
        "E_J": E_J,
        "max_delta_EJ_rel": np.max(delta_EJ_rel),
        "max_delta_E_pair_rel": np.max(delta_E_pair_rel),
        "dist_l4": dist_l4,
        "max_dist_l4": np.max(dist_l4),
    }


def measure_jupiter_phase_lag(n_orbits=200):
    """Measure Jupiter's phase lag error at two step sizes (P_J/400 and P_J/1000).

    Delta phi_J = phi_J(t_end) - Omega * t_end.
    Verify that phase lag error scales as h^2 within 20%.
    """
    t_end = n_orbits * P_J
    masses = np.array([M_SUN, M_JUP], dtype=float)
    x0 = np.array([[-MU, 0.0], [1.0 - MU, 0.0]], dtype=float)
    v0 = np.array([[0.0, -MU], [0.0, 1.0 - MU]], dtype=float)

    def acc_fn(pos):
        return nbody_acc(pos, masses)

    # Step size 1: h1 = P_J / 400
    steps1 = 400
    t1 = np.linspace(0.0, t_end, n_orbits * steps1 + 1)
    x1, _ = orbit(acc_fn, x0, v0, t1, method="leapfrog")
    phi1 = np.unwrap(np.arctan2(x1[:, 1, 1], x1[:, 1, 0]))[-1]
    lag1 = phi1 - OMEGA * t_end

    # Step size 2: h2 = P_J / 1000
    steps2 = 1000
    t2 = np.linspace(0.0, t_end, n_orbits * steps2 + 1)
    x2, _ = orbit(acc_fn, x0, v0, t2, method="leapfrog")
    phi2 = np.unwrap(np.arctan2(x2[:, 1, 1], x2[:, 1, 0]))[-1]
    lag2 = phi2 - OMEGA * t_end

    ratio = lag1 / lag2
    expected_ratio = (steps2 / steps1) ** 2  # (1000 / 400)^2 = 6.25
    rel_diff = np.abs(ratio - expected_ratio) / expected_ratio

    return {
        "n_orbits": n_orbits,
        "lag1_rad": lag1,
        "lag2_rad": lag2,
        "lag1_deg": np.degrees(lag1),
        "lag2_deg": np.degrees(lag2),
        "ratio": ratio,
        "expected_ratio": expected_ratio,
        "rel_diff": rel_diff,
        "passed": rel_diff <= 0.20,
    }


# =============================================================================
# 3. Main Runner and Acceptance Evaluation
# =============================================================================

def main():
    print("=" * 75)
    print("Problem 4: Sun-Jupiter-Trojan 3-Body Dynamics (Option A)")
    print("=" * 75)

    n_orbits = 200
    steps_per_orbit = 400

    print(f"\nRunning 3 test particle simulations ({n_orbits} Jupiter orbits, h = P_J/{steps_per_orbit})...")

    print("  [1/3] Case 1: L4 Lagrange Point...")
    res_l4 = run_test_particle_simulation("l4", n_orbits=n_orbits, steps_per_orbit=steps_per_orbit)

    print("  [2/3] Case 2: Tadpole Orbit (theta = 40 deg)...")
    res_tadpole = run_test_particle_simulation("tadpole", theta_deg=40.0, n_orbits=n_orbits, steps_per_orbit=steps_per_orbit)

    print("  [3/3] Case 3: Horseshoe Orbit (theta = 25 deg)...")
    res_horseshoe = run_test_particle_simulation("horseshoe", theta_deg=25.0, n_orbits=n_orbits, steps_per_orbit=steps_per_orbit)

    print("  Measuring Jupiter's phase lag for h = P_J/400 vs P_J/1000...")
    res_lag = measure_jupiter_phase_lag(n_orbits=n_orbits)

    print("\n" + "=" * 75)
    print("Acceptance Criteria Verification")
    print("=" * 75)

    all_passed = True

    # Criterion 1: L4
    c1_passed = res_l4["max_dist_l4"] < 1e-3
    all_passed &= c1_passed
    print("\n[Criterion 1: L4 Lagrange Point]")
    print(f"  * Maximum distance from L4 in rotating frame: {res_l4['max_dist_l4']:.6e} (Threshold: < 1.0e-3)")
    print(f"    -> Status: {'PASS' if c1_passed else 'FAIL'}")

    # Criterion 2: Tadpole Orbit (40 deg)
    lon_min_tad = np.min(res_tadpole["lon_ahead"])
    lon_max_tad = np.max(res_tadpole["lon_ahead"])
    rb_min_tad = np.min(res_tadpole["r_bary"])
    rb_max_tad = np.max(res_tadpole["r_bary"])
    ca_tad = res_tadpole["closest_approach_rH"]

    c2_lon_passed = (lon_min_tad > 30.0) and (lon_max_tad < 100.0)
    c2_rb_passed = (rb_min_tad >= 0.96) and (rb_max_tad <= 1.04)
    c2_ca_passed = ca_tad > 8.0
    c2_passed = c2_lon_passed and c2_rb_passed and c2_ca_passed
    all_passed &= c2_passed

    print("\n[Criterion 2: Tadpole Orbit (theta = 40 deg)]")
    print(f"  * Relative longitude range: [{lon_min_tad:.2f} deg, {lon_max_tad:.2f} deg] (Expected: inside (30 deg, 100 deg))")
    print(f"    -> Longitude check: {'PASS' if c2_lon_passed else 'FAIL'}")
    print(f"  * Barycentric radius range: [{rb_min_tad:.4f}, {rb_max_tad:.4f}] (Expected: inside [0.96, 1.04])")
    print(f"    -> Radius check:    {'PASS' if c2_rb_passed else 'FAIL'}")
    print(f"  * Closest approach to Jupiter: {ca_tad:.2f} r_H (Threshold: > 8 r_H)")
    print(f"    -> Closest approach check: {'PASS' if c2_ca_passed else 'FAIL'}")

    # Criterion 3: Horseshoe Orbit (25 deg)
    lon_min_hs = np.min(res_horseshoe["lon_ahead"])
    lon_max_hs = np.max(res_horseshoe["lon_ahead"])
    ca_hs = res_horseshoe["closest_approach_rH"]
    crosses_180 = (lon_min_hs < 180.0) and (lon_max_hs > 180.0)

    c3_cross_passed = crosses_180
    c3_ca_passed = ca_hs > 4.0
    c3_hill_passed = ca_hs > 1.0
    c3_passed = c3_cross_passed and c3_ca_passed and c3_hill_passed
    all_passed &= c3_passed

    print("\n[Criterion 3: Horseshoe Orbit (theta = 25 deg)]")
    print(f"  * Relative longitude range: [{lon_min_hs:.2f} deg, {lon_max_hs:.2f} deg]")
    print(f"    -> Passes through 180 deg (crosses far side of Sun to L5): {'YES (PASS)' if c3_cross_passed else 'NO (FAIL)'}")
    print(f"  * Closest approach to Jupiter: {ca_hs:.2f} r_H (Threshold: > 4 r_H)")
    print(f"    -> Closest approach check: {'PASS' if c3_ca_passed else 'FAIL'}")
    print(f"    -> Never enters Jupiter's Hill sphere (r > 1 r_H): {'PASS' if c3_hill_passed else 'FAIL'}")

    # Criterion 4: Jacobi Energy Conservation
    ej_l4 = res_l4["max_delta_EJ_rel"]
    ej_tad = res_tadpole["max_delta_EJ_rel"]
    ej_hs = res_horseshoe["max_delta_EJ_rel"]
    c4_passed = (ej_l4 < 1e-5) and (ej_tad < 1e-5) and (ej_hs < 1e-5)
    all_passed &= c4_passed

    print("\n[Criterion 4: Jacobi Energy Conservation (|Delta E_J / E_J| < 1.0e-5)]")
    print(f"  * Case 1 (L4):        {ej_l4:.6e} -> {'PASS' if ej_l4 < 1e-5 else 'FAIL'}")
    print(f"  * Case 2 (Tadpole):   {ej_tad:.6e} -> {'PASS' if ej_tad < 1e-5 else 'FAIL'}")
    print(f"  * Case 3 (Horseshoe): {ej_hs:.6e} -> {'PASS' if ej_hs < 1e-5 else 'FAIL'}")

    # Criterion 5: Sun-Jupiter Pair Energy Conservation
    esj_l4 = res_l4["max_delta_E_pair_rel"]
    esj_tad = res_tadpole["max_delta_E_pair_rel"]
    esj_hs = res_horseshoe["max_delta_E_pair_rel"]
    c5_passed = (esj_l4 < 1e-6) and (esj_tad < 1e-6) and (esj_hs < 1e-6)
    all_passed &= c5_passed

    print("\n[Criterion 5: Sun-Jupiter Pair Orbital Energy Conservation (|Delta E / E_0| < 1.0e-6)]")
    print(f"  * Max relative error across runs: {max(esj_l4, esj_tad, esj_hs):.6e}")
    print(f"    -> Status: {'PASS' if c5_passed else 'FAIL'}")

    # Criterion 6: Jupiter Phase Lag Scaling
    c6_passed = res_lag["passed"]
    all_passed &= c6_passed

    print("\n[Criterion 6: Jupiter Phase-Lag Error Scaling]")
    print(f"  * Phase lag at h1 = P_J/400:  {res_lag['lag1_deg']:.4f} deg ({res_lag['lag1_rad']:.6f} rad)")
    print(f"  * Phase lag at h2 = P_J/1000: {res_lag['lag2_deg']:.4f} deg ({res_lag['lag2_rad']:.6f} rad)")
    print(f"  * Measured ratio (lag1 / lag2): {res_lag['ratio']:.4f}")
    print(f"  * Theoretical ratio (h1/h2)^2: {res_lag['expected_ratio']:.4f}")
    print(f"  * Relative difference:         {res_lag['rel_diff']:.4%} (Threshold: <= 20.0%)")
    print(f"    -> Status: {'PASS' if c6_passed else 'FAIL'}")

    print("\n" + "=" * 75)
    print(f"Overall Acceptance Result: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")
    print("=" * 75)

    # =============================================================================
    # 4. Generate Required Figures
    # =============================================================================
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    # Figure 1: figures/trojan_orbits.png
    fig1_path = fig_dir / "trojan_orbits.png"
    plt.figure(figsize=(9, 9))

    # Plot Sun at (-mu, 0)
    plt.plot(-MU, 0.0, marker="*", color="gold", markersize=16, markeredgecolor="black", label=f"Sun (1 - mu = {M_SUN:.4f})")

    # Plot Jupiter at (1 - mu, 0)
    plt.plot(1.0 - MU, 0.0, marker="o", color="orange", markersize=10, markeredgecolor="black", label=f"Jupiter (mu = {MU:.2e})")

    # Draw Jupiter's Hill sphere (radius r_H)
    circle_theta = np.linspace(0.0, 2.0 * np.pi, 200)
    plt.plot(
        1.0 - MU + R_H * np.cos(circle_theta),
        R_H * np.sin(circle_theta),
        "--",
        color="gray",
        alpha=0.7,
        label=f"Hill Sphere ($r_H \\approx {R_H:.3f}$)",
    )

    # Plot Lagrange points L4 and L5
    plt.plot(0.5 - MU, np.sqrt(3.0) / 2.0, "x", color="black", markersize=8, markeredgewidth=2, label="L4")
    plt.plot(0.5 - MU, -np.sqrt(3.0) / 2.0, "+", color="black", markersize=8, markeredgewidth=2, label="L5")

    # Plot trajectories in the rotating frame
    plt.plot(res_horseshoe["xr"], res_horseshoe["yr"], color="#2ca02c", linewidth=0.9, alpha=0.85, label="Horseshoe Orbit (25 deg)")
    plt.plot(res_tadpole["xr"], res_tadpole["yr"], color="#1f77b4", linewidth=1.1, alpha=0.9, label="Tadpole Orbit (40 deg)")
    plt.plot(res_l4["xr"], res_l4["yr"], color="#d62728", linewidth=1.8, label="L4 Motion")

    plt.axis("equal")
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)
    plt.xlabel("Rotating Frame $x_r$", fontsize=12)
    plt.ylabel("Rotating Frame $y_r$", fontsize=12)
    plt.title("Problem 4 Option A: Trojan Asteroid Orbits in Jupiter Co-Rotating Frame", fontsize=13)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  * Saved {fig1_path}")

    # Figure 2: figures/relative_longitude.png
    fig2_path = fig_dir / "relative_longitude.png"
    plt.figure(figsize=(10, 6))

    plt.plot(res_horseshoe["orbits"], res_horseshoe["lon_ahead"], color="#2ca02c", linewidth=1.2, label="Horseshoe Orbit (25 deg)")
    plt.plot(res_tadpole["orbits"], res_tadpole["lon_ahead"], color="#1f77b4", linewidth=1.4, label="Tadpole Orbit (40 deg)")
    plt.plot(res_l4["orbits"], res_l4["lon_ahead"], color="#d62728", linewidth=1.5, label="L4 Orbit")

    # Add reference lines for key Lagrange points
    plt.axhline(60.0, color="gray", linestyle="--", alpha=0.7, label=r"L4 Longitude ($60^\circ$)")
    plt.axhline(180.0, color="purple", linestyle=":", alpha=0.7, label=r"Far Side of Sun / L3 ($180^\circ$)")
    plt.axhline(300.0, color="brown", linestyle="--", alpha=0.7, label=r"L5 Longitude ($300^\circ \equiv -60^\circ$)")

    plt.xlabel("Time [Jupiter Orbits $t / P_J$]", fontsize=12)
    plt.ylabel("Relative Longitude ahead of Jupiter [deg]", fontsize=12)
    plt.title("Problem 4 Option A: Test Particle Longitude Relative to Jupiter vs Time", fontsize=13)
    plt.ylim(0.0, 360.0)
    plt.yticks(np.arange(0, 361, 60))
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  * Saved {fig2_path}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
