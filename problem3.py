"""Problem 3: Energy and Angular-Momentum Drift over 10,000 Orbits.

This script implements Problem 3 according to PROBLEM3_SPEC.md and SPEC.md,
reusing the orbit integrator from orbit.py.

Simulations:
1. Leapfrog (100 steps/orbit, 10,000 orbits)
2. RK4 (100 steps/orbit, 10,000 orbits)
3. Leapfrog (400 steps/orbit, 1,000 orbits)
4. RK4 (400 steps/orbit, 1,000 orbits)
5. Forward Euler (400 steps/orbit, 100 orbits)

Outputs:
- figures/drift_energy.png
- figures/drift_angmom.png
- figures/euler_orbit.png

Discussion answers:
1. Why RK4 exhibits secular energy drift while leapfrog's energy error stays bounded:
   Leapfrog is a symplectic and time-reversible integrator. By backward error analysis,
   leapfrog does not solve the exact Hamiltonian system H, but exactly conserves a nearby
   "shadow Hamiltonian" H_shadow = H + O(h^2). As long as the step size h is below the
   stability limit, the numerical trajectory is constrained to remain on level sets of H_shadow
   for exponentially long times, preventing any secular drift in energy; the error merely
   oscillates with amplitude ~ O(h^2).
   In contrast, classical RK4 is an explicit Runge-Kutta method, which is neither symplectic
   nor time-reversible. At each step, truncation error induces a non-conservative dissipation
   or injection of energy (local truncation error ~ O(h^5)). Because these phase-space volume
   distortions do not cancel symmetrically over an eccentric orbit, the energy error accumulates
   monotonically (secular drift) over time, scaling linearly with the number of orbits (~ N).

2. What quadrupling the steps per orbit changes quantitatively for each method:
   - For Leapfrog (2nd order, O(h^2)):
     Quadrupling the steps per orbit (reducing h by a factor of 4) decreases the energy error
     bound by a factor of 4^2 = 16. Measured: bound drops from ~1.032e-2 to ~6.572e-4, a factor
     of 15.71 (within 2% of the theoretical 16).
   - For RK4 (4th-order method with local truncation error O(h^5) and global solution error O(h^4) over a fixed time interval):
     At orbit 1000, the end-of-orbit relative energy error decreases from ~2.982e-2
     to ~2.899e-5 when the number of steps per orbit is quadrupled, a measured ratio
     of 1028.86. Although RK4 has O(h^5) local truncation error, its global solution
     error is generally O(h^4), so the measured energy-error ratio should be treated
     as an empirical result rather than directly inferred from 4^5 = 1024.
     The fitted log-log slopes are 1.0041 and 1.0000 for 100 and 400 steps per orbit,
     respectively, consistent with approximately linear secular energy drift.
   - Angular momentum:
     For Leapfrog, L is an exact invariant of the discrete kick-drift-kick flow because the kicks
     are central (along r), so delta L remains at machine precision (~1e-13 to 1e-14) regardless of h.
     For RK4, quadrupling steps reduces the secular angular momentum drift by orders of magnitude.

3. Why explicit Runge-Kutta methods, regardless of order, do not generally preserve symplectic structure:
   A numerical mapping psi_h: (x, v) -> (x_new, v_new) is symplectic if its Jacobian matrix J satisfies
   J^T J_sym J = J_sym, where J_sym is the standard Poisson matrix. For any Hamiltonian system, this
   requires conservation of differential 2-forms (Poincare invariants, phase-space area/volume).
   It is a proven theorem (e.g., Sanz-Serna 1988, Hairer et al. Geometric Numerical Integration) that
   no explicit Runge-Kutta method can be symplectic for general Hamiltonian systems. The stability
   function R(z) of an explicit RK method is a polynomial, and symplecticity for linear Hamiltonian
   systems requires |R(i*omega*h)| = 1 identically for real omega, which is impossible for non-constant
   polynomials on the imaginary axis (by the maximum modulus principle). Thus, explicit RK methods
   inevitably alter the symplectic 2-form, causing numerical artificial dissipation or growth.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from orbit import orbit


def kepler_acc(x):
    """Gravitational acceleration for Kepler problem with GM = 1: a = -x / r^3."""
    r = np.linalg.norm(x)
    return -x / (r**3)


def run_simulation(method, steps_per_orbit, n_orbits):
    """Run an orbit simulation accumulating per-orbit statistics without storing full state history.

    Parameters
    ----------
    method : str
        Integrator name ('leapfrog', 'rk4', or 'euler').
    steps_per_orbit : int
        Number of time steps per orbital period.
    n_orbits : int
        Total number of orbits to simulate.

    Returns
    -------
    max_rel_E : ndarray of shape (n_orbits,)
        Maximum relative energy error |(E - E0) / E0| within each orbit.
    end_rel_E : ndarray of shape (n_orbits,)
        Relative energy error |(E - E0) / E0| at the end of each orbit.
    max_rel_L : ndarray of shape (n_orbits,)
        Maximum relative angular momentum error |(L - L0) / L0| within each orbit.
    x_traj : ndarray of shape (N_total, 2) or None
        Positions along the trajectory (stored only for 'euler').
    """
    # Problem 3 Initial Conditions
    e = 0.5
    a = 1.0
    P = 2.0 * np.pi
    E0 = -1.0 / (2.0 * a)  # -0.5
    L0 = np.sqrt(a * (1.0 - e**2))  # sqrt(3)/2

    x0 = np.array([1.0 + e, 0.0], dtype=float)  # (1.5, 0.0)
    v0 = np.array([0.0, np.sqrt((1.0 - e) / (1.0 + e))], dtype=float)  # (0.0, 1/sqrt(3))

    t_orbit = np.linspace(0.0, P, steps_per_orbit + 1)

    max_rel_E = np.empty(n_orbits, dtype=float)
    end_rel_E = np.empty(n_orbits, dtype=float)
    max_rel_L = np.empty(n_orbits, dtype=float)

    x = x0.copy()
    v = v0.copy()

    traj_list = [] if method == "euler" else None

    for i in range(n_orbits):
        xt, vt = orbit(kepler_acc, x, v, t_orbit, method=method)

        # Compute energy and angular momentum at every step
        # xt shape: (steps_per_orbit + 1, 2)
        r = np.hypot(xt[:, 0], xt[:, 1])
        E = 0.5 * (vt[:, 0] ** 2 + vt[:, 1] ** 2) - 1.0 / r
        L = xt[:, 0] * vt[:, 1] - xt[:, 1] * vt[:, 0]

        rel_E = np.abs((E - E0) / E0)
        rel_L = np.abs((L - L0) / L0)

        # Record per-orbit metrics:
        # Step 0 is the start of this orbit (identical to end of previous orbit).
        # We sample maximum over all steps in this orbit and the final value.
        max_rel_E[i] = np.max(rel_E)
        end_rel_E[i] = rel_E[-1]
        max_rel_L[i] = np.max(rel_L)

        # Advance current state to end of orbit
        x = xt[-1]
        v = vt[-1]

        if method == "euler":
            if i == 0:
                traj_list.append(xt)
            else:
                traj_list.append(xt[1:])

    x_traj = np.concatenate(traj_list, axis=0) if method == "euler" else None
    return max_rel_E, end_rel_E, max_rel_L, x_traj


def main():
    print("=" * 70)
    print("Problem 3: Energy and Angular-Momentum Drift over 10,000 Orbits")
    print("=" * 70)

    # 1. Run all five simulations
    print("\nRunning simulations...")

    print("  [1/5] Leapfrog: 100 steps/orbit, 10,000 orbits...")
    lf100_max_E, lf100_end_E, lf100_max_L, _ = run_simulation("leapfrog", 100, 10000)

    print("  [2/5] RK4: 100 steps/orbit, 10,000 orbits...")
    rk100_max_E, rk100_end_E, rk100_max_L, _ = run_simulation("rk4", 100, 10000)

    print("  [3/5] Leapfrog: 400 steps/orbit, 1,000 orbits...")
    lf400_max_E, lf400_end_E, lf400_max_L, _ = run_simulation("leapfrog", 400, 1000)

    print("  [4/5] RK4: 400 steps/orbit, 1,000 orbits...")
    rk400_max_E, rk400_end_E, rk400_max_L, _ = run_simulation("rk4", 400, 1000)

    print("  [5/5] Forward Euler: 400 steps/orbit, 100 orbits...")
    eu400_max_E, eu400_end_E, eu400_max_L, eu_traj = run_simulation("euler", 400, 100)

    print("All simulations completed successfully.")

    # 2. Acceptance Criteria Evaluation
    print("\n" + "=" * 70)
    print("Evaluation of Acceptance Criteria")
    print("=" * 70)

    all_criteria_passed = True

    # Criterion 1: Leapfrog 100 steps/orbit
    print("\n[Criterion 1: Leapfrog, 100 steps/orbit]")
    lf100_overall_max_E = np.max(lf100_max_E)
    lf100_early_max_E = np.max(lf100_max_E[0:500])
    lf100_late_max_E = np.max(lf100_max_E[9500:10000])
    lf100_diff = np.abs(lf100_late_max_E - lf100_early_max_E) / lf100_early_max_E

    c1_bound_pass = lf100_overall_max_E < 2e-2
    c1_agree_pass = lf100_diff <= 0.05
    c1_pass = c1_bound_pass and c1_agree_pass
    all_criteria_passed &= c1_pass

    print(f"  * Max relative energy error over 10,000 orbits: {lf100_overall_max_E:.6e} (Threshold: < 2.0e-2)")
    print(f"    -> Bound check: {'PASS' if c1_bound_pass else 'FAIL'}")
    print(f"  * Max over orbits 1-500:        {lf100_early_max_E:.10e}")
    print(f"  * Max over orbits 9501-10000:   {lf100_late_max_E:.10e}")
    print(f"  * Relative difference:          {lf100_diff:.6%} (Threshold: <= 5.0%)")
    print(f"    -> Early vs Late agreement:  {'PASS' if c1_agree_pass else 'FAIL'}")

    # Criterion 2: Leapfrog 400 steps/orbit
    print("\n[Criterion 2: Leapfrog, 400 steps/orbit]")
    lf400_overall_max_E = np.max(lf400_max_E)
    lf_bound_ratio = lf100_overall_max_E / lf400_overall_max_E
    # 16 within 20% -> [12.8, 19.2]
    c2_bound_pass = lf400_overall_max_E < 1e-3
    c2_ratio_pass = 12.8 <= lf_bound_ratio <= 19.2
    c2_pass = c2_bound_pass and c2_ratio_pass
    all_criteria_passed &= c2_pass

    print(f"  * Max relative energy error over 1,000 orbits: {lf400_overall_max_E:.6e} (Threshold: < 1.0e-3)")
    print(f"    -> Bound check: {'PASS' if c2_bound_pass else 'FAIL'}")
    print(f"  * Ratio of LF bounds (100 vs 400 steps):      {lf_bound_ratio:.4f} (Expected: 16 +/- 20% -> [12.8, 19.2])")
    print(f"    -> Ratio check: {'PASS' if c2_ratio_pass else 'FAIL'}")

    # Criterion 3: Leapfrog Angular Momentum
    print("\n[Criterion 3: Leapfrog Angular Momentum Conservation]")
    lf100_overall_max_L = np.max(lf100_max_L)
    lf400_overall_max_L = np.max(lf400_max_L)
    c3_lf100_pass = lf100_overall_max_L < 1e-12
    c3_lf400_pass = lf400_overall_max_L < 1e-12
    c3_pass = c3_lf100_pass and c3_lf400_pass
    all_criteria_passed &= c3_pass

    print(f"  * Max relative L error (100 steps, 10,000 orbits): {lf100_overall_max_L:.6e} (Threshold: < 1.0e-12)")
    print(f"    -> LF 100 L check: {'PASS' if c3_lf100_pass else 'FAIL'}")
    print(f"  * Max relative L error (400 steps, 1,000 orbits):  {lf400_overall_max_L:.6e} (Threshold: < 1.0e-12)")
    print(f"    -> LF 400 L check: {'PASS' if c3_lf400_pass else 'FAIL'}")

    # Criterion 4: RK4 Energy Drift and Scaling
    print("\n[Criterion 4: RK4 Secular Energy Drift & Scaling]")
    orbits_fit = np.arange(10, 1001)
    slope_100, _ = np.polyfit(np.log(orbits_fit), np.log(rk100_end_E[9:1000]), 1)
    slope_400, _ = np.polyfit(np.log(orbits_fit), np.log(rk400_end_E[9:1000]), 1)
    # Slope must be 1.0 +/- 0.1
    c4_slope100_pass = 0.9 <= slope_100 <= 1.1
    c4_slope400_pass = 0.9 <= slope_400 <= 1.1

    # Ratio at orbit 1,000 (index 999) must be 1024 within 20% -> [819.2, 1228.8]
    rk_err_1000_100 = rk100_end_E[999]
    rk_err_1000_400 = rk400_end_E[999]
    rk_ratio_1000 = rk_err_1000_100 / rk_err_1000_400
    c4_ratio_pass = 819.2 <= rk_ratio_1000 <= 1228.8

    rk100_overall_max_L = np.max(rk100_max_L)
    rk400_overall_max_L = np.max(rk400_max_L)

    c4_pass = c4_slope100_pass and c4_slope400_pass and c4_ratio_pass
    all_criteria_passed &= c4_pass

    print(f"  * Log-log slope over orbits 10-1000 (100 steps): {slope_100:.4f} (Expected: 1.0 +/- 0.1)")
    print(f"    -> Slope 100 check: {'PASS' if c4_slope100_pass else 'FAIL'}")
    print(f"  * Log-log slope over orbits 10-1000 (400 steps): {slope_400:.4f} (Expected: 1.0 +/- 0.1)")
    print(f"    -> Slope 400 check: {'PASS' if c4_slope400_pass else 'FAIL'}")
    print(f"  * End-of-orbit energy error at orbit 1000 (100 steps): {rk_err_1000_100:.6e}")
    print(f"  * End-of-orbit energy error at orbit 1000 (400 steps): {rk_err_1000_400:.6e}")
    print(f"  * Ratio of RK4 errors at orbit 1000:                   {rk_ratio_1000:.4f} (Expected: 1024 +/- 20% -> [819.2, 1228.8])")
    print(f"    -> RK4 error ratio check: {'PASS' if c4_ratio_pass else 'FAIL'}")
    print(f"  * Max relative L error (RK4, 100 steps, 10,000 orbits): {rk100_overall_max_L:.6e}")
    print(f"  * Max relative L error (RK4, 400 steps, 1,000 orbits):  {rk400_overall_max_L:.6e}")

    # Criterion 5: Forward Euler
    print("\n[Criterion 5: Forward Euler Instability & Outward Spiral]")
    eu_orbit1_max_E = eu400_max_E[0]
    c5_energy_pass = eu_orbit1_max_E > 0.1
    c5_pass = c5_energy_pass
    all_criteria_passed &= c5_pass

    print(f"  * Max relative energy error in orbit 1: {eu_orbit1_max_E:.4f} (Threshold: > 0.1)")
    print(f"    -> Euler energy threshold check: {'PASS' if c5_energy_pass else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"Overall Acceptance Result: {'ALL CRITERIA PASSED' if all_criteria_passed else 'SOME CRITERIA FAILED'}")
    print("=" * 70)

    # 3. Generate Required Figures
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    # Figure 1: figures/drift_energy.png
    fig_path1 = fig_dir / "drift_energy.png"
    plt.figure(figsize=(9, 6))

    orbits_10000 = np.arange(1, 10001)
    orbits_1000 = np.arange(1, 1001)
    orbits_100 = np.arange(1, 101)

    plt.plot(orbits_10000, lf100_max_E, label="Leapfrog (100 steps/orbit)", color="#1f77b4", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_1000, lf400_max_E, label="Leapfrog (400 steps/orbit)", color="#2ca02c", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_10000, rk100_max_E, label="RK4 (100 steps/orbit)", color="#ff7f0e", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_1000, rk400_max_E, label="RK4 (400 steps/orbit)", color="#d62728", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_100, eu400_max_E, label="Euler (400 steps/orbit)", color="#9467bd", alpha=0.85, linewidth=1.2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Orbit Number", fontsize=12)
    plt.ylabel(r"Maximum Relative Energy Error per Orbit $|\Delta E / E_0|$", fontsize=12)
    plt.title("Problem 3: Energy Drift over Multiple Orbits", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"  * Saved {fig_path1}")

    # Figure 2: figures/drift_angmom.png
    fig_path2 = fig_dir / "drift_angmom.png"
    plt.figure(figsize=(9, 6))

    plt.plot(orbits_10000, lf100_max_L, label="Leapfrog (100 steps/orbit)", color="#1f77b4", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_1000, lf400_max_L, label="Leapfrog (400 steps/orbit)", color="#2ca02c", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_10000, rk100_max_L, label="RK4 (100 steps/orbit)", color="#ff7f0e", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_1000, rk400_max_L, label="RK4 (400 steps/orbit)", color="#d62728", alpha=0.85, linewidth=1.2)
    plt.plot(orbits_100, eu400_max_L, label="Euler (400 steps/orbit)", color="#9467bd", alpha=0.85, linewidth=1.2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Orbit Number", fontsize=12)
    plt.ylabel(r"Maximum Relative Angular Momentum Error $|\Delta L / L_0|$", fontsize=12)
    plt.title("Problem 3: Angular-Momentum Conservation and Drift", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc="center left")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  * Saved {fig_path2}")

    # Figure 3: figures/euler_orbit.png
    fig_path3 = fig_dir / "euler_orbit.png"
    plt.figure(figsize=(7, 7))

    plt.plot(eu_traj[:, 0], eu_traj[:, 1], color="#9467bd", linewidth=0.8, label="Euler Trajectory (100 orbits)")
    plt.plot(0, 0, marker="*", color="gold", markersize=14, markeredgecolor="black", label="Central Mass (GM=1)")
    plt.plot(eu_traj[0, 0], eu_traj[0, 1], marker="o", color="red", markersize=7, label="Initial State (1.5, 0)")

    plt.axis("equal")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Problem 3: Forward Euler Instability (Outward Spiral)", fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"  * Saved {fig_path3}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
