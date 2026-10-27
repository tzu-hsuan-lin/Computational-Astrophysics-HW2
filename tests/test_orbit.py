"""Acceptance tests for Problem 1: Orbit Integrators.

Requirements defined in SPEC.md and hw2_orbit_integrators.pdf:
1. Convergence order: For the harmonic oscillator x'' = -x, with x(0) = 1 and v(0) = 0,
   integrate to t = 10 using h = 0.02 and h = 0.01. Compare the position error at t = 10
   with the exact solution cos(10). The ratio of the error |x(10) - cos(10)| must be
   within 20% of 2 for Euler, 4 for RK2, 4 for Leapfrog, and 16 for RK4.
2. Leapfrog energy error: For the same oscillator, use h = 0.01 over t = 0 to 100.
   Calculate E = (v**2 + x**2) / 2 at every step. The maximum relative energy error
   |ΔE / E_0| must be within 10% of h**2 / 4 = 2.5 * 10**(-5).
3. Time reversibility: Use the Kepler orbit and initial conditions from Problem 3
   in hw2_orbit_integrators.pdf, with eccentricity e = 0.5 and 100 steps per orbit.
   Integrate one orbit forward, negate the velocity, and integrate for one more orbit.
   The Leapfrog final position residual must be less than 1e-12; the RK4 residual
   must be greater than 1e-6.
4. One force evaluation per step: Verify that acc is called exactly N + 1 times
   for N Leapfrog steps.
"""

import numpy as np
import pytest

from orbit import orbit


# =============================================================================
# Interface Contract & Shape Verification
# =============================================================================
@pytest.mark.parametrize("method", ["euler", "rk2", "rk4", "leapfrog"])
def test_orbit_interface_contract(method):
    """Verify common interface, output shapes, and initial state for all methods."""
    def acc(x):
        return -x

    # Test with 2D vector state
    x0 = np.array([1.5, 0.0])
    v0 = np.array([0.0, 0.5])
    t = np.linspace(0.0, 1.0, 11)

    x, v = orbit(acc, x0, v0, t, method=method)

    # Return shape must be (len(t),) + shape(x0)
    assert x.shape == (len(t),) + x0.shape
    assert v.shape == (len(t),) + v0.shape

    # Row 0 must contain initial position and velocity
    np.testing.assert_allclose(x[0], x0)
    np.testing.assert_allclose(v[0], v0)


def test_orbit_default_method():
    """Verify that default method is 'leapfrog'."""
    def acc(x):
        return -x

    x0 = np.array([1.0, 0.0])
    v0 = np.array([0.0, 1.0])
    t = np.linspace(0.0, 1.0, 11)

    x_default, v_default = orbit(acc, x0, v0, t)
    x_leapfrog, v_leapfrog = orbit(acc, x0, v0, t, method="leapfrog")

    np.testing.assert_allclose(x_default, x_leapfrog)
    np.testing.assert_allclose(v_default, v_leapfrog)


# =============================================================================
# Acceptance Requirement 1: Convergence Order
# =============================================================================
@pytest.mark.parametrize(
    ("method", "expected_ratio"),
    [
        ("euler", 2.0),
        ("rk2", 4.0),
        ("leapfrog", 4.0),
        ("rk4", 16.0),
    ],
)
def test_convergence_order(method, expected_ratio):
    """Verify convergence order for Euler, RK2, Leapfrog, and RK4 on harmonic oscillator.

    Integrate x'' = -x with x(0) = 1, v(0) = 0 to t = 10 using h = 0.02 and h = 0.01.
    Compare position error |x(10) - cos(10)|.
    The ratio of the error |x(10) - cos(10)| must be within 20% of:
      - 2  for Euler    (order 1, 2^1 = 2)
      - 4  for RK2      (order 2, 2^2 = 4)
      - 4  for Leapfrog (order 2, 2^2 = 4)
      - 16 for RK4      (order 4, 2^4 = 16)
    """
    def acc(x):
        return -x

    x0 = 1.0
    v0 = 0.0

    # Grid for h = 0.02 (500 steps) and h = 0.01 (1000 steps) to t = 10
    t1 = np.linspace(0.0, 10.0, int(round(10.0 / 0.02)) + 1)
    t2 = np.linspace(0.0, 10.0, int(round(10.0 / 0.01)) + 1)

    x1, _ = orbit(acc, x0, v0, t1, method=method)
    x2, _ = orbit(acc, x0, v0, t2, method=method)

    exact_x10 = np.cos(10.0)
    err1 = float(np.abs(np.squeeze(x1[-1]) - exact_x10))
    err2 = float(np.abs(np.squeeze(x2[-1]) - exact_x10))

    ratio = err1 / err2
    assert ratio == pytest.approx(expected_ratio, rel=0.20)


# =============================================================================
# Acceptance Requirement 2: Leapfrog Energy Error Bounded and Predictable
# =============================================================================
def test_leapfrog_energy_error():
    """Verify Leapfrog energy error is bounded and matches theoretical oscillation.

    For x'' = -x, x(0) = 1, v(0) = 0, h = 0.01 over t in [0, 100].
    Energy E = (v**2 + x**2) / 2 sampled at every step.
    The maximum relative energy error |ΔE / E_0| must be within 10%
    of h**2 / 4 = 2.5 * 10**(-5).
    """
    def acc(x):
        return -x

    x0 = 1.0
    v0 = 0.0
    h = 0.01
    t = np.linspace(0.0, 100.0, int(round(100.0 / h)) + 1)

    x, v = orbit(acc, x0, v0, t, method="leapfrog")

    # E = (v**2 + x**2) / 2 at every step; E0 = (0**2 + 1**2) / 2 = 0.5
    E = 0.5 * (v**2 + x**2)
    E0 = 0.5 * (v0**2 + x0**2)

    rel_error = np.abs((E - E0) / E0)
    max_rel_error = float(np.max(rel_error))

    expected_bound = (h**2) / 4.0  # 2.5e-5
    assert max_rel_error == pytest.approx(expected_bound, rel=0.10)


# =============================================================================
# Acceptance Requirement 3: Time Reversibility on Kepler Orbit
# =============================================================================
def test_time_reversibility():
    """Verify time-reversibility on an eccentric Kepler orbit (Problem 3 ICs).

    Eccentric Kepler orbit: e = 0.5, semi-major axis a = 1, period P = 2*pi.
    Start at apocenter:
      r0 = (1 + e, 0) = (1.5, 0)
      v0 = (0, sqrt((1 - e) / (1 + e))) = (0, 1 / sqrt(3))
    Use 100 steps per orbit.
    Integrate one orbit forward, negate the velocity, and integrate for one more orbit.
    - Leapfrog final position residual must be less than 1e-12.
    - RK4 residual must be greater than 1e-6 (RK4 is not time-reversible).
    """
    def kepler_acc(x):
        r3 = (x[0]**2 + x[1]**2)**1.5
        return -x / r3

    e = 0.5
    x0 = np.array([1.0 + e, 0.0])
    v0 = np.array([0.0, np.sqrt((1.0 - e) / (1.0 + e))])

    # 100 steps per orbit -> 101 points over period 2*pi
    t_orbit = np.linspace(0.0, 2.0 * np.pi, 101)

    # 1. Leapfrog reversibility
    x1_lf, v1_lf = orbit(kepler_acc, x0, v0, t_orbit, method="leapfrog")
    x2_lf, _ = orbit(kepler_acc, x1_lf[-1], -v1_lf[-1], t_orbit, method="leapfrog")
    lf_residual = float(np.linalg.norm(x2_lf[-1] - x0))

    assert lf_residual < 1e-12, f"Leapfrog residual {lf_residual} exceeds 1e-12"

    # 2. RK4 irreversibility check
    x1_rk4, v1_rk4 = orbit(kepler_acc, x0, v0, t_orbit, method="rk4")
    x2_rk4, _ = orbit(kepler_acc, x1_rk4[-1], -v1_rk4[-1], t_orbit, method="rk4")
    rk4_residual = float(np.linalg.norm(x2_rk4[-1] - x0))

    assert rk4_residual > 1e-6, f"RK4 residual {rk4_residual} unexpectedly <= 1e-6"


# =============================================================================
# Acceptance Requirement 4: One Force Evaluation per Step
# =============================================================================
def test_one_force_evaluation_per_step():
    """Verify that acc is called exactly N + 1 times for N Leapfrog steps.

    The kick-drift-kick scheme carries a = acc(x) between steps so the closing
    kick of step n and opening kick of step n+1 share one evaluation.
    """
    call_count = 0

    def counting_acc(x):
        nonlocal call_count
        call_count += 1
        return -x

    N = 100
    t = np.linspace(0.0, 10.0, N + 1)
    x0 = 1.0
    v0 = 0.0

    orbit(counting_acc, x0, v0, t, method="leapfrog")
    assert call_count == N + 1, (
        f"Expected {N + 1} force evaluations for {N} steps, got {call_count}"
    )
