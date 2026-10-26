HW2 Problem 1: Orbit Integrators

Goal
Build (or extend) a package with one driver for second-order systems x'' = a(x):

orbit(acc, x0, v0, t, method="leapfrog")

Return (x, v), with both arrays having shape (len(t),) + shape(x0). The first row must contain the initial position and velocity.

Reference Files
Read integrators.py and hw2_orbit_integrators.pdf (especially Problem1, Problem3 and Appendix A) before implementation. Treat both as read-only reference files.
- Reuse the existing integrate(f, y0, t, method) in integrators.py for Runge–Kutta methods.
- Implement Leapfrog using the kick-drift-kick (KDK) algorithm described in Appendix A in hw2_orbit_integrators.pdf.

Requirements
- All four methods — "euler", "rk2", "rk4", and "leapfrog" — should be selectable through the same orbit() interface.
- For Runge–Kutta methods, use the first-order state y = (x, v), with dy/dt = (v, acc(x)).
- acc(x) must return an array with the same shape as x.
- t must be a one-dimensional array of output times. Each step uses t[n+1] - t[n].
- For N Leapfrog steps, acc must be called exactly N + 1 times.
- Do not print, plot, or use hidden state.

Acceptance Tests
1. Convergence order: For the harmonic oscillator x'' = -x, with x(0) = 1 and v(0) = 0, integrate to t = 10 using h = 0.02 and h = 0.01. Compare the position error at t = 10 with the exact solution cos(10). The ratio of the error |x(10) - cos(10)| must be within 20% of 2 for Euler, 4 for RK2, 4 for Leapfrog, and 16 for RK4.
2. Leapfrog energy error: For the same oscillator, use h = 0.01 over t = 0 to 100. Calculate E = (v**2 + x**2) / 2 at every step. The maximum relative energy error |ΔE / E_0| must be within 10% of h**2 / 4 = 2.5 * 10**(-5).
3. Time reversibility: Use the Kepler orbit and initial conditions from Problem 3 in hw2_orbit_integrators.pdf, with eccentricity e = 0.5 and 100 steps per orbit. Integrate one orbit forward, negate the velocity, and integrate for one more orbit. The Leapfrog final position residual must be less than 1e-12; the RK4 residual must be greater than 1e-6.
4. One force evaluation per step: Verify that acc is called exactly N + 1 times for N Leapfrog steps.

Acceptance Criteria
- All four methods use the same orbit() interface.
- All four methods pass the required acceptance tests.
- Tests run with uv run pytest -q.
- The implementation works from a clean clone.
