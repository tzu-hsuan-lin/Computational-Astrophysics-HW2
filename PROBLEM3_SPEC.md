Problem 3: Energy and Angular-Momentum Drift
Implement Problem 3 from hw2_orbit_integrators.pdf, following the existing SPEC.md and reusing the code in orbit.py and integrators.py.

1. Initial Conditions
Use the eccentric Kepler orbit:
* Eccentricity: e = 0.5
* Semi-major axis: a = 1
* Orbital period: P = 2*pi
* Initial energy: E0 = -1/2
* Initial angular momentum: L0 = sqrt(3)/2
* Initial position: r0 = (1.5, 0)
* Initial velocity: v0 = (0, 1/sqrt(3))
At every integration step, calculate:
* Energy: E = 0.5*(vx**2 + vy**2) - 1/r
* Angular momentum: L = x*vy - y*vx
For each orbit, record:
1. Maximum absolute relative energy error |ΔE/E_0| within the orbit.
2. Absolute relative energy error at the end of the orbit.
3. Maximum absolute relative angular-momentum error |ΔL/L_0| within the orbit.
Accumulate statistics inside the integration loop. Do not store all states for the long simulations.

2. Required Simulations
Run all five cases:
Integrator	Steps per orbit	Number of orbits
Leapfrog	100	10,000
RK4	100	10,000
Leapfrog	400	1,000
RK4	400	1,000
Forward Euler	400	100

3. Required Figures
Save the figures in figures/:
* drift_energy.png: maximum relative energy error per orbit versus orbit number, using log-log axes and overlaying all runs.
* drift_angmom.png: maximum relative angular-momentum error per orbit versus orbit number, overlaying all runs.
* euler_orbit.png: the Euler trajectory in the x-y plane, showing its outward spiral.

4. Acceptance Criteria
Calculate and report the actual numerical values for every criterion.
* Leapfrog, 100 steps/orbit: Maximum relative energy error over all 10,000 orbits must be less than 2e-2. The maximum over orbits 9501–10,000 must agree with the maximum over orbits 1–500 to within 5%.
* Leapfrog, 400 steps/orbit: Maximum relative energy error over 1,000 orbits must be less than 1e-3. The ratio of the two leapfrog energy-error bounds must be 16 within 20%.
* Leapfrog angular momentum: Relative angular-momentum error must remain below 1e-12 throughout both runs.
* RK4: Fit a log-log slope to the end-of-orbit relative energy error over orbits 10–1,000. The slope must be 1.0 ± 0.1 at both resolutions. At orbit 1,000, the ratio of RK4 end-of-orbit energy errors (100 steps/orbit divided by 400 steps/orbit) must be 1024 within 20%. Report RK4 angular-momentum error as well.
* Forward Euler: Relative energy error must exceed 0.1 within the first orbit, and the trajectory must show an outward spiral.

5. Discussion
Use the measured results to explain:
1. Why RK4 exhibits secular energy drift while leapfrog's energy error stays bounded.
2. What quadrupling the steps per orbit changes quantitatively for each method.
3. Why explicit Runge–Kutta methods, regardless of order, do not generally preserve the symplectic structure preserved by leapfrog.

6. Implementation Constraints
* Read the existing code, tests, and SPEC.md before implementing.
* Reuse existing functions and integrators. Do not rewrite integrator algorithms.
* Do not modify existing tests or unrelated files without asking first.
* Work only in this standalone test directory. Do not commit changes.
* Run the existing tests and all five simulations.
* Do not claim an acceptance criterion passes unless it has been checked using actual computed values.
* Report missing dependencies, failures, or unmet criteria honestly.
In the final report, list the changed files, commands executed, measured acceptance results, generated figure paths, and any failures or limitations.

