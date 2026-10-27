Problem 4 Specification — Sun, Jupiter, and a Trojan Asteroid
1. Objective
Implement Option A of Problem 4 in hw2_orbit_integrators.pdf.

Please read the following files before implementing:
orbit.py, integrators.py, hw2_orbit_integrators.pdf (especially Problem4 Option A and Appendix B)

Reuse the orbit() function and its existing leapfrog integrator in orbit.py. Do not implement a new leapfrog integrator or modify the existing integration methods unless necessary to satisfy Problem 4.
Use the existing direct-summation acceleration calculation described in Appendix B, adapting it only as needed for the Sun–Jupiter–test-particle system.
Do not modify any existing files.

2. Physical Setup
Use the normalized units
* (G=1)
* Total Sun–Jupiter mass (=1)
* Sun–Jupiter separation (=1)
* Jupiter's orbital period (P_J=2\pi)
* Angular frequency (\Omega=1)
Define
[\mu=\frac{m_J}{M_\odot+m_J}=9.537\times10^{-4}.]
The Sun and Jupiter have initial conditions
Body	Mass	Position	Velocity
Sun	(1-\mu)	((-\mu,0))	((0,-\mu))
Jupiter	(\mu)	((1-\mu,0))	((0,1-\mu))
The test particle has zero mass and does not affect the Sun or Jupiter.
Follow the acceleration prescription specified in Appendix B.
3. Test Particle Initial Conditions
Run all three cases for at least 200 Jupiter orbits, with time step (h\leq P_J/400).
Case 1: L4
[\mathbf r=\left(\frac12-\mu,\frac{\sqrt3}{2}\right), \mathbf v=\left(-\frac{\sqrt3}{2},\frac12-\mu\right).]
Case 2: Tadpole orbit
Set (\theta=40^\circ):
[\mathbf r=(\cos\theta,\sin\theta), \mathbf v=(-\sin\theta,\cos\theta).]
Case 3: Horseshoe orbit
Use the same initial conditions as Case 2, but set (\theta=25^\circ).
4. Rotating Frame
Transform positions into the frame that follows Jupiter's actual angular position,
[\phi_J(t)=\operatorname{atan2}(y_J,x_J).]
Rotate positions by (-\phi_J(t)) for the rotating-frame analysis and trajectory plots. Do not substitute (-\Omega t) for Jupiter's actual angle.
5. Required Diagnostics
For each test particle run, calculate and report:
* Relative longitude with respect to Jupiter.
* Barycentric radius.
* Closest approach to Jupiter, expressed in Hill radii, where [r_H=\left(\frac{\mu}{3}\right)^{1/3}\approx0.0683.]
* Jacobi energy, using inertial-frame quantities: [E_J=\frac12(v_x^2+v_y^2)-\frac{1-\mu}{r_1}-\frac{\mu}{r_2}-(xv_y-yv_x),] where (r_1) and (r_2) are the distances from the test particle to the Sun and Jupiter, respectively.
Also check conservation of the Sun–Jupiter pair's orbital energy.
6. Required Plots
1. Plot the three test particle trajectories in the frame co-rotating with Jupiter. Save the figure as figures/trojan_orbits.png.
2. Plot relative longitude against time for the three cases. Save the figure as figures/relative_longitude.png.
7. Jupiter Phase-Lag Test
Measure Jupiter's phase lag,
[\Delta\phi_J=\phi_J(t_{\rm end})-\Omega t_{\rm end},]
using two time steps, for example (h_1=P_J/400) and (h_2=P_J/1000).
Check whether the phase-lag error scales as (h^2): the ratio between the two phase-lag errors should agree with ((h_1/h_2)^2) to within 20%.
8. Acceptance Criteria
* L4: The particle remains within (10^{-3}) of the L4 position in the rotating frame throughout the run.
* 40° case: The relative longitude stays between (30^\circ) and (100^\circ), the barycentric radius stays between (0.96) and (1.04), and the closest approach exceeds (8r_H).
* 25° case: The relative longitude passes through (180^\circ), indicating a horseshoe orbit that reaches the far side of the Sun and visits the L5 side. The closest approach exceeds (4r_H), and the particle never enters Jupiter's Hill sphere.
* Jacobi energy: (|\Delta E_J/E_J|<10^{-5}) for all three test particle runs.
* Sun–Jupiter energy: (|\Delta E/E_0|<10^{-6}).
* Phase lag: The measured phase-lag error follows the expected second-order scaling within 20%.
9. File Handling
* Save the generated figures in the existing figures/ directory.
* Save the Problem 4 implementation as a new Python file, named 3_body_orbit.py.
* Do not modify any existing files.
* Report the paths of the newly created files when finished.
