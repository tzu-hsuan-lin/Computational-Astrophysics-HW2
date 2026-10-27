#Computational Astrophysics — Homework 2
This repository contains my solutions to Homework 2, covering numerical integration interface, Kepler orbit convergence, long-term energy and angular-momentum drift, and the Sun–Jupiter–Trojan system.

##Requirements
•Python 3.11 or later
•uv
•NumPy
•Matplotlib
•SciPy
•pytest

##Environment Setup
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt

##Run Tests

###Problem 1 — Integrator Suite
The orbit integrators are implemented in orbit.py. Run the acceptance tests using the command:
uv run pytest -q tests/test_orbit.py
Run the following scripts from the repository root.
###Problem 2 — Kepler Convergence Study
uv run python kepler_convergence.py
This script generates:
	figures/convergence.png
	figures/convergence_10_orbits.png
###Problem 3 — Energy and Angular-Momentum Drift
uv run python problem3.py
This script generates:
	figures/drift_energy.png
	figures/drift_angmom.png
	figures/euler_orbit.png
###Problem 4 — Sun–Jupiter–Trojan System
uv run python 3_body_orbit.py
This script generates:
	figures/trojan_orbits.png
	figures/relative_longitude.png

##Repository Structure
•orbit.py — Orbit integration interface and numerical integrators.
•integrators.py — Existing integration utilities.
•tests/test_orbit.py — Acceptance tests for Problem 1.
•kepler_convergence.py — Problem 2 convergence study.
•problem3.py — Problem 3 long-term energy and angular-momentum analysis.
•3_body_orbit.py — Problem 4 Sun–Jupiter–Trojan simulation.
•figures/ — Generated figures used in the report.
•report.pdf — Homework2 report.
•AI_LOG.md — Record of AI usage during the assignment.

##Report
See report.pdf for the discussion of Problem1 to Problem4.

##AI Usage
See AI_LOG.md for the record of AI assistance, prompts, and verification of the generated code and results.

