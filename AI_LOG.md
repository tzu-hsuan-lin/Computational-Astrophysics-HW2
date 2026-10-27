# AI Usage Log

This log summarizes how AI tools were used to assist in completing Homework 2.

AI Agent: Antigravity CLI, GPT-5.6 Luna
---

## Problem 1: Integrator suite behind one interface

- Workflow Understanding:
  Pasted the Problem 1 statement in HW2.pdf and asked GPT to outline a clear, step-by-step problem-solving workflow (Step 1, Step 2, etc.) to gain a full understanding of the task before implementation.
- SPEC.md Generation:
  Provided the example SPEC.md from Lecture 4 slides (Appendix) alongside the problem prompt to GPT, generated a draft specification, and double-checked and modified the draft against the requirements in Problem 1.
- Tests and interface Generation:
  Asked Antigravity to write the tests first, then implement the orbit interface by following the SPEC.md, HW2.pdf (read Problem1 and Appendix A), and integrators.py.

- Prompt used in Antigravity:

  1. Work only within the current HW2 directory. Do not explore any unrelated folders or files. Only read the files explicitly required for this task. Please read SPEC.md and hw2_orbit_integrators.pdf, especially Problem 1, Appendix A, and Problem 3.
Your task is to write the acceptance tests for Problem 1 only. Do not implement orbit.py or modify integrators.py. Do not ask another agent to implement anything.
Before writing tests:
Inspect the existing repository structure and integrators.py to understand the current code and available interfaces.
Read SPEC.md and the assignment PDF carefully. Treat them as the requirements.
If any detail is ambiguous, explain the ambiguity before making assumptions.
Create tests/test_orbit.py with pytest tests for all four acceptance requirements specified in SPEC.md.
After writing the tests, show me the complete tests/test_orbit.py and briefly explain how each test corresponds to the assignment. Do not commit anything yet; I want to review the tests first.
Run the tests if possible, but do not modify production code to make them pass.

2. Work only within the current HW2 directory. Do not explore any unrelated folders or files. Only read the files explicitly required for this task.
Now implement Problem 1 of HW2.
Before starting, read SPEC.md, hw2_orbit_integrators.pdf (especially Problem 1 and Appendix A), and the existing integrators.py and tests/test_orbit.py.
Your task is to implement orbit.py only.
Requirements:
Implement the orbit() function according to SPEC.md and the assignment PDF.
Do not modify integrators.py, SPEC.md, or tests/test_orbit.py. Do not weaken, skip, or rewrite the acceptance tests to make them pass.
Keep the implementation focused on Problem 1; do not implement unrelated problems.
After implementing orbit.py:
Run the existing acceptance tests using uv run pytest -q tests/test_orbit.py.
If tests fail, investigate and fix orbit.py only. Do not change the tests or other protected files.
Report the test results, including which tests passed or failed, and explain any remaining failures honestly.
Do not commit anything yet. Show me the implementation and test results for review first.

---

## Problem 2: Kepler convergence study

- Workflow Understanding:
  Pasted the Problem2 statement in HW2.pdf and asked GPT to outline a clear, step-by-step problem-solving workflow (Step 1, Step 2, etc.) to gain a full understanding of the task before implementation.
- Script Verification:
  Prompted the GPT with the Problem2 statement and the draft script written by me to verify arithmetic, algorithm steps, and output correctness. Verified kepler_convergence.py.

---

## Problem 3: Energy and angular-momentum drift over 10000 orbits

- Workflow Understanding:
  Pasted the Problem3 statement in HW2.pdf and asked GPT to outline a clear, step-by-step problem-solving workflow (Step 1, Step 2, etc.) to gain a full understanding of the task before implementation.
- PROBLEM3_SPEC.md Generation:
  Provided the problem prompt to GPT, generated a draft specification, and double-checked and modified the draft against the requirements in Problem 3.
- Code Generation:
  1. Asked Antigravity to write the problem3.py that solves Problem3 by following the PROBLEM3_SPEC.md, HW2.pdf (read Problem3 and Appendix A), orbit.py, and integrators.py.

- Prompt used in Antigravity:

  Work only within the current directory. Do not explore any unrelated folders or files. Only read the files explicitly required for this task.
Please implement Problem 3 according to PROBLEM3_SPEC.md.
Before making any changes, read and understand:
1.	PROBLEM3_SPEC.md
2.	SPEC.md
3.	orbit.py
4.	integrators.py
5.	hw2_orbit_integrators.pdf
6.	The existing tests, especially the orbit integrator tests
Create a new file named problem3.py that satisfies every requirement and acceptance criterion in PROBLEM3_SPEC.md.
The simulation must accumulate per-orbit statistics without storing the full trajectory of all integration steps for the long simulations. Store the Euler trajectory only if needed for its required plot.
Generate all required figures at the exact paths specified in the specification.
1.	Do not modify orbit.py, integrators.py, existing tests, or other unrelated files unless a change is genuinely necessary.
2.	Do not modify kepler_reference.py.
3.	Do not modify Git history, commit, or push anything.
4.	Before editing, inspect the current working directory and report which files you plan to create or modify.
Run the relevant existing tests first, then run problem3.py.
Check every acceptance criterion in PROBLEM3_SPEC.md against the actual numerical results. Do not merely assume that the program is correct because it runs.
If any criterion fails:
•	Investigate the underlying cause.
•	Check the numerical method, error definitions, orbit indexing, and plotting requirements.
•	Fix the implementation if appropriate, then rerun the relevant checks.
•	Do not weaken or change any acceptance threshold to make the program pass.
When finished, summarize:
1.	Which files were created or modified.
2.	Which tests and commands were run, and whether they passed.
3.	The measured values for every acceptance criterion and whether each passed.
4.	The paths of the generated figures.
5.	Any remaining failures, uncertainties, or limitations.
Do not claim that a test passed unless you actually ran it and observed the result.

---

## Problem 4: Few-body application

- Workflow Understanding:
  Pasted the Problem4 statement in HW2.pdf and asked GPT to outline a clear, step-by-step problem-solving workflow (Step 1, Step 2, etc.) to gain a full understanding of the task before implementation.
- PROBLEM4_SPEC.md Generation:
  Provided the problem prompt to GPT, generated a draft specification, and double-checked and modified the draft against the requirements in Problem 4.
- Code Generation:
  1. Asked Antigravity to write the 3_body_orbit.py that solves Problem 4 Option A by following the PROBLEM4_SPEC.md, HW2.pdf (read Problem1, Appendix A, and Appendix B), orbit.py, and integrators.py.
  2. Asked Antigravity to explain the implementation, step by step, how the existing code is reused, how the initial conditions are set, how the simulation works, and how the rotating-frame transformation is performed.

- Prompt used in Antigravity:

Work only within the current directory. Do not explore any unrelated folders or files. Only read the files explicitly required for this task. And do not modify any existing files.
Please implement Problem 4 according to PROBLEM4_SPEC.md.
Before coding:
7.	Read PROBLEM4_SPEC.md, orbit.py, integrators.py, and hw2_orbit_integrators.pdf, especially Problem 4 Option A and Appendix B.
8.	Inspect the existing code to understand how the orbit() function, leapfrog integrator, and acceleration calculations are implemented.
9.	Reuse the existing code wherever applicable. Do not rewrite the leapfrog integrator.
Implementation requirements:
5.	Follow the specification and the original assignment.
6.	Do not modify any existing files.
7.	Create a new Python file for Problem 4, following the project's naming conventions.
8.	Save the two required figures at the paths specified in the specification.
9.	Run the implementation and check the required diagnostics and acceptance criteria. Do not claim a criterion passes unless the results support it.
10.	If you encounter a conflict between the specification and the original assignment, follow the original assignment and explain the conflict.
After implementation:
•	Summarize which files you created and where they are saved.
•	Report the numerical results and whether each acceptance criterion passes.
•	Explain the implementation to me in Traditional Chinese, step by step. In particular, explain how the existing code is reused, how the initial conditions are set, how the simulation works, how the rotating-frame transformation is performed, and how each diagnostic and plot is produced.
•	Explain the important functions and code sections, including the relevant equations and how they correspond to Problem 4.
•	Identify any assumptions, limitations, or criteria that could not be verified.
Keep the implementation focused on the assignment. Do not add unnecessary features, extra plots, or unrelated files.

---

## Documentation & Report Refinement (README.md & report.pdf & AI_LOG.md)

- English Checking:
  Used GPT to check grammar, improve readability, and refine phrasing in the content without changing the underlying mathematical and technical facts.

---

## One case where the AI was wrong

In Problem 3, Antigravity initially reported incorrect values for the RK4 end-of-orbit relative energy error at orbit 1000 in the discussion section of problem3.py. It reported 1.428e-4 for 100 steps per orbit and 1.388e-7 for 400 steps per orbit.
I detected the error by rerunning problem3.py and checking the numerical output against the values reported in the discussion. The measured errors at orbit 1000 were 2.982488e-2 and 2.898827e-5, respectively. I corrected the discussion to match the simulation results.
This showed that a successful program run alone was not sufficient to validate the reported numerical results; the numerical output and the written discussion also had to be checked against each other.

