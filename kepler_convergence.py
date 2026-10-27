import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from orbit import orbit


# ==========================================
# 1. Problem setup
# ==========================================

def acc(r):
    """Kepler acceleration, with GM = 1."""
    radius = np.linalg.norm(r)
    return -r / radius**3


x0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])

P = 2 * np.pi

methods = ["euler", "rk2", "rk4", "leapfrog"]

# Number of time steps per orbit
steps_per_orbit = [100, 200, 400, 800, 1600, 3200]

# Step sizes
h_values = P / np.array(steps_per_orbit)

# Simulate one orbit and ten orbits
orbit_counts = [1, 10]

# Create the output directory for figures
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Run simulations and calculate errors
# ==========================================

errors = {}

for n_orbits in orbit_counts:

    errors[n_orbits] = {}

    for method in methods:

        method_errors = []

        for N in steps_per_orbit:

            # Time array
            t = np.linspace(
                0,
                n_orbits * P,
                n_orbits * N + 1
            )

            # Run the orbit simulation
            x, v = orbit(acc, x0, v0, t, method=method)

            # Calculate the final position error
            error = np.linalg.norm(x[-1] - x0)

            method_errors.append(error)

        # Store the six errors for this method
        errors[n_orbits][method] = np.array(method_errors)

        print(
            f"{n_orbits} orbit(s), {method}:",
            errors[n_orbits][method]
        )


# ==========================================
# 3. Least-squares slopes
# ==========================================

slopes = {}

for n_orbits in orbit_counts:

    slopes[n_orbits] = {}

    for method in methods:

        E = errors[n_orbits][method]

        # Fit a straight line in log-log space
        slope, intercept = np.polyfit(
            np.log(h_values),
            np.log(E),
            1
        )

        slopes[n_orbits][method] = slope

        print(
            f"{n_orbits} orbit(s), {method}: "
            f"slope = {slope:.4f}"
        )


# ==========================================
# 4. Richardson estimates
# ==========================================

richardson = {}

for n_orbits in orbit_counts:

    richardson[n_orbits] = {}

    for method in methods:

        E = errors[n_orbits][method]
        estimates = []

        # Compare each pair of adjacent step sizes
        for i in range(len(E) - 1):

            p = np.log2(E[i] / E[i + 1])
            estimates.append(p)

        richardson[n_orbits][method] = estimates

        print(
            f"{n_orbits} orbit(s), {method}: "
            f"Richardson estimates = {[f'{p:.4f}' for p in estimates]}"
        )


# ==========================================
# 5. Plot one-orbit convergence
# ==========================================

fig, ax = plt.subplots(figsize=(8, 6))

for method in methods:

    E = errors[1][method]
    slope = slopes[1][method]

    ax.loglog(
        h_values,
        E,
        marker="o",
        linestyle="-",
        label=f"{method}, slope = {slope:.2f}"
    )

ax.set_xlabel("Step size h")
ax.set_ylabel(r"Final position error $||r(P)-r_0||$")
ax.set_title("Kepler convergence study: one orbit")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(output_dir / "convergence.png", dpi=300)
plt.close(fig)


# ==========================================
# 6. Plot ten-orbit convergence
# ==========================================

fig, ax = plt.subplots(figsize=(8, 6))

for method in methods:

    E = errors[10][method]
    slope = slopes[10][method]

    ax.loglog(
        h_values,
        E,
        marker="o",
        linestyle="-",
        label=f"{method}, slope = {slope:.2f}"
    )

ax.set_xlabel("Step size h")
ax.set_ylabel(r"Final position error $||r(10P)-r_0||$")
ax.set_title("Kepler convergence study: ten orbits")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(output_dir / "convergence_10_orbits.png", dpi=300)
plt.close(fig)

print("Saved figures/convergence.png")
print("Saved figures/convergence_10_orbits.png")
