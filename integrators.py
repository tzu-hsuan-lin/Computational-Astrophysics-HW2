"""Fixed-step explicit ODE integrators  (Lecture 4, Appendix B).

Interface contract
------------------
integrate(f, y0, t, method) integrates dy/dt = f(t, y).
* f(t, y) returns an array with the same shape as y and does not modify y.
* t is a 1-D array of output times; t[0] is the initial time; the n-th step
  is t[n+1] - t[n] (any spacing).
* Returns an array of shape (len(t),) + shape(y0); row 0 is y0.
* method is "euler", "rk2" (explicit midpoint) or "rk4" (classical RK4);
  any other name raises KeyError.
* No hidden state, no printing, no plotting.
"""
import numpy as np


def euler_step(f, t, y, h):
    """Forward Euler: follow the slope at the start of the step."""
    return y + h * f(t, y)


def rk2_step(f, t, y, h):
    """Explicit midpoint method: slope at the start, then slope at the midpoint."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    return y + h * k2


def rk4_step(f, t, y, h):
    """Classical fourth-order Runge-Kutta: four slopes, weights 1 2 2 1."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEPPERS = {"euler": euler_step, "rk2": rk2_step, "rk4": rk4_step}


def integrate(f, y0, t, method="rk4"):
    """Integrate dy/dt = f(t, y) on the time grid t.

    Parameters
    ----------
    f : callable(t, y) -> ndarray, same shape as y
    y0 : array_like, initial state at t[0]
    t : 1-D array of output times (defines the steps)
    method : one of {"euler", "rk2", "rk4"}

    Returns
    -------
    y : ndarray, shape (len(t),) + shape(y0); y[0] == y0
    """
    step = _STEPPERS[method]
    y = np.empty((len(t),) + np.shape(y0), dtype=float)
    y[0] = y0
    for n in range(len(t) - 1):
        y[n + 1] = step(f, t[n], y[n], t[n + 1] - t[n])
    return y

