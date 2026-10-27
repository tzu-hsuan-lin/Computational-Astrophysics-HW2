"""Fixed-step second-order ODE driver (HW2 Problem 1).

Interface contract
------------------
orbit(acc, x0, v0, t, method="leapfrog") integrates x'' = acc(x).
* acc(x) returns an array with the same shape as x and does not modify x.
* x0, v0 are initial position and velocity at t[0].
* t is a 1-D array of output times; t[0] is the initial time; the n-th step
  is t[n+1] - t[n].
* Returns (x, v), with both arrays having shape (len(t),) + shape(x0).
  The first row contains the initial position and velocity.
* method is one of {"euler", "rk2", "rk4", "leapfrog"}; any other name raises KeyError.
* No hidden state, no printing, no plotting.
"""

import numpy as np

from integrators import integrate


def kdk_step(acc, x, v, a, h):
    """One kick-drift-kick leapfrog step; a = acc(x) is carried between steps.

    Returns (x_new, v_new, a_new). Exactly one call of acc per step.
    """
    vh = v + 0.5 * h * a          # kick, half step
    x_new = x + h * vh            # drift, full step
    a_new = acc(x_new)            # the one force evaluation
    v_new = vh + 0.5 * h * a_new  # kick, half step
    return x_new, v_new, a_new


def leapfrog(acc, x0, v0, t):
    """Kick-drift-kick leapfrog (velocity Verlet) for x'' = acc(x).

    Second order, symplectic, time-reversible, one acc() call per step.
    Returns (x, v), each of shape (len(t),) + shape(x0).
    """
    x = np.empty((len(t),) + np.shape(x0), dtype=float)
    v = np.empty_like(x)
    x[0], v[0] = x0, v0
    a = acc(x[0])
    for n in range(len(t) - 1):
        x[n + 1], v[n + 1], a = kdk_step(acc, x[n], v[n], a, t[n + 1] - t[n])
    return x, v


def _rk_orbit(acc, x0, v0, t, method):
    """Wrap first-order integrate() for second-order Runge-Kutta integration."""
    x0_arr = np.asarray(x0, dtype=float)
    v0_arr = np.asarray(v0, dtype=float)
    orig_shape = np.shape(x0)
    d = x0_arr.size

    y0 = np.concatenate([x0_arr.ravel(), v0_arr.ravel()])

    def f(t_val, y):
        x_val = y[:d].reshape(orig_shape)
        v_val = y[d:].reshape(orig_shape)
        a_val = acc(x_val)
        return np.concatenate([v_val.ravel(), np.asarray(a_val, dtype=float).ravel()])

    y = integrate(f, y0, t, method=method)
    x = y[:, :d].reshape((len(t),) + orig_shape)
    v = y[:, d:].reshape((len(t),) + orig_shape)
    return x, v


def orbit(acc, x0, v0, t, method="leapfrog"):
    """Integrate second-order system x'' = acc(x) on the time grid t.

    Parameters
    ----------
    acc : callable(x) -> ndarray, same shape as x
    x0 : array_like, initial position at t[0]
    v0 : array_like, initial velocity at t[0]
    t : 1-D array of output times
    method : {"euler", "rk2", "rk4", "leapfrog"}, default "leapfrog"

    Returns
    -------
    x, v : ndarray, each of shape (len(t),) + shape(x0)
    """
    if method == "leapfrog":
        return leapfrog(acc, x0, v0, t)
    elif method in ("euler", "rk2", "rk4"):
        return _rk_orbit(acc, x0, v0, t, method)
    else:
        raise KeyError(f"Unknown method '{method}'")
