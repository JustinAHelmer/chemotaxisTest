# chemotaxis_sim/environment.py

from config import ROCK_GRADIENT, T_period, T_rest, T_travel, x_max, L_max, sigma
import numpy as np


def mu(t):
    """Returns the center of the ligand gradient at time t."""
    if not ROCK_GRADIENT:
        return x_max

    phase = t % T_period
    if phase < T_rest:
        return x_max
    elif phase < T_rest + T_travel:
        alpha = (phase - T_rest) / T_travel
        return x_max - 2 * x_max * alpha
    elif phase < 2 * T_rest + T_travel:
        return -x_max
    else:
        alpha = (phase - (2 * T_rest + T_travel)) / T_travel
        return -x_max + 2 * x_max * alpha


def ligand(x, t):
    """Gaussian ligand field centered at mu(t)."""
    return L_max * np.exp(-((x - mu(t)) ** 2) / (2 * sigma ** 2))
