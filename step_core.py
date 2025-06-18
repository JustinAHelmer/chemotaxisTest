# chemotaxis_sim/step_core.py

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def step_all_cells(x, yp, m, state, theta, CheY_tot, dt, t,
                   epsilon0, epsilon1, N, Koff, Kon, F_star, tau,
                   epsilon2, epsilon3, K, omega, Dr, v,
                   k_Y, k_Z, gamma_Y, CheZ, mu, sigma, L_max):

    M = x.shape[0]
    L_arr = np.zeros(M)

    for i in prange(M):
        # Ligand concentration
        L = L_max * np.exp(-((x[i] - mu) ** 2) / (2 * sigma ** 2))

        # Free energy and CheY-P
        F = epsilon0 + epsilon1 * m[i] + N * np.log((1 + L / Koff) / (1 + L / Kon))
        A = 1.0 / (1.0 + np.exp(F))
        Y = (k_Y * A * CheY_tot[i]) / ((k_Y * A) + (k_Z * CheZ) + gamma_Y)

        # Adaptation
        term = N * np.log((1 + L / Koff) / (1 + L / Kon))
        m_inf = (epsilon0 + term - F_star) / (-epsilon1)
        m[i] += (-(m[i] - m_inf) / tau) * dt

        # Switching probabilities
        gY = (epsilon2 / 4) - (epsilon3 / 2) / (1 + K / Y)
        p_RT = omega * np.exp(-gY) * dt
        p_TR = omega * np.exp(gY) * dt

        rand = np.random.rand()
        if state[i] == 0:
            if rand < p_RT:
                state[i] = 1
            else:
                theta[i] += np.sqrt(2 * Dr * dt) * np.random.randn()
                x[i] += v * np.cos(theta[i]) * dt
                yp[i] += v * np.sin(theta[i]) * dt
        else:
            if rand < p_TR:
                state[i] = 0
                theta[i] = np.random.uniform(0, 2 * np.pi)
                x[i] += v * np.cos(theta[i]) * dt
                yp[i] += v * np.sin(theta[i]) * dt

        L_arr[i] = L

    return x, yp, m, state, theta, L_arr
