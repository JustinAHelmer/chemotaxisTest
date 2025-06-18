# chemotaxis_sim/cell.py

import numpy as np
from config import *
from step_core import step_all_cells

class CellBatch:
    def __init__(self, M):
        self.M = M
        self.x = np.zeros(M)
        self.yp = np.zeros(M)
        self.m = np.full(M, m_star)
        self.state = np.zeros(M, dtype=np.int32)
        self.theta = np.random.uniform(0, 2 * np.pi, size=M)
        self.CheY_tot = np.full(M, CheY_tot)

        self.ligand_exposure = np.zeros(M)

        # History arrays (list of M lists)
        self.x_hist = [[] for _ in range(M)]
        self.yp_hist = [[] for _ in range(M)]

        # Initialize histories with initial positions
        for i in range(M):
            self.x_hist[i].append(self.x[i])
            self.yp_hist[i].append(self.yp[i])

    def step_all(self, dt, mu_val, step_idx):
        t_sim = step_idx * dt

        self.x, self.yp, self.m, self.state, self.theta, L_arr = step_all_cells(
            self.x, self.yp, self.m, self.state, self.theta, self.CheY_tot,
            dt, t_sim,
            epsilon0, epsilon1, N, Koff, Kon, F_star, tau,
            epsilon2, epsilon3, K, omega, Dr, v,
            k_Y, k_Z, gamma_Y, CheZ, mu_val, sigma, L_max
        )

        # Update exposure
        self.ligand_exposure += L_arr * dt

        # Update history
        for i in range(self.M):
            self.x_hist[i].append(self.x[i])
            self.yp_hist[i].append(self.yp[i])

    def exposures(self):
        return self.ligand_exposure
