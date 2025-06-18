# chemotaxis_sim/cell.py

import numpy as np
from time import perf_counter
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

        # History arrays: store snapshots for each step (faster than per‑cell appends)
        self.x_hist = [self.x.copy()]
        self.yp_hist = [self.yp.copy()]

    def step_all(self, dt, mu_val, step_idx):
        # --- timing checkpoint 0 -------------------------------------------
        #t0 = perf_counter()

        t_sim = step_idx * dt
        self.x, self.yp, self.m, self.state, self.theta, L_arr = step_all_cells(
            self.x, self.yp, self.m, self.state, self.theta, self.CheY_tot,
            dt, t_sim,
            epsilon0, epsilon1, N, Koff, Kon, F_star, tau,
            epsilon2, epsilon3, K, omega, Dr, v,
            k_Y, k_Z, gamma_Y, CheZ, mu_val, sigma, L_max
        )

        # --- timing checkpoint 1 -------------------------------------------
        #t1 = perf_counter()

        # Update exposure (vectorised)
        self.ligand_exposure += L_arr * dt

        # --- timing checkpoint 2 -------------------------------------------
        #t2 = perf_counter()

        # Update history (single vector copy instead of Python loop)
        self.x_hist.append(self.x.copy())
        self.yp_hist.append(self.yp.copy())

        # --- timing checkpoint 3 -------------------------------------------
        #t3 = perf_counter()

        # Simple printout of durations
        #print(
        #    f"[timing] step {step_idx} | "
        #    f"core: {t1 - t0:.6f}s | "
        #    f"exposure: {t2 - t1:.6f}s | "
        #    f"history: {t3 - t2:.6f}s | "
        #    f"total: {t3 - t0:.6f}s"
        #)

    def exposures(self):
        return self.ligand_exposure
