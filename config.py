# Simulation Parameters
dt = 0.01
T = 5000.0
M = 50
rock_interval = 350
ROCK_GRADIENT = True
USE_GRN_REGULATION = False
SHOW_STAT_PLOT = False

# Ligand Parameters
L_max = 0.08
sigma = 2500
x_max = 4000
T_rest = rock_interval
T_travel = 250
T_period = 2 * (T_rest + T_travel)

# Dufour Parameters
epsilon0 = 6
epsilon1 = -1
N = 6
Koff = 0.0182
Kon = 3
epsilon2 = 80
epsilon3 = 80
K = 2.0
omega = 1.3
v = 20
L0 = 0.02
grad = 0.002
tau = 30
Dr = 0.062

# Vladimirov Parameters
k_Y = 100.0
k_Z = 10.0
gamma_Y = 0.1
CheZ = 5.0
CheY_tot = 6.0

# Output
csv_file = "ligand_exposure_results.csv"

# Derived constants for initialization
import numpy as np
mu0 = x_max  # starting gradient center
L_local = L_max * np.exp(-((0.0 - mu0) ** 2) / (2 * sigma ** 2))
F_star = np.log((k_Y * (CheY_tot - K) / (K * (k_Z * CheZ + gamma_Y))) - 1.0)
m_star = (epsilon0 + N * np.log((1 + L_local / Koff) / (1 + L_local / Kon)) - F_star) / (-epsilon1)
