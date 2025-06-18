# chemotaxis_sim/main.py

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from config import *
from cell import CellBatch
from environment import mu

# Simulation Setup
if not ROCK_GRADIENT:
    rock_interval = T + 1
steps_per_flip = int(rock_interval / dt)
n_steps = int(T / dt)

# Run Simulation
cells = CellBatch(M)

for step in range(n_steps):
    mu_val = mu(step * dt)
    cells.step_all(dt, mu_val, step)
    if step % 1000 == 0:
        print(f"Step {step} / {n_steps}")
# Log Exposure
avg_exposure = np.mean(cells.exposures())
print(f"\nAverage ligand exposure: {avg_exposure:.4f} mM·s")

result_row = {
    "Mode": f"{'GRN' if USE_GRN_REGULATION else 'Normal'}",
    "Rocking": ROCK_GRADIENT,
    "Cells": M,
    "Time": T,
    "dt": dt,
    "AvgExposure": round(avg_exposure, 6)
}

csv_file = "ligand_exposure_results.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=result_row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(result_row)

print(f"Result logged to {csv_file}")

# Plot Trajectories with ligand background
plt.figure(figsize=(6, 6))

if not ROCK_GRADIENT:
    grid_size = 200
    x_grid = np.linspace(-4000, 4000, grid_size)
    y_grid = np.linspace(-4000, 4000, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    T_final = T

    def compute_ligand(x, t):
        center = mu(t)
        return L_max * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    L_grid = np.array([[compute_ligand(x, T_final) for x in x_grid] for _ in y_grid])

    plt.imshow(L_grid, extent=[-4000, 4000, -4000, 4000], origin='lower',
               cmap='viridis', alpha=0.6, aspect='auto')

for i in range(M):
    plt.plot(cells.x_hist[i], cells.yp_hist[i], lw=0.8, alpha=0.9)

plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.title("Cell trajectories")
plt.xlim(-4000, 4000)
plt.ylim(-4000, 4000)
plt.tight_layout()
plt.show()
