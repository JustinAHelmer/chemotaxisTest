#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import jax.numpy as jnp

from chemotaxis_sim.grn_network_realistic import ptrR_idx
from grn_network_realistic import VectorizedGRN, node_index, dt

# ── PARAMETERS ───────────────────────────────────────────────────────────────
t_spinup    = 10.0       # minutes
t_ligand    = 10     # minutes
lig_conc    = 1000.0     # µM
M           = 1

N = len(node_index)

spinup_steps  = int(t_spinup  / dt)
ligand_steps = int(t_ligand  / dt)

# ── LOAD & CLEAN EXPERIMENTAL DATA ───────────────────────────────────────────
exp_df = (
    pd.read_csv('experimental_data.csv', usecols=['Node','log2FC'], dtype=str)
      .assign(log2FC=lambda df: pd.to_numeric(df.log2FC.str.strip(), errors='coerce'))
      .dropna(subset=['log2FC'])
)
exp_map = dict(zip(exp_df['Node'], exp_df['log2FC']))

# ── LOAD TRAINED PARAMETERS ──────────────────────────────────────────────────
npz = np.load("full_network_trained.npz")
W_act, W_rep = jnp.array(npz["W_act"]), jnp.array(npz["W_rep"])
S_vec, n_vec = jnp.array(npz["S_vec"]), jnp.array(npz["n_vec"])
rp_mat, k_vec= jnp.array(npz["rp_mat"]), jnp.array(npz["k_vec"])

static_seed = 1
rng_init    = np.random.default_rng(static_seed)
init_x   = rng_init.uniform(50.0, 60.0, size=(N,)).astype(np.float32)

# pack into the params tuple, including the initial state
params = (
    W_act, W_rep,
    S_vec, n_vec,
    rp_mat, k_vec,
    jnp.array(init_x)[None, :]  # shape (1, N)
)

# index of the CheY node:
cheY_idx = node_index['cheY']
ptrR_idx = node_index['ptrR']
acrR_idx = node_index['acrR']
fnr_idx = node_index['fnr']
ompR_idx = node_index['ompR']

# instantiate with fixed params
model = VectorizedGRN(M, seed=1, static_seed=1, params=params)

# traces
cheY_trace, ptrR_trace, acrR_trace, fnr_trace, ompR_trace = [], [], [], [], []

# initial values
cheY_trace.append(float(model.x[0, cheY_idx]))
ptrR_trace.append(float(model.x[0, ptrR_idx]))
acrR_trace.append(float(model.x[0, acrR_idx]))
fnr_trace .append(float(model.x[0, fnr_idx]))
ompR_trace.append(float(model.x[0, ompR_idx]))

# ── SPIN-UP at 0 µM ──
for _ in range(spinup_steps):
    model.step_all(jnp.array([0.0], jnp.float32))
    cheY_trace.append(float(model.x[0, cheY_idx]))
    ptrR_trace.append(float(model.x[0, ptrR_idx]))
    acrR_trace.append(float(model.x[0, acrR_idx]))
    fnr_trace.append(float(model.x[0, fnr_idx]))
    ompR_trace.append(float(model.x[0, ompR_idx]))
resting = model.x[0].copy()

# ── PERTURB at 1000 µM ──────────────────────────────────────────────────────
for _ in range(ligand_steps):
    model.step_all(jnp.array([lig_conc], jnp.float32))
    cheY_trace.append(float(model.x[0, cheY_idx]))
    ptrR_trace.append(float(model.x[0, ptrR_idx]))
    acrR_trace.append(float(model.x[0, acrR_idx]))
    fnr_trace.append(float(model.x[0, fnr_idx]))
    ompR_trace.append(float(model.x[0, ompR_idx]))
final = model.x[0].copy()

print(final[ptrR_idx])

# ── BUILD SCATTER DATA ──────────────────────────────────────────────────────
xs, ys, labels = [], [], []
for gene, exp_val in exp_map.items():
    idx = node_index.get(gene)
    if idx is None:
        continue
    r, f = resting[idx], final[idx]
    if r > 0 and f > 0:
        xs.append(exp_val)
        ys.append(np.log2(f / r))
        labels.append(gene)

xs = np.array(xs)
ys = np.array(ys)

# ── PLOT ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(7,6))
plt.scatter(xs, ys, edgecolor='k', alpha=0.7)

# annotate each point
for x, y, g in zip(xs, ys, labels):
    plt.annotate(
        g,
        (x, y),
        textcoords="offset points",
        xytext=(3, 3),
        fontsize=8,
        alpha=0.7
    )

# regression line
slope, intercept, *_ = linregress(xs, ys)
xlim = plt.gca().get_xlim()
xfit = np.linspace(*xlim, 100)
plt.plot(xfit, slope*xfit + intercept, color='C3', lw=1.5,
         label=f'LSRL: slope={slope:.2f}')

# identity line
ylim = plt.gca().get_ylim()
m, Mv = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
plt.plot([m, Mv], [m, Mv], '--', color='k', label='Identity')

plt.xlabel('Experimental log₂ FC')
plt.ylabel('Simulated log₂ FC')
plt.title('Simulated vs Experimental log₂ Fold-Change')
plt.legend()
plt.tight_layout()
plt.show()

times = np.arange(len(cheY_trace)) * dt
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 12))
# CheY
axs[0].plot(times, cheY_trace)
axs[0].set_ylabel('CheY (µM)')
axs[0].axvline(t_spinup, color='C3', ls='--')
# PtrR
axs[1].plot(times, ptrR_trace)
axs[1].set_ylabel('PtrR (µM)')
axs[1].axvline(t_spinup, color='C3', ls='--')
# AcrR
axs[2].plot(times, acrR_trace)
axs[2].set_ylabel('AcrR (µM)')
axs[2].axvline(t_spinup, color='C3', ls='--')
# Fnr
axs[3].plot(times, fnr_trace)
axs[3].set_ylabel('Fnr (µM)')
axs[3].set_xlabel('Time (min)')
axs[3].axvline(t_spinup, color='C3', ls='--')

plt.tight_layout()
plt.show()









