# grn_network_realistic.py

import json
import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from jax import vmap, lax, random

# ── USER-SETTABLE PARAMETERS ──────────────────────
dt = 0.01                               # minutes per step (float)
seed = 1                                # for build_params

# ── 1) LOAD NODE LIST & EDGES ────────────────────
with open('layers.txt') as f:
    reservoir = json.load(f)['level_0']
focus = ['ptrR', 'cheY']

edges = pd.read_csv('tableData.csv', dtype=str)

mask = (
    (edges['3)RegulatorGeneName'].isin(reservoir) &
     edges['5)regulatedName'].isin(reservoir))
  | (edges['3)RegulatorGeneName'].isin(focus) &
     edges['5)regulatedName'].isin(reservoir))
  | (edges['3)RegulatorGeneName'].isin(reservoir) &
     edges['5)regulatedName'].isin(focus))
)
internal = edges[mask & (edges['3)RegulatorGeneName'] != edges['5)regulatedName'])]

# ── 2) INDEX NODES ────────────────────────────────
node_list  = list(reservoir + focus)
node_index = {n:i for i,n in enumerate(node_list)}
N          = len(node_list)
ptrR_idx   = node_index['ptrR']
cheY_idx   = node_index['cheY']
acrR_idx = node_index['acrR']

# ── 3) BUILD PARAMETERS (NumPy RNG OK) ───────────
def build_params(seed, static_seed):
    rng_dyn  = onp.random.default_rng(seed)
    rng_stat = onp.random.default_rng(static_seed)

    W_act = onp.zeros((N, N), onp.float32)
    W_rep = onp.zeros((N, N), onp.float32)
    for _, row in internal.iterrows():
        i = node_index[row['3)RegulatorGeneName']]
        j = node_index[row['5)regulatedName']]
        func = row['6)function'] if row['6)function'] in ['+','-'] else rng_stat.choice(['+','-'])
        v = max(0.0, rng_stat.normal(5.0, 1.0))
        if func=='+': W_act[j,i] = v
        else:          W_rep[j,i] = v

    S_vec = onp.maximum(0, rng_stat.normal(0.505, 0.99/6.0, size=N)).astype(onp.float32)
    n_vec = onp.maximum(0, rng_stat.normal(2.5, 3.0/6.0, size=N)).astype(onp.float32)
    rp_mat= onp.maximum(0, rng_stat.normal(0.055,0.09/6.0,size=(N,N))).astype(onp.float32)
    k_vec = onp.maximum(0, rng_stat.normal(0.04, 0.06/6.0, size=N)).astype(onp.float32)

    return W_act, W_rep, S_vec, n_vec, rp_mat, k_vec

# ── 4) JAX‐VERSION OF ptrR_reset ───────────────────
@jax.jit
def ptrR_update_jax(ligand: float) -> float:
    # basal copy‐number instead of µM:
    x0    = 55.0
    # same fold-change target:
    x_tgt = x0 * (2.0 ** -0.738)
    # ligand still 0→1000 scale
    t     = jnp.clip(ligand / 1000.0, 0.0, 1.0)
    return x0 + t * (x_tgt - x0)


# ── 5) PURE‐JAX STEP FUNCTION ──────────────────────
@jax.jit
def grn_step_all_jax(x, W_act, W_rep,
                     S_vec, n_vec, rp_mat, k_vec,
                     ligands):
    """
    x:        [M, N] state matrix
    ligands: [M] ligand conc. array
    returns: new x [M,N], cheY readout [M]
    """
    def cell_step(x_row, ligand):
        # clip negative
        x0 = jnp.clip(x_row, 0.0)
        # Hill activation/repression
        xp = x0 ** n_vec
        Sp = S_vec ** n_vec
        act = (xp / (Sp + xp)) @ W_act.T
        rep = jnp.sum(W_rep * (Sp[None] / (Sp[None] + rp_mat * xp[None])), axis=1)
        rep = jnp.where(rep > 0.0, rep, 1.0)
        # ODE Euler step
        F  = -k_vec * x0 + act * rep + 0.005
        x1 = jnp.clip(x0 + dt * F, 0.0)
        # ptrR reset
        x1 = x1.at[ptrR_idx].set(ptrR_update_jax(ligand))
        return x1

    x_new    = vmap(cell_step)(x, ligands)
    cheY_out = x_new[:, cheY_idx]
    return x_new, cheY_out

# ── 6) VECTORGED CLASS (for existing APIs) ─────────
class VectorizedGRN:
    def __init__(self, M, seed=None, static_seed=None, params=None):
        self.M = M
        if params is None:
            W_act, W_rep, S_vec, n_vec, rp_mat, k_vec = build_params(seed, static_seed)
            self.W_act, self.W_rep = jnp.array(W_act), jnp.array(W_rep)
            self.S_vec, self.n_vec = jnp.array(S_vec), jnp.array(n_vec)
            self.rp_mat, self.k_vec = jnp.array(rp_mat), jnp.array(k_vec)
            key = random.PRNGKey(static_seed or 0)
            self.x = random.uniform(key, (M, N), minval=0.0, maxval=1.0)
        else:
            (self.W_act, self.W_rep,
             self.S_vec, self.n_vec,
             self.rp_mat, self.k_vec,
             self.x) = params

    def step_all(self, ligands):
        # ligands: jnp array of shape (M,)
        self.x, cheY = grn_step_all_jax(
            self.x,
            self.W_act, self.W_rep,
            self.S_vec, self.n_vec,
            self.rp_mat, self.k_vec,
            ligands
        )
        return self.x, cheY
