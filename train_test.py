#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from tqdm import trange
from jax import lax, random

from grn_network_realistic import (
    internal, node_index, build_params,
    grn_step_all_jax, dt
)

# ── 1) Load ALL experimental log₂‐FCs ────────────────────────────────────────
exp_df = (
    pd.read_csv("experimental_data.csv", usecols=["Node","log2FC"])
      .assign(log2FC=lambda df: pd.to_numeric(df.log2FC, errors="coerce"))
      .dropna(subset=["log2FC"])
)
genes      = list(exp_df["Node"])
exp_vals   = jnp.array(exp_df["log2FC"].values, dtype=jnp.float32)
exp_idx    = jnp.array([node_index[g] for g in genes], dtype=jnp.int32)
num_meas   = len(exp_vals)
eps        = 1e-8
N          = len(node_index)

# ── 2) Partition edges ────────────────────────────────────────────────────────
pos, neg, unk = [], [], []
for _, r in internal.iterrows():
    i, j = node_index[r["3)RegulatorGeneName"]], node_index[r["5)regulatedName"]]
    if   r["6)function"] == "+": pos.append((j,i))
    elif r["6)function"] == "-": neg.append((j,i))
    else:                        unk.append((j,i))
pos_idx = jnp.array(pos, dtype=jnp.int32).T
neg_idx = jnp.array(neg, dtype=jnp.int32).T
unk_idx = jnp.array(unk, dtype=jnp.int32).T

# ── 3) Fixed baseline params ─────────────────────────────────────────────────
W_act0, W_rep0, S0, n0, rp0, k0 = build_params(seed=0, static_seed=1)
W_act_base = jnp.array(W_act0)
W_rep_base = jnp.array(W_rep0)
n_vec      = jnp.array(n0)
rp_mat     = jnp.array(rp0)

# ── 4) Fixed init_x (nonzero) ────────────────────────────────────────────────
static_seed = 1
rng_init     = np.random.default_rng(static_seed)
init_x_np    = rng_init.uniform(50.0,60.0,size=(N,)).astype(np.float32)
init_x       = jnp.array(init_x_np)[None,:]  # shape (1,N)

# ── 5) Initialize trainable logs for pos/neg/unk + S + k ───────────────────
rng = np.random.default_rng(42)
pa0 = jnp.log(jnp.array([W_act0[j,i]+eps for j,i in pos], dtype=jnp.float32))
nr0 = jnp.log(jnp.array([W_rep0[j,i]+eps for j,i in neg], dtype=jnp.float32))
ua0 = jnp.log(jnp.array(rng.uniform(0.01,0.1,len(unk)), dtype=jnp.float32))
ur0 = jnp.log(jnp.array(rng.uniform(0.01,0.1,len(unk)), dtype=jnp.float32))
tS0 = jnp.log(jnp.array(S0, dtype=jnp.float32))
tk0 = jnp.log(jnp.array(k0, dtype=jnp.float32))
theta0 = (pa0, nr0, ua0, ur0, tS0, tk0)

# penalty weight on log-k drift
λ_k = 1e-3

# ── 6) Rebuild function ──────────────────────────────────────────────────────
@jax.jit
def rebuild(theta):
    pa,nr,ua,ur,tS,tk = theta
    W_act = W_act_base.at[(pos_idx[0],pos_idx[1])].set(jnp.exp(pa))
    W_rep = W_rep_base.at[(neg_idx[0],neg_idx[1])].set(jnp.exp(nr))
    W_act = W_act.at   [(unk_idx[0],unk_idx[1])].set(jnp.exp(ua))
    W_rep = W_rep.at   [(unk_idx[0],unk_idx[1])].set(jnp.exp(ur))
    S_vec = jnp.exp(tS)
    k_vec = jnp.exp(tk)
    return W_act, W_rep, S_vec, n_vec, rp_mat, k_vec

# ── 7) Rollout & gather simulated log₂‐FCs ──────────────────────────────────
def simulate_all(theta):
    W_act,W_rep,S_vec,n_vec_,rp_mat_,k_vec = rebuild(theta)
    steps = int(10.0/dt)

    def step_fn(x, lig):
        x_new,_ = grn_step_all_jax(
            x, W_act, W_rep,
            S_vec, n_vec_, rp_mat_, k_vec,
            jnp.full((1,), lig, jnp.float32)
        )
        return x_new, None

    # 10 min @0
    x_ss,_  = lax.scan(step_fn, init_x, jnp.zeros((steps,),jnp.float32))
    # 10 min @1000
    x_fin,_ = lax.scan(step_fn, x_ss, jnp.ones((steps,),jnp.float32)*1000.0)

    # compute log₂FC for every measured index
    base = x_ss[-1, exp_idx]
    fin  = x_fin[-1, exp_idx]
    return jnp.log2((fin+eps)/(base+eps))

# ── 8) Loss = MSE across all measured nodes ─────────────────────────────────
@jax.jit
def loss_fn(theta):
    sim = simulate_all(theta)
    return jnp.mean((sim - exp_vals)**2)


loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# ── 9) Optimizer setup ───────────────────────────────────────────────────────
opt    = optax.adam(5e-2)
opt_st = opt.init(theta0)

def train_step(theta, opt_st):
    loss, grads = loss_and_grad(theta)
    updates, opt_st = opt.update(grads, opt_st, theta)
    return optax.apply_updates(theta, updates), opt_st, loss

# ──10) Warm‐up compile & loop ───────────────────────────────────────────────
print("Compiling…")
_ = loss_and_grad(theta0)
print("Training…")
theta, state = theta0, opt_st
for i in trange(1,1001):
    theta, state, L = train_step(theta, state)
    if i%100==0:
        print(f"{i:04d} loss={L:.3e}")

# ──11) Save all 7+init_x ───────────────────────────────────────────────────
W_act_tr, W_rep_tr, S_tr, n_tr, rp_tr, k_tr = rebuild(theta)
np.savez("full_network_trained.npz",
         W_act=W_act_tr, W_rep=W_rep_tr,
         S_vec=S_tr, n_vec=n_tr,
         rp_mat=rp_tr, k_vec=k_tr,
         init_x=init_x_np)
print("→ saved full_network_trained.npz")
