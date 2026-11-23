"""Advanced supplementary experiments (real numbers):
  * N-1 line contingency (Table 8)
  * projection vs soft-penalty vs unconstrained ablation (Table 9)
  * GNN-DDPG / GNN-PPO / GNN-SAC backbone comparison (Table 11)
  * PCA of latent node embeddings (figure for Section 7.7)

Reuses grid_env (now with constraint_mode), gnn_policy (GraphSAC/DDPG/PPO) and
the training/eval helpers of run_gnn / run_gnn_mixed.
"""
import json
import os
import numpy as np

from grid_env import make_feeder, FeederEnv, FeederConfig
from gnn_policy import normalize_adj, GraphSAC, GraphDDPG, GraphPPO
import run_gnn as RG

RES = RG.RES
NF = FeederConfig.NF
N_MAX = 10
TRAIN_SIZES = [6, 7, 8, 9, 10]


def pad(X, Ahat, mask, nmax=N_MAX):
    n = X.shape[0]
    Xp = np.zeros((nmax, X.shape[1]), np.float32); Xp[:n] = X
    Ap = np.zeros((nmax, nmax), np.float32); Ap[:n, :n] = Ahat
    for j in range(n, nmax):
        Ap[j, j] = 1.0
    mp = np.zeros(nmax, np.float32); mp[:n] = mask
    return Xp, Ap, mp


def contingency(topo, seed):
    """Reroute one line (changes connectivity) to emulate an N-1 event."""
    t = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in topo.items()}
    rng = np.random.default_rng(seed)
    n = t["n_bus"]
    cands = [j for j in range(2, n) if t["parent"][j] != 0]
    if not cands:
        return t
    j = int(rng.choice(cands))
    old = t["parent"][j]
    q = int(rng.choice([c for c in range(j) if c != old]))
    t["parent"] = t["parent"].copy(); t["parent"][j] = q
    t["r"] = t["r"].copy(); t["x"] = t["x"].copy()
    t["r"][j] = rng.uniform(0.015, 0.05); t["x"][j] = rng.uniform(0.015, 0.05)
    t["order"] = np.argsort(-np.arange(n))
    return t


def off_policy_train(agent, pool, episodes, mode, seed=0):
    """Generic replay-based loop for GraphSAC / GraphDDPG (padded to N_MAX)."""
    rng = np.random.default_rng(seed)
    warmup, steps = 400, 0
    for ep in range(episodes):
        topo = pool[int(rng.integers(len(pool)))]
        env = FeederEnv(topo, safety=True, seed=4000 + ep, constraint_mode=mode)
        X = env.reset(seed=4000 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        Xp, Ap, mp = pad(X, Ahat, mask)
        done = False
        while not done:
            a_full = (rng.uniform(-1, 1, (N_MAX, 2)).astype(np.float32)
                      if steps < warmup else agent.act(Xp, Ap, mp))
            X2, r, done, info = env.step(a_full[:env.n])
            Xp2, Ap2, mp2 = pad(X2, Ahat, mask)
            agent.buf.add(Xp, Ap, a_full, r, Xp2, Ap2, float(done), mp)
            Xp = Xp2; steps += 1
            if steps >= warmup:
                agent.update()
    return agent


def ppo_train(agent, pool, episodes, mode, seed=0):
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        topo = pool[int(rng.integers(len(pool)))]
        env = FeederEnv(topo, safety=True, seed=4000 + ep, constraint_mode=mode)
        X = env.reset(seed=4000 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        Xp, Ap, mp = pad(X, Ahat, mask)
        done = False
        while not done:
            a_full = agent.act(Xp, Ap, mp)
            X2, r, done, info = env.step(a_full[:env.n])
            agent.store(Xp, Ap, mp, a_full, r, done)
            Xp, _, _ = pad(X2, Ahat, mask)
        if (ep + 1) % 8 == 0:
            agent.finish()
    agent.finish()
    return agent


def eval_rates(agent, topos, days, mode):
    import numpy as _np
    costs = []
    box_steps = vbus = tot_steps = tot_bus = 0
    for topo in topos:
        mask = topo["controllable"].astype(np.float32)
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=90000 + d, constraint_mode=mode)
            X = env.reset(seed=90000 + d)
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            done, tot = False, 0.0
            while not done:
                a = agent.act(X, Ahat, mask, deterministic=True)
                X, r, done, info = env.step(a)
                tot += info["cost"]
            costs.append(tot)
            box_steps += env.box_viol; vbus += env.vviol
            tot_steps += env.cfg.T; tot_bus += env.cfg.T * env.n
    return (float(_np.mean(costs)), 100.0 * box_steps / tot_steps,
            100.0 * vbus / tot_bus)


def pca_embeddings(agent, sizes):
    import torch
    E, lab = [], []
    for s in sizes:
        topo = make_feeder(s, 2200 + s)
        env = FeederEnv(topo, safety=True, seed=5)
        X = env.reset(seed=5)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        with torch.no_grad():
            H = agent.actor.enc(torch.as_tensor(X[None]),
                                torch.as_tensor(Ahat[None]))[0].numpy()
        for j in range(topo["n_bus"]):
            E.append(H[j])
            if j == 0:
                lab.append(0)
            elif topo["controllable"][j]:
                lab.append(1)
            elif topo["has_pv"][j]:
                lab.append(2)
            else:
                lab.append(3)
    E = np.array(E); lab = np.array(lab)
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    coords = Ec @ Vt[:2].T
    return coords, lab


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep8", type=int, default=500)
    ap.add_argument("--epmix", type=int, default=400)
    ap.add_argument("--days", type=int, default=20)
    args = ap.parse_args()
    out = {}
    days = args.days

    # ---- A. N-1 contingency (8-bus) ----
    print("[A] N-1 contingency ...", flush=True)
    train8 = [make_feeder(8, s) for s in RG.TRAIN_SEEDS]
    test8 = [make_feeder(8, s) for s in RG.TEST_SEEDS]
    gnn, _ = RG.train_gnn(train8, args.ep8)
    mlp, _ = RG.train_mlp(train8, args.ep8)
    cont = [contingency(t, 500 + i) for i, t in enumerate(test8)]
    out["n1"] = {
        "rule": {"normal": RG.eval_rule(test8, days), "cont": RG.eval_rule(cont, days)},
        "mlp": {"normal": RG.eval_mlp(mlp, test8, days), "cont": RG.eval_mlp(mlp, cont, days)},
        "gnn": {"normal": RG.eval_gnn(gnn, test8, days), "cont": RG.eval_gnn(gnn, cont, days)},
    }
    for k, v in out["n1"].items():
        print(f"  {k}: normal {v['normal'][0]:.1f} -> cont {v['cont'][0]:.1f} "
              f"(vv {v['cont'][1]:.2f})", flush=True)

    # ---- B. constraint-handling ablation (mixed 6-10, eval 10-bus) ----
    print("[B] projection vs soft vs none ...", flush=True)
    pool = [make_feeder(s, 300 + s * 10 + i) for s in TRAIN_SIZES for i in range(3)]
    e10 = [make_feeder(10, 900 + 100 + i) for i in range(4)]
    abl = {}
    agents_proj = None
    for mode in ("none", "soft", "projection"):
        ag = GraphSAC(NF, N_MAX, layers=3, seed=0)
        off_policy_train(ag, pool, args.epmix, mode)
        abl[mode] = eval_rates(ag, e10, days, mode)
        print(f"  {mode}: cost {abl[mode][0]:.1f} box% {abl[mode][1]:.2f} "
              f"volt% {abl[mode][2]:.2f}", flush=True)
        if mode == "projection":
            agents_proj = ag
    out["ablation"] = abl

    # ---- C. backbone comparison (mixed 6-10, eval 10-bus) ----
    print("[C] DDPG / PPO / SAC backbones ...", flush=True)
    ddpg = GraphDDPG(NF, N_MAX, layers=3, seed=0)
    off_policy_train(ddpg, pool, args.epmix, "projection")
    ppo = GraphPPO(NF, layers=3, seed=0)
    ppo_train(ppo, pool, args.epmix, "projection")
    bb = {}
    for name, ag in (("ddpg", ddpg), ("ppo", ppo), ("sac", agents_proj)):
        c, _, vr = eval_rates(ag, e10, days, "projection")
        bb[name] = {"cost": c, "volt_pct": vr}
        print(f"  {name}: cost {c:.1f} volt% {vr:.2f}", flush=True)
    out["backbones"] = bb

    # ---- D. PCA of latent embeddings ----
    print("[D] PCA embeddings ...", flush=True)
    coords, lab = pca_embeddings(agents_proj, [8, 12, 16])
    np.savez(os.path.join(RES, "gnn_pca.npz"), coords=coords, lab=lab)

    with open(os.path.join(RES, "gnn_advanced.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] saved gnn_advanced.json + gnn_pca.npz")


if __name__ == "__main__":
    main()
