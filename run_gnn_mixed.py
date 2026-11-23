"""Supplementary experiment: mixed-size GNN training.

Trains the GNN policy on a MIX of feeder sizes {6,7,8,9,10} (a fully-connected
policy cannot do this at all), using fixed-tensor batching via padding to a
maximum of 10 buses, then evaluates zero-shot on sizes {8,10,12,14,16}. The
hypothesis: training across sizes further improves extrapolation to larger
unseen feeders versus training on a single size.

Usage:  python run_gnn_mixed.py [--episodes 700] [--eval-days 40]
"""
import argparse
import json
import os
import numpy as np

from grid_env import make_feeder, FeederEnv, FeederConfig
from gnn_policy import normalize_adj, GraphSAC
import run_gnn as RG

RES = RG.RES
N_MAX = 10
TRAIN_SIZES = [6, 7, 8, 9, 10]
EVAL_SIZES = [8, 10, 12, 14, 16]
FEEDERS_PER_SIZE = 4


def pad(X, Ahat, mask, nmax=N_MAX):
    n = X.shape[0]
    Xp = np.zeros((nmax, X.shape[1]), np.float32)
    Xp[:n] = X
    Ap = np.zeros((nmax, nmax), np.float32)
    Ap[:n, :n] = Ahat
    for j in range(n, nmax):
        Ap[j, j] = 1.0                       # isolated normalised self-loop
    mp = np.zeros(nmax, np.float32)
    mp[:n] = mask
    return Xp, Ap, mp


def train_mixed(train_pool, episodes, seed=0):
    agent = GraphSAC(FeederConfig.NF, N_MAX, layers=3, seed=seed)
    rng = np.random.default_rng(seed)
    warmup, steps, curve = 400, 0, []
    for ep in range(episodes):
        topo = train_pool[int(rng.integers(len(train_pool)))]
        env = FeederEnv(topo, safety=True, seed=3000 + ep)
        X = env.reset(seed=3000 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        Xp, Ap, mp = pad(X, Ahat, mask)
        done, ep_r = False, 0.0
        while not done:
            if steps < warmup:
                a_full = rng.uniform(-1, 1, (N_MAX, 2)).astype(np.float32)
            else:
                a_full = agent.act(Xp, Ap, mp)
            a = a_full[:env.n]
            X2, r, done, info = env.step(a)
            Xp2, Ap2, mp2 = pad(X2, Ahat, mask)
            agent.buf.add(Xp, Ap, a_full, r, Xp2, Ap2, float(done), mp)
            Xp, ep_r = Xp2, ep_r + r
            steps += 1
            if steps >= warmup:
                agent.update()
        curve.append(ep_r)
    return agent, np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=700)
    ap.add_argument("--eval-days", type=int, default=40)
    args = ap.parse_args()

    train_pool = []
    for s in TRAIN_SIZES:
        for i in range(3):
            train_pool.append(make_feeder(s, 300 + s * 10 + i))

    print("[train] mixed-size GNN-SAC ...", flush=True)
    agent, curve = train_mixed(train_pool, args.episodes)

    sweep = []
    for size in EVAL_SIZES:
        topos = [make_feeder(size, 900 + size * 10 + i) for i in range(FEEDERS_PER_SIZE)]
        gc, gv = RG.eval_gnn(agent, topos, args.eval_days)
        rc, rv = RG.eval_rule(topos, args.eval_days)
        sweep.append({"size": size, "gnn_cost": gc, "gnn_vviol": gv,
                      "rule_cost": rc, "rule_vviol": rv})
        print(f"  [size {size:2d}] mixed-gnn {gc:7.1f} (vviol {gv:5.2f}) | "
              f"rule {rc:7.1f} (vviol {rv:5.2f})", flush=True)

    with open(os.path.join(RES, "gnn_mixed_results.json"), "w") as f:
        json.dump({"train_sizes": TRAIN_SIZES, "size_sweep": sweep}, f, indent=2)
    np.savez(os.path.join(RES, "gnn_mixed_curve.npz"), curve=curve)
    print("[done] saved to", RES)


if __name__ == "__main__":
    main()
