"""Hyperparameter sensitivity: depth L x width H for the mixed-size GNN.

Trains the mixed-size GNN-SAC (6-10 bus padding protocol, same as
run_gnn_mixed) for each (L, H) in {2,3,4} x {32,64,128} and evaluates
zero-shot on held-out 10-bus feeders (cost, voltage violations, box
violations) plus training time. Results land in results/gnn_sens.json.

Usage: python run_gnn_sens.py --eps 500 --days 20 [--quick]
"""
import argparse
import json
import os
import time
import numpy as np

from grid_env import make_feeder, FeederEnv
from gnn_policy import normalize_adj, GraphSAC
import run_gnn as RG
import run_gnn_mixed as RM

RES = RG.RES
NF = RM.FeederConfig.NF if hasattr(RM, 'FeederConfig') else 9
from grid_env import FeederConfig
NF = FeederConfig.NF
N_MAX = RM.N_MAX
TRAIN_SIZES = RM.TRAIN_SIZES
pad = RM.pad


def build_pool():
    return [make_feeder(s, 300 + s * 10 + i) for s in TRAIN_SIZES for i in range(3)]


def eval_10(agent, days):
    topos = [make_feeder(10, 900 + 100 + i) for i in range(4)]
    costs, vviol, box = [], [], 0
    tot_steps = 0
    for topo in topos:
        mask = topo["controllable"].astype(np.float32)
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=90000 + d)
            X = env.reset(seed=90000 + d)
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            Xp, Ap, mp = pad(X, Ahat, mask)
            done, tot = False, 0.0
            while not done:
                a = agent.act(Xp, Ap, mp, deterministic=True)
                X2, r, done, info = env.step(a[:env.n])
                tot += info["cost"]
                Xp, Ap, mp = pad(X2, Ahat, mask)
            costs.append(tot); vviol.append(env.vviol); box += env.box_viol
            tot_steps += env.cfg.T
    return (float(np.mean(costs)), float(np.mean(vviol)),
            100.0 * box / max(1, tot_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=500)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.eps, args.days = 30, 2

    pool = build_pool()
    out = {}
    for L in (2, 3, 4):
        for H in (32, 64, 128):
            key = f"L{L}_H{H}"
            print(f"[train] {key} ...", flush=True)
            t0 = time.time()
            agent = GraphSAC(NF, N_MAX, hidden=H, layers=L, seed=0)
            RM.train_mixed_pool = None  # noop guard
            # reuse the mixed training loop with custom hyperparams
            rng = np.random.default_rng(0)
            warmup, steps = 400, 0
            for ep in range(args.eps):
                topo = pool[int(rng.integers(len(pool)))]
                env = FeederEnv(topo, safety=True, seed=3000 + ep)
                X = env.reset(seed=3000 + ep)
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
            train_s = time.time() - t0
            cost, vv, boxpct = eval_10(agent, args.days)
            out[key] = {"cost": cost, "vvolt": vv, "box_pct": boxpct,
                        "train_s": train_s,
                        "params_k": sum(p.numel() for p in
                                        agent.actor.parameters()) / 1000.0}
            print(f"  {key}: cost {cost:.1f} vv {vv:.2f} box% {boxpct:.2f} "
                  f"train {train_s:.0f}s params {out[key]['params_k']:.0f}k",
                  flush=True)

    with open(os.path.join(RES, "gnn_sens.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] saved gnn_sens.json")


if __name__ == "__main__":
    main()
