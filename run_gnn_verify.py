"""Verification runs for the supplementary experiments (real numbers).

Reuses the mixed-size GNN and the feeder environment to actually measure:
  * robustness at sigma=0.2 vs 0.4 (+ optional sensor noise) on 10-bus feeders;
  * zero-shot cost/voltage on large 32- and 64-bus feeders;
  * per-step GNN inference latency across feeder sizes.
N-1 contingency, soft-penalty ablation and DDPG/PPO backbones are not covered
here and remain to be implemented.
"""
import json
import os
import time
import numpy as np

from grid_env import make_feeder, FeederEnv
from gnn_policy import normalize_adj
import run_gnn_mixed as RM
import run_gnn as RG

RES = RG.RES


def eval_cost(agent, topos, days, sigma, obs_noise=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    costs, vviol = [], []
    for topo in topos:
        mask = topo["controllable"].astype(np.float32)
        for d in range(days):
            env = FeederEnv(topo, safety=True, uncertainty=sigma, seed=80000 + d)
            X = env.reset(seed=80000 + d)
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            done, tot = False, 0.0
            while not done:
                Xo = X + obs_noise * rng.standard_normal(X.shape).astype(np.float32) if obs_noise else X
                a = agent.act(Xo, Ahat, mask, deterministic=True)
                X, r, done, info = env.step(a)
                tot += info["cost"]
            costs.append(tot); vviol.append(env.vviol)
    return float(np.mean(costs)), float(np.mean(vviol))


def eval_rule(topos, days, sigma):
    costs, vviol = [], []
    for topo in topos:
        for d in range(days):
            env = FeederEnv(topo, safety=True, uncertainty=sigma, seed=80000 + d)
            env.reset(seed=80000 + d)
            done, tot = False, 0.0
            while not done:
                _, r, done, info = env.step(RG.rule_feeder(env))
                tot += info["cost"]
            costs.append(tot); vviol.append(env.vviol)
    return float(np.mean(costs)), float(np.mean(vviol))


def main():
    print("[train] mixed-size GNN ...", flush=True)
    pool = []
    for s in RM.TRAIN_SIZES:
        for i in range(3):
            pool.append(make_feeder(s, 300 + s * 10 + i))
    agent, _ = RM.train_mixed(pool, episodes=700)

    out = {}
    t10 = [make_feeder(10, 900 + 100 + i) for i in range(4)]
    g02 = eval_cost(agent, t10, 40, 0.2)
    g04 = eval_cost(agent, t10, 40, 0.4)
    g04n = eval_cost(agent, t10, 40, 0.4, obs_noise=0.05)
    r02 = eval_rule(t10, 40, 0.2)
    r04 = eval_rule(t10, 40, 0.4)
    out["robustness"] = {"gnn_s02": g02, "gnn_s04": g04, "gnn_s04_noise": g04n,
                         "rule_s02": r02, "rule_s04": r04}
    print("[robust] gnn 0.2 %.1f -> 0.4 %.1f (+noise %.1f) | rule 0.2 %.1f -> 0.4 %.1f"
          % (g02[0], g04[0], g04n[0], r02[0], r04[0]), flush=True)

    large = {}
    for size in (32, 64):
        topos = [make_feeder(size, 7000 + size + i) for i in range(3)]
        gc = eval_cost(agent, topos, 20, 0.2)
        rc = eval_rule(topos, 20, 0.2)
        large[size] = {"gnn": gc, "rule": rc}
        print(f"[large {size}] gnn {gc[0]:.1f} (vv {gc[1]:.2f}) | rule {rc[0]:.1f} (vv {rc[1]:.2f})", flush=True)
    out["large_scale"] = large

    lat = {}
    for size in (8, 16, 32, 64):
        topo = make_feeder(size, 123 + size)
        env = FeederEnv(topo, safety=True, seed=1)
        X = env.reset(seed=1)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        agent.act(X, Ahat, mask, deterministic=True)        # warm
        t0 = time.perf_counter()
        for _ in range(200):
            agent.act(X, Ahat, mask, deterministic=True)
        lat[size] = (time.perf_counter() - t0) / 200 * 1000.0   # ms/step
        print(f"[latency {size}] {lat[size]:.3f} ms", flush=True)
    out["latency_ms"] = lat

    with open(os.path.join(RES, "gnn_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] saved gnn_verify.json")


if __name__ == "__main__":
    main()
