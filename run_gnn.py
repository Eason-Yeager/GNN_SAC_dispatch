"""Train a topology-aware GNN dispatch policy and test zero-shot generalisation.

Protocol
--------
* Train feeders : N_TRAIN random 8-bus radial feeders.
* Test feeders  : unseen 8-bus feeders (different topology/params/DER siting).
* Extrapolation : a 12-bus feeder (only the GNN can run on it).

Policies: GNN-SAC (ours), MLP-SAC (flattened, structure-blind), rule-based.
All share the feeder environment and its box/balance safety projection, so the
comparison isolates topology generalisation.

Usage:  python run_gnn.py [--episodes 400] [--eval-days 30]
"""
import argparse
import json
import os
import numpy as np
import torch

from grid_env import make_feeder, FeederEnv, FeederConfig
from gnn_policy import normalize_adj, GraphSAC
from cvar_sac import SAC            # reused as the flat MLP-SAC baseline

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

N_TRAIN = 6
N_BUS = 8
TRAIN_SEEDS = list(range(10, 10 + N_TRAIN))
TEST_SEEDS = list(range(100, 106))       # unseen 8-bus feeders
EXTRA_SEED = 500                          # 12-bus extrapolation feeder
SIZE_SWEEP = [6, 8, 10, 12, 14]           # zero-shot size generalisation
FEEDERS_PER_SIZE = 4


def rule_feeder(env):
    """Per-bus price heuristic (charge valley, discharge+DG at peak)."""
    c = env.cfg
    price = env.price[env.t]
    a = np.zeros((env.n, 2), dtype=np.float32)
    if price <= 0.08:
        ab, ad = -0.8, -1.0
    elif price >= 0.30:
        ab, ad = 0.8, 0.3
    else:
        ab, ad = 0.0, -0.5
    a[:, 0] = ab
    a[:, 1] = ad
    return a


def mask_of(topo):
    return topo["controllable"].astype(np.float32)


def train_gnn(train_topos, episodes, seed=0):
    agent = GraphSAC(FeederConfig.NF, N_BUS, layers=3, seed=seed)
    rng = np.random.default_rng(seed)
    warmup, steps, curve = 400, 0, []
    for ep in range(episodes):
        topo = train_topos[int(rng.integers(len(train_topos)))]
        env = FeederEnv(topo, safety=True, seed=1000 + ep)
        X = env.reset(seed=1000 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = mask_of(topo)
        done, ep_r = False, 0.0
        while not done:
            if steps < warmup:
                a = rng.uniform(-1, 1, (env.n, 2)).astype(np.float32)
            else:
                a = agent.act(X, Ahat, mask)
            X2, r, done, info = env.step(a)
            agent.buf.add(X, Ahat, a, r, X2, Ahat, float(done), mask)
            X, ep_r = X2, ep_r + r
            steps += 1
            if steps >= warmup:
                agent.update()
        curve.append(ep_r)
    return agent, np.array(curve)


def train_mlp(train_topos, episodes, seed=0):
    obs_dim, act_dim = N_BUS * FeederConfig.NF, N_BUS * 2
    agent = SAC(obs_dim, act_dim, cvar_lambda=0.0, auto_alpha=True, seed=seed)
    rng = np.random.default_rng(seed + 1)
    warmup, steps, curve = 400, 0, []
    for ep in range(episodes):
        topo = train_topos[int(rng.integers(len(train_topos)))]
        env = FeederEnv(topo, safety=True, seed=2000 + ep)
        X = env.reset(seed=2000 + ep)
        done, ep_r = False, 0.0
        while not done:
            o = X.flatten()
            if steps < warmup:
                a_flat = rng.uniform(-1, 1, act_dim).astype(np.float32)
            else:
                a_flat = agent.act(o)
            a = a_flat.reshape(N_BUS, 2)
            X2, r, done, info = env.step(a)
            agent.buf.add(o, a_flat, r, X2.flatten(), float(done))
            X, ep_r = X2, ep_r + r
            steps += 1
            if steps >= warmup:
                agent.update()
        curve.append(ep_r)
    return agent, np.array(curve)


def eval_gnn(agent, topos, days):
    costs, vviol = [], []
    for topo in topos:
        mask = mask_of(topo)
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=70000 + d)
            X = env.reset(seed=70000 + d)
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            done, tot = False, 0.0
            while not done:
                a = agent.act(X, Ahat, mask, deterministic=True)
                X, r, done, info = env.step(a)
                tot += info["cost"]
            costs.append(tot)
            vviol.append(env.vviol)
    return float(np.mean(costs)), float(np.mean(vviol))


def eval_mlp(agent, topos, days):
    costs, vviol = [], []
    for topo in topos:
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=70000 + d)
            X = env.reset(seed=70000 + d)
            done, tot = False, 0.0
            while not done:
                a = agent.act(X.flatten(), deterministic=True).reshape(N_BUS, 2)
                X, r, done, info = env.step(a)
                tot += info["cost"]
            costs.append(tot)
            vviol.append(env.vviol)
    return float(np.mean(costs)), float(np.mean(vviol))


def eval_rule(topos, days):
    costs, vviol = [], []
    for topo in topos:
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=70000 + d)
            env.reset(seed=70000 + d)
            done, tot = False, 0.0
            while not done:
                _, r, done, info = env.step(rule_feeder(env))
                tot += info["cost"]
            costs.append(tot)
            vviol.append(env.vviol)
    return float(np.mean(costs)), float(np.mean(vviol))


def eval_size_sweep(gnn, days):
    """Zero-shot cost/voltage of the size-8-trained GNN across feeder sizes."""
    out = []
    for size in SIZE_SWEEP:
        topos = [make_feeder(size, 900 + size * 10 + i) for i in range(FEEDERS_PER_SIZE)]
        gc, gv = eval_gnn(gnn, topos, days)
        rc, rv = eval_rule(topos, days)
        out.append({"size": size, "gnn_cost": gc, "gnn_vviol": gv,
                    "rule_cost": rc, "rule_vviol": rv})
        print(f"  [size {size:2d}] gnn {gc:7.1f} (vviol {gv:5.2f}) | "
              f"rule {rc:7.1f} (vviol {rv:5.2f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--eval-days", type=int, default=30)
    args = ap.parse_args()

    train_topos = [make_feeder(N_BUS, s) for s in TRAIN_SEEDS]
    test_topos = [make_feeder(N_BUS, s) for s in TEST_SEEDS]
    extra_topo = make_feeder(12, EXTRA_SEED)

    print("[train] GNN-SAC ...", flush=True)
    gnn, gnn_curve = train_gnn(train_topos, args.episodes)
    print("[train] MLP-SAC ...", flush=True)
    mlp, mlp_curve = train_mlp(train_topos, args.episodes)

    res = {}
    gs = eval_gnn(gnn, train_topos, args.eval_days)
    gu = eval_gnn(gnn, test_topos, args.eval_days)
    gx = eval_gnn(gnn, [extra_topo], args.eval_days)
    ms = eval_mlp(mlp, train_topos, args.eval_days)
    mu = eval_mlp(mlp, test_topos, args.eval_days)
    rs = eval_rule(train_topos, args.eval_days)
    ru = eval_rule(test_topos, args.eval_days)
    rx = eval_rule([extra_topo], args.eval_days)

    res["gnn"] = {"seen_cost": gs[0], "seen_vviol": gs[1],
                  "unseen_cost": gu[0], "unseen_vviol": gu[1],
                  "extra_cost": gx[0], "extra_vviol": gx[1]}
    res["mlp"] = {"seen_cost": ms[0], "seen_vviol": ms[1],
                  "unseen_cost": mu[0], "unseen_vviol": mu[1],
                  "extra_cost": None, "extra_vviol": None}
    res["rule"] = {"seen_cost": rs[0], "seen_vviol": rs[1],
                   "unseen_cost": ru[0], "unseen_vviol": ru[1],
                   "extra_cost": rx[0], "extra_vviol": rx[1]}
    for k in ("gnn", "mlp"):
        g = (res[k]["unseen_cost"] - res[k]["seen_cost"]) / res[k]["seen_cost"]
        res[k]["gen_gap_pct"] = 100.0 * g

    print("[eval ] size sweep (zero-shot) ...", flush=True)
    res["size_sweep"] = eval_size_sweep(gnn, args.eval_days)

    with open(os.path.join(RES, "gnn_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(RES, "gnn_curves.npz"), gnn=gnn_curve, mlp=mlp_curve)

    print("\n           seen    unseen   extrap  gen-gap")
    for k in ("gnn", "mlp", "rule"):
        r = res[k]
        ex = "  n/a" if r["extra_cost"] is None else f"{r['extra_cost']:7.1f}"
        gp = r.get("gen_gap_pct")
        gp = "   -" if gp is None else f"{gp:5.1f}%"
        print(f"  {k:5s} {r['seen_cost']:7.1f} {r['unseen_cost']:7.1f} {ex} {gp}")
    print("[done] saved to", RES)


if __name__ == "__main__":
    main()
