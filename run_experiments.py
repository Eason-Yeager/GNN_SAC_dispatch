"""End-to-end experiment runner.

Trains four SAC variants (vanilla / safe-only / cvar-only / ours) and two
non-learning baselines (rule-based, greedy MPC), evaluates them on a common
set of stochastic days, and dumps metrics + trajectories for figure plotting.

Usage:  python run_experiments.py [--episodes 200] [--eval-days 100]
"""
import argparse
import json
import os
import numpy as np
import torch

from env import MicrogridEnv, MicrogridConfig
from safety_layer import project
from cvar_sac import SAC
from baselines import rule_policy, greedy_policy

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

VARIANTS = {
    "vanilla": dict(safety=False, cvar=0.0),
    "safe":    dict(safety=True,  cvar=0.0),
    "cvar":    dict(safety=False, cvar=1.0),
    "ours":    dict(safety=True,  cvar=1.0),
}
TRAIN_SEED_BASE = 1000
EVAL_SEED_BASE = 50000      # disjoint from training days


def make_env(safety, seed, sigma=0.25):
    return MicrogridEnv(cfg=MicrogridConfig(),
                        safety_layer=(project if safety else None),
                        uncertainty=sigma, seed=seed)


def train(variant, episodes, seed=0):
    cfg = VARIANTS[variant]
    env = make_env(cfg["safety"], seed=TRAIN_SEED_BASE + 100 * seed)
    agent = SAC(env.obs_dim, env.act_dim,
                cvar_lambda=cfg["cvar"], cvar_alpha=0.9, seed=seed)
    curve = []
    warmup = 500
    steps = 0
    for ep in range(episodes):
        o = env.reset(seed=TRAIN_SEED_BASE + 100 * seed + 1 + ep)  # random day
        ep_base_r = 0.0
        ep_cost = 0.0
        done = False
        ep_transitions = []
        while not done:
            if steps < warmup:
                a = np.random.uniform(-1, 1, env.act_dim).astype(np.float32)
            else:
                a = agent.act(o)
            o2, r, done, info = env.step(a)
            ep_base_r += r
            ep_cost += info["cost"]
            ep_transitions.append([o, a, r, o2, float(done)])
            o = o2
            steps += 1
            if steps >= warmup:
                agent.update()
                agent.update()
        # CVaR tail penalty on the completed day's cost, distributed uniformly
        # over the day's transitions (dense trajectory-level risk credit).
        pen = agent.cvar_penalty(ep_cost)
        if pen > 0:
            per_step = pen / len(ep_transitions)
            for tr in ep_transitions:
                tr[2] -= per_step
        for tr in ep_transitions:
            agent.buf.add(*tr)
        curve.append(ep_base_r)
    return agent, np.array(curve)


def eval_policy(policy_fn, safety, eval_days, record_day=None):
    """Evaluate a policy_fn(env)->action over eval_days stochastic days."""
    daily_cost, daily_viol, daily_util = [], [], []
    traj = None
    for d in range(eval_days):
        env = make_env(safety, seed=EVAL_SEED_BASE + d)
        o = env.reset(seed=EVAL_SEED_BASE + d)
        done = False
        tot_cost, pv_avail, pv_curt = 0.0, 0.0, 0.0
        rec = {"p_batt": [], "p_diesel": [], "p_grid": [], "pv": [],
               "load": [], "soc": [], "price": list(env.price)}
        while not done:
            a = policy_fn(env, o)
            o, r, done, info = env.step(a)
            tot_cost += info["cost"]
            pv_avail += info["pv"]
            pv_curt += info["curtail"]
            for k in ("p_batt", "p_diesel", "p_grid", "pv", "load", "soc"):
                rec[k].append(info[k])
        daily_cost.append(tot_cost)
        daily_viol.append(env.viol_count)
        daily_util.append(1.0 - pv_curt / max(pv_avail, 1e-6))
        if record_day is not None and d == record_day:
            traj = rec
    daily_cost = np.array(daily_cost)
    k = max(1, int(0.1 * len(daily_cost)))
    cvar = np.mean(np.sort(daily_cost)[-k:])        # CVaR_0.9 of daily cost
    return {
        "mean_cost": float(np.mean(daily_cost)),
        "std_cost": float(np.std(daily_cost)),
        "cvar90_cost": float(cvar),
        "mean_viol": float(np.mean(daily_viol)),
        "viol_rate": float(np.mean(np.array(daily_viol) > 0)),
        "renew_util": float(np.mean(daily_util)),
    }, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eval-days", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    metric_keys = ["mean_cost", "cvar90_cost", "mean_viol", "viol_rate", "renew_util"]
    per_seed = {v: {k: [] for k in metric_keys} for v in VARIANTS}
    curves = {}
    sample_day = None

    for seed in range(args.seeds):
        for v in VARIANTS:
            print(f"[train] {v} seed={seed} ...", flush=True)
            agent, curve = train(v, args.episodes, seed=seed)
            if seed == 0:
                curves[v] = curve
            safety = VARIANTS[v]["safety"]
            rec_day = 0 if (v == "ours" and seed == 0) else None
            pol = lambda env, o, ag=agent: ag.act(o, deterministic=True)
            m, traj = eval_policy(pol, safety, args.eval_days, record_day=rec_day)
            for k in metric_keys:
                per_seed[v][k].append(m[k])
            print(f"[eval ] {v:8s} seed={seed} cost={m['mean_cost']:.1f} "
                  f"cvar90={m['cvar90_cost']:.1f} viol={m['mean_viol']:.2f}", flush=True)
            if v == "ours" and seed == 0:
                sample_day = traj

    # Aggregate RL variants over seeds (mean +/- std).
    results = {}
    for v in VARIANTS:
        results[v] = {}
        for k in metric_keys:
            arr = np.array(per_seed[v][k])
            results[v][k] = float(arr.mean())
            results[v][k + "_std"] = float(arr.std())

    # Baselines are deterministic given the eval days -> run once.
    for name, fn in (("rule", lambda env, o: rule_policy(env)),
                     ("greedy", lambda env, o: greedy_policy(env))):
        print(f"[eval ] {name} ...", flush=True)
        m, _ = eval_policy(fn, True, args.eval_days)
        results[name] = {k: m[k] for k in metric_keys}
        for k in metric_keys:
            results[name][k + "_std"] = 0.0
        print(f"[eval ] {name:8s} cost={m['mean_cost']:.1f} "
              f"cvar90={m['cvar90_cost']:.1f} viol={m['mean_viol']:.2f}", flush=True)

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(RES, "curves.npz"), **curves)
    if sample_day is not None:
        np.savez(os.path.join(RES, "sample_day.npz"),
                 **{k: np.array(v) for k, v in sample_day.items()})
    print("[done] results saved to", RES)


if __name__ == "__main__":
    main()
