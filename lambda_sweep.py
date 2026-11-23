"""CVaR risk-weight (lambda) sensitivity sweep.

Trains the Safe+CVaR configuration for several risk weights lambda and records
the mean-cost / tail-cost (CVaR_0.9) trade-off, i.e. the risk-return frontier.
This is the principled demonstration of the CVaR knob: as lambda grows the
policy accepts a little more mean cost to shave the expensive tail.

Usage:  python lambda_sweep.py [--episodes 400] [--eval-days 80]
"""
import argparse
import json
import os
import numpy as np

import run_experiments as R
from safety_layer import project
from cvar_sac import SAC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = R.RES
LAMBDAS = [0.0, 1.0, 2.0, 4.0, 8.0]


def train_lambda(lam, episodes, seed=0):
    env = R.make_env(True, seed=R.TRAIN_SEED_BASE + 100 * seed)
    agent = SAC(env.obs_dim, env.act_dim, cvar_lambda=lam, cvar_alpha=0.9, seed=seed)
    warmup, steps = 500, 0
    for ep in range(episodes):
        o = env.reset(seed=R.TRAIN_SEED_BASE + 100 * seed + 1 + ep)
        ep_cost, done, trans = 0.0, False, []
        while not done:
            a = (np.random.uniform(-1, 1, env.act_dim).astype(np.float32)
                 if steps < warmup else agent.act(o))
            o2, r, done, info = env.step(a)
            ep_cost += info["cost"]
            trans.append([o, a, r, o2, float(done)])
            o = o2
            steps += 1
            if steps >= warmup:
                agent.update()
                agent.update()
        pen = agent.cvar_penalty(ep_cost)
        if pen > 0:
            for tr in trans:
                tr[2] -= pen / len(trans)
        for tr in trans:
            agent.buf.add(*tr)
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--eval-days", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    out = []
    for lam in LAMBDAS:
        means, cvars = [], []
        for seed in range(args.seeds):
            agent = train_lambda(lam, args.episodes, seed=seed)
            pol = lambda env, o, ag=agent: ag.act(o, deterministic=True)
            m, _ = R.eval_policy(pol, True, args.eval_days)
            means.append(m["mean_cost"])
            cvars.append(m["cvar90_cost"])
        rec = {"lambda": lam,
               "mean_cost": float(np.mean(means)),
               "mean_cost_std": float(np.std(means)),
               "cvar90_cost": float(np.mean(cvars)),
               "cvar90_cost_std": float(np.std(cvars))}
        out.append(rec)
        print(f"[sweep] lambda={lam:4.1f}  mean={rec['mean_cost']:.1f}  "
              f"cvar90={rec['cvar90_cost']:.1f}", flush=True)

    with open(os.path.join(RES, "lambda_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] lambda sweep saved")


if __name__ == "__main__":
    main()
