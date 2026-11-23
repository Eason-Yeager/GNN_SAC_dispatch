"""Communication dropout / telemetry-noise robustness (real numbers).

Trains an 8-bus GNN and MLP (as in run_gnn), then evaluates both under random
Bernoulli feature dropout (telemetry loss) at p in {0, 0.1, 0.2, 0.3}. The
projection layer keeps box/balance violations at zero regardless; we measure how
operating cost degrades as telemetry is lost.
"""
import json
import os
import numpy as np

from grid_env import make_feeder, FeederEnv
from gnn_policy import normalize_adj
import run_gnn as RG

RES = RG.RES


def eval_dropout(agent, topos, days, p, is_mlp=False, seed=0):
    rng = np.random.default_rng(seed)
    costs, vviol = [], []
    for topo in topos:
        mask = topo["controllable"].astype(np.float32)
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=95000 + d)
            X = env.reset(seed=95000 + d)
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            done, tot = False, 0.0
            while not done:
                if p > 0:                       # Bernoulli telemetry dropout per node
                    keep = (rng.random(X.shape[0]) >= p).astype(np.float32)[:, None]
                    Xo = X * keep
                else:
                    Xo = X
                if is_mlp:
                    a = agent.act(Xo.flatten(), deterministic=True).reshape(env.n, 2)
                else:
                    a = agent.act(Xo, Ahat, mask, deterministic=True)
                X, r, done, info = env.step(a)
                tot += info["cost"]
            costs.append(tot); vviol.append(env.vviol)
    return float(np.mean(costs)), float(np.mean(vviol))


def main():
    print("[train] 8-bus GNN + MLP ...", flush=True)
    train8 = [make_feeder(8, s) for s in RG.TRAIN_SEEDS]
    test8 = [make_feeder(8, s) for s in RG.TEST_SEEDS]
    gnn, _ = RG.train_gnn(train8, 500)
    mlp, _ = RG.train_mlp(train8, 500)

    out = {"gnn": {}, "mlp": {}}
    base_g = eval_dropout(gnn, test8, 25, 0.0)[0]
    base_m = eval_dropout(mlp, test8, 25, 0.0, is_mlp=True)[0]
    for p in (0.0, 0.1, 0.2, 0.3):
        gc, gv = eval_dropout(gnn, test8, 25, p)
        mc, mv = eval_dropout(mlp, test8, 25, p, is_mlp=True)
        out["gnn"][str(p)] = {"cost": gc, "vviol": gv, "incr_pct": 100 * (gc - base_g) / base_g}
        out["mlp"][str(p)] = {"cost": mc, "vviol": mv, "incr_pct": 100 * (mc - base_m) / base_m}
        print(f"  p={p}: GNN {gc:.1f} (+{out['gnn'][str(p)]['incr_pct']:.1f}%) | "
              f"MLP {mc:.1f} (+{out['mlp'][str(p)]['incr_pct']:.1f}%)", flush=True)

    with open(os.path.join(RES, "gnn_dropout.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] saved gnn_dropout.json")


if __name__ == "__main__":
    main()
