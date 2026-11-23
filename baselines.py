"""Non-learning baseline dispatch policies.

* rule_policy   : a priced-based heuristic (charge in valley / PV surplus,
                  discharge and run diesel at peak).
* greedy_policy : a myopic one-step optimiser (grid search over the action
                  box, pick the lowest immediate cost) -- a strong, perfect
                  current-information greedy / horizon-1 MPC reference.

Both return a raw action in [-1, 1]^2 and are evaluated through the same
environment (with the feasibility projection layer) as the RL agents.
"""
import numpy as np


def rule_policy(env):
    c = env.cfg
    t = env.t
    price = env.price[t]
    pv, load, soc = env.pv[t], env.load[t], env.soc

    # Battery: charge in valley or on PV surplus, discharge at peak.
    if price <= 0.08:
        a_batt = -0.8
    elif price >= 0.30:
        a_batt = 0.8
    elif pv > load:
        a_batt = -0.5
    else:
        a_batt = 0.0
    if soc < c.soc_min + 0.05:          # avoid deep discharge
        a_batt = min(a_batt, 0.0)

    # Diesel: off in valley, part load at shoulder, higher at peak.
    if price >= 0.30:
        a_diesel = 0.0                  # -> half of p_max
    elif price >= 0.18:
        a_diesel = -0.5
    else:
        a_diesel = -1.0                 # off
    return np.array([a_batt, a_diesel], dtype=np.float32)


def _snapshot(env):
    return (env.soc, env.t, env.viol_count, env.viol_energy, env.peak_import)


def _restore(env, s):
    env.soc, env.t, env.viol_count, env.viol_energy, env.peak_import = s


def greedy_policy(env, n_grid=11):
    """One-step look-ahead: pick the action with lowest immediate cost."""
    best_a, best_cost = None, np.inf
    batt_grid = np.linspace(-1, 1, n_grid)
    diesel_grid = np.linspace(-1, 1, n_grid)
    for ab in batt_grid:
        for ad in diesel_grid:
            s = _snapshot(env)
            _, _, _, info = env.step(np.array([ab, ad], dtype=np.float32))
            cost = info["cost"]
            _restore(env, s)
            if cost < best_cost:
                best_cost, best_a = cost, np.array([ab, ad], dtype=np.float32)
    return best_a
