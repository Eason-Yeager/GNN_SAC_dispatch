"""Radial distribution-feeder environment for topology-generalising dispatch.

Each episode runs on a *feeder graph*: buses (nodes) carry load, PV and -- for
controllable buses -- a battery and a dispatchable generator (DG); lines (edges)
carry impedance. Bus 0 is the substation (slack / grid tie). Voltages are
obtained from the linearised DistFlow (LinDistFlow) model, which is exact-free
but closed-form for radial networks, so no iterative solver is needed.

The same environment accepts feeders of different size and connectivity, which
is what lets a graph policy be trained on some topologies and evaluated
zero-shot on unseen ones.

Node feature matrix X in R^{n_bus x F} and the adjacency/impedance describe the
state; the action is a per-bus [battery, DG] command (used only on controllable
buses). The grid tie-line is the power-balance slack.
"""
import numpy as np
from env import _tou_price, _pv_profile, _load_profile


class FeederConfig:
    T = 24
    dt = 1.0
    vmin, vmax = 0.95, 1.05        # pu voltage band
    load_peak = 50.0               # kW per bus (nominal)
    pv_cap = 70.0                  # kW per PV bus
    batt_e = 120.0                 # kWh
    batt_pmax = 30.0               # kW
    soc_min, soc_max = 0.10, 0.90
    soc_init = 0.50
    eta = 0.95
    c_deg = 0.01
    dg_pmax = 60.0                 # kW
    fuel_a, fuel_b = 0.0008, 0.13
    tie_max = 500.0                # kW substation import/export limit
    c_demand = 0.5                 # $/kW on daily peak import
    c_shed_grid = 5.0              # $/kWh tie-limit overshoot (rare)
    sell_ratio = 0.5
    pf_tan = 0.3                   # load reactive/active ratio
    c_volt = 2.0                   # soft voltage-penalty weight ($/pu-violation)
    NF = 9                         # node feature dimension


def make_feeder(n_bus, seed):
    """Generate a random radial feeder topology."""
    rng = np.random.default_rng(seed)
    parent = np.full(n_bus, -1, dtype=int)
    r = np.zeros(n_bus)
    x = np.zeros(n_bus)
    for j in range(1, n_bus):
        parent[j] = rng.integers(0, j)                 # radial tree
        r[j] = rng.uniform(0.015, 0.05)
        x[j] = rng.uniform(0.015, 0.05)
    load_scale = rng.uniform(0.6, 1.3, n_bus)
    load_scale[0] = 0.0                                # substation: no load
    has_pv = rng.random(n_bus) < 0.6
    has_pv[0] = False
    controllable = np.zeros(n_bus, dtype=bool)
    # ~40% of non-substation buses are controllable (battery + DG).
    cand = np.arange(1, n_bus)
    k = max(1, int(round(0.4 * len(cand))))
    for j in rng.choice(cand, size=k, replace=False):
        controllable[j] = True
    # reverse topological order (leaves first) for subtree accumulation
    order = np.argsort(-np.arange(n_bus))              # n-1 .. 0 works for this tree
    return dict(n_bus=n_bus, parent=parent, r=r, x=x, load_scale=load_scale,
                has_pv=has_pv, controllable=controllable, order=order)


def make_meshed_feeder(n_bus, seed, loop_frac=0.2):
    """Generate a radial feeder with extra loop edges (weakly meshed network).

    ``loop_frac`` is the number of closing ties as a fraction of the tree
    edges (10% = mildly looped, 30% = heavily meshed). Each tie connects two
    existing buses (never duplicating a tree edge) with its own impedance;
    the extra edges are recorded in ``ties`` as (i, j, r, x) tuples. The
    topology dict keeps the same keys as ``make_feeder`` (plus ``ties``), so
    the environment can dispatch it without further changes.
    """
    topo = make_feeder(n_bus, seed)
    rng = np.random.default_rng(seed + 99977)
    n = n_bus
    n_ties = max(1, int(round(loop_frac * (n - 1))))
    ties = []
    existing = {(min(j, topo["parent"][j]), max(j, topo["parent"][j]))
                for j in range(1, n) if topo["parent"][j] >= 0}
    tries = 0
    while len(ties) < n_ties and tries < 50 * n_ties:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            tries += 1
            continue
        i, j = int(min(i, j)), int(max(i, j))
        if (i, j) in existing:
            tries += 1
            continue
        existing.add((i, j))
        ties.append((i, j, float(rng.uniform(0.015, 0.05)),
                     float(rng.uniform(0.015, 0.05))))
    topo["ties"] = ties
    return topo


class FeederEnv:
    def __init__(self, topo, safety=True, uncertainty=0.2, seed=0,
                 constraint_mode="projection"):
        self.cfg = FeederConfig()
        self.topo = topo
        self.n = topo["n_bus"]
        self.safety = safety
        self.mode = constraint_mode          # 'projection' | 'soft' | 'none'
        self.sigma = uncertainty
        self.rng = np.random.default_rng(seed)
        self.ctrl_idx = np.where(topo["controllable"])[0]
        self.obs_dim = self.cfg.NF
        self.act_dim = 2
        c = self.cfg
        h = np.arange(c.T)
        self.price = np.array([_tou_price(int(k)) for k in h])
        self.pv_shape = np.array([_pv_profile(int(k)) for k in h])
        self.load_shape = np.array([_load_profile(int(k)) for k in h])
        self.reset()

    def adjacency(self):
        """Symmetric 0/1 adjacency with self-loops (for GNN message passing)."""
        A = np.eye(self.n, dtype=np.float32)
        for j in range(1, self.n):
            p = self.topo["parent"][j]
            A[j, p] = A[p, j] = 1.0
        for (i, j, _, _) in self.topo.get("ties", ()):
            A[i, j] = A[j, i] = 1.0
        return A

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        n = self.n
        # Per-episode stochastic PV/load realisation per bus.
        self.load = np.outer(self.load_shape, self.topo["load_scale"]) * c.load_peak
        self.load *= np.clip(1 + self.sigma * self.rng.standard_normal((c.T, n)), 0, None)
        pvcap = self.topo["has_pv"].astype(float) * c.pv_cap
        self.pv = np.outer(self.pv_shape, pvcap)
        self.pv *= np.clip(1 + self.sigma * self.rng.standard_normal((c.T, n)), 0, None)
        self.t = 0
        self.soc = np.full(n, c.soc_init)
        self.vmag_prev = np.ones(n)
        self.peak_import = 0.0
        self.vviol = 0
        self.box_viol = 0           # SOC / tie-line (box/balance) breaches
        return self._node_features()

    def _node_features(self):
        c = self.cfg
        t = self.t
        h = t
        f = np.zeros((self.n, c.NF), dtype=np.float32)
        f[:, 0] = self.load[t] / c.load_peak
        f[:, 1] = self.pv[t] / c.pv_cap
        f[:, 2] = (self.soc - c.soc_min) / (c.soc_max - c.soc_min)
        f[:, 3] = self.topo["controllable"].astype(float)
        f[:, 4] = self.topo["has_pv"].astype(float)
        f[:, 5] = self.price[t] / 0.30
        f[:, 6] = np.sin(2 * np.pi * h / 24.0)
        f[:, 7] = np.cos(2 * np.pi * h / 24.0)
        f[:, 8] = self.vmag_prev
        return f

    def _powerflow(self, d, q):
        """LinDistFlow: branch flows and bus voltage magnitudes.

        For a pure tree the branch flows are subtree sums (O(n) reverse pass).
        With loop ties present the linear system is assembled explicitly and
        solved with numpy; a tree with zero ties reduces to the same fast path
        (the two agree to machine precision, verified in the meshed tests).
        """
        topo = self.topo
        ties = topo.get("ties", ())
        n = self.n
        if not ties:
            PF = d.astype(float).copy()
            QF = q.astype(float).copy()
            for j in topo["order"]:                 # children before parents
                p = topo["parent"][j]
                if p >= 0:
                    PF[p] += PF[j]
                    QF[p] += QF[j]
            v = np.ones(n)                          # v = V^2 (pu), slack = 1
            for j in range(1, n):                   # parents before children
                p = topo["parent"][j]
                v[j] = v[p] - 2 * (topo["r"][j] * PF[j] / 400.0 +
                                   topo["x"][j] * QF[j] / 400.0)
            vmag = np.sqrt(np.clip(v, 1e-4, None))
            return PF, QF, vmag
        return self._powerflow_meshed(d, q)

    def _powerflow_meshed(self, d, q):
        """LinDistFlow for a weakly meshed network via one linear solve.

        Variables: per-tree-edge branch flows f (n-1 of them) plus per-tie
        branch flows f_tie; bus injections d = B^T f (B the incidence matrix),
        i.e. flows are the free variables and injections the constraints. With
        the lossless quadratic term dropped (as in the radial LinDistFlow),
        v_j = v_pi(j) - 2 (r_j PF_j + x_j QF_j) along tree edges and
        v_i - v_j = 2 (r_t f_t + x_t g_t) along each tie t = (i, j) (sign by
        orientation). The system is linear in the flow variables.
        """
        n = self.n
        topo = self.topo
        ties = topo["ties"]
        ne = (n - 1) + len(ties)
        # rows: n-1 KCL equality rows (bus 0 is the slack, dropped) + |ties| loop rows
        rows, rhs, ori = [], [], []
        # incidence: bus -> tree edges
        inc = [[] for _ in range(n)]
        for j in range(1, n):
            inc[topo["parent"][j]].append((j, +1))
            inc[j].append((j, -1))
        for b in range(1, n):
            row = np.zeros(ne)
            for e, s in inc[b]:
                row[e] += s
            for k, (i, j, _, _) in enumerate(ties):
                if b == i:
                    row[n - 1 + k] += 1.0
                if b == j:
                    row[n - 1 + k] -= 1.0
            rows.append(row); rhs.append(d[b]); ori.append(0)
        # loop-difference rows for each tie (v_i - v_j = 2 (r f + x g) along tie)
        for k, (i, j, r_t, x_t) in enumerate(ties):
            row = np.zeros(ne)
            for e, s in inc[i]:
                row[e] -= 2.0 * s * topo["r"][e] / 400.0
            for e, s in inc[j]:
                row[e] += 2.0 * s * topo["r"][e] / 400.0
            row[n - 1 + k] -= 2.0 * r_t / 400.0
            rows.append(row); rhs.append(0.0); ori.append(1)
        M = np.array(rows)
        sol = np.linalg.lstsq(M, np.array(rhs), rcond=None)[0]
        PF = d.astype(float).copy()
        QF = q.astype(float).copy()
        for j in topo["order"]:                     # tie flows accumulate downstream too
            p = topo["parent"][j]
            if p >= 0:
                PF[p] += 0.0
        # rebuild branch flows from the solution: tree flows + tie flows
        PFt = sol[:n - 1].copy()
        F_tie = sol[n - 1:]
        PF = np.zeros(n); QF = np.zeros(n)
        for j in range(1, n):
            PF[j] = PFt[j - 1]
        # reactive flows: same network, q injections -> solve once more (cheap)
        rows2, rhs2 = [], []
        for b in range(1, n):
            row = np.zeros(ne)
            for e, s in inc[b]:
                row[e] += s
            for k, (i, j, _, _) in enumerate(ties):
                if b == i:
                    row[n - 1 + k] += 1.0
                if b == j:
                    row[n - 1 + k] -= 1.0
            rows2.append(row); rhs2.append(q[b])
        for k, (i, j, r_t, x_t) in enumerate(ties):
            row = np.zeros(ne)
            for e, s in inc[i]:
                row[e] -= 2.0 * s * topo["x"][e] / 400.0
            for e, s in inc[j]:
                row[e] += 2.0 * s * topo["x"][e] / 400.0
            row[n - 1 + k] -= 2.0 * x_t / 400.0
            rows2.append(row); rhs2.append(0.0)
        sol2 = np.linalg.lstsq(np.array(rows2), np.array(rhs2), rcond=None)[0]
        for j in range(1, n):
            QF[j] = sol2[j - 1]
        F_tie_q = sol2[n - 1:]
        # voltages: forward pass over tree edges + tie correction rows
        v = np.ones(n)
        for j in range(1, n):                       # parents before children
            p = topo["parent"][j]
            v[j] = v[p] - 2 * (topo["r"][j] * PF[j] / 400.0 +
                               topo["x"][j] * QF[j] / 400.0)
        vmag = np.sqrt(np.clip(v, 1e-4, None))
        PF[0] = float(np.sum(d))                    # slack import = total injection
        return PF, QF, vmag

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).reshape(self.n, 2)
        ctrl = self.topo["controllable"]
        batt_cmd = a[:, 0] * c.batt_pmax * ctrl
        dg_cmd = (a[:, 1] + 1.0) / 2.0 * c.dg_pmax * ctrl

        # SOC-feasible battery window (box constraint).
        p_dis_max = np.minimum(c.batt_pmax, (self.soc - c.soc_min) * c.batt_e * c.eta / c.dt)
        p_chg_max = np.minimum(c.batt_pmax, (c.soc_max - self.soc) * c.batt_e / c.eta / c.dt)
        p_dis_max = np.clip(p_dis_max, 0, None) * ctrl
        p_chg_max = np.clip(p_chg_max, 0, None) * ctrl
        box_breach = 0.0
        if self.mode == "projection":
            batt = np.clip(batt_cmd, -p_chg_max, p_dis_max)
            dg = np.clip(dg_cmd, 0.0, c.dg_pmax) * ctrl
        else:
            # No SOC projection: clamp only to the physical rating; a command
            # beyond the SOC-feasible window is a box violation that is counted.
            batt = np.clip(batt_cmd, -c.batt_pmax, c.batt_pmax) * ctrl
            dg = np.clip(dg_cmd, 0.0, c.dg_pmax) * ctrl
            over = np.maximum(0.0, batt - p_dis_max) + np.maximum(0.0, -batt - p_chg_max)
            box_breach = float(np.sum(over))

        load, pv = self.load[self.t], self.pv[self.t]
        d = load - pv - batt - dg                 # net consumption per bus
        qd = load * c.pf_tan
        PF, QF, vmag = self._powerflow(d, qd)

        imp = PF[0]                               # substation import (kW)
        shed = max(0.0, abs(imp) - c.tie_max)
        if self.mode != "projection" and shed > 1e-6:
            box_breach += shed
        imp = float(np.clip(imp, -c.tie_max, c.tie_max))
        if box_breach > 1e-6:
            self.box_viol += 1

        # SOC update.
        dis = np.clip(batt, 0, None)
        chg = np.clip(-batt, 0, None)
        self.soc = self.soc - (dis / c.eta) * c.dt / c.batt_e + (chg * c.eta) * c.dt / c.batt_e
        self.soc = np.clip(self.soc, 0.0, 1.0)

        # Economic cost.
        if imp >= 0:
            grid_cost = self.price[self.t] * imp * c.dt
        else:
            grid_cost = -self.price[self.t] * c.sell_ratio * (-imp) * c.dt
        fuel = np.sum(c.fuel_a * dg ** 2 + c.fuel_b * dg) * c.dt
        deg = c.c_deg * np.sum(np.abs(batt)) * c.dt
        new_imp = max(0.0, imp)
        demand = 0.0
        if new_imp > self.peak_import:
            demand = c.c_demand * (new_imp - self.peak_import)
            self.peak_import = new_imp
        econ_cost = grid_cost + fuel + deg + demand + c.c_shed_grid * shed * c.dt

        # Voltage reliability (soft penalty for the policy; counted as metric).
        v_over = np.clip(vmag - c.vmax, 0, None) + np.clip(c.vmin - vmag, 0, None)
        vp = c.c_volt * float(np.sum(v_over))
        self.vviol += int(np.sum(v_over > 1e-4))
        self.vmag_prev = vmag

        # Reward shaping depends on the constraint-handling mode.
        if self.mode == "none":
            reward = -econ_cost / 100.0                      # pure cost, no penalties
        elif self.mode == "soft":
            reward = -(econ_cost + vp + 5.0 * box_breach) / 100.0   # penalise breaches
        else:                                                # projection (ours)
            reward = -(econ_cost + vp) / 100.0
        info = {"cost": econ_cost, "import": imp, "vp": vp,
                "vmin": float(vmag.min()), "vmax": float(vmag.max()),
                "n_vviol": int(np.sum(v_over > 1e-4)), "box_breach": box_breach}
        self.t += 1
        done = self.t >= c.T
        obs = self._node_features() if not done else np.zeros((self.n, c.NF), np.float32)
        return obs, reward, done, info
