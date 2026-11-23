"""Grid-connected microgrid environment for real-time economic dispatch.

A lightweight, dependency-free (numpy + gym-style) environment modelling a
microgrid with PV, a battery energy storage system (BESS), a dispatchable
diesel generator and a grid tie-line under renewable/load uncertainty.

State  s_t  : [sin(h), cos(h), load, pv, soc, price, net-load forecast]
Action a_t  : [battery power, diesel power]  in [-1, 1]^2  (tanh output)
Reward r_t  : -(operating cost)  (+ optional risk shaping handled by agent)

The grid tie-line acts as the power-balance slack variable. Feasibility
(SOC / power / tie-line limits + power balance) is enforced by the safety
projection layer (see safety_layer.py); when disabled, violations are
simulated physically and counted for the comparison.
"""
import numpy as np


class MicrogridConfig:
    """Static system parameters (Table 1 in the paper)."""
    T = 24                 # steps per episode (hours)
    dt = 1.0               # hours

    pv_cap = 300.0         # kW
    load_base = 180.0      # kW nominal peak

    # Battery (BESS)
    e_cap = 500.0          # kWh
    p_batt_max = 100.0     # kW
    soc_min, soc_max = 0.10, 0.90
    soc_init = 0.50
    eta_c, eta_d = 0.95, 0.95
    c_deg = 0.01           # $/kWh throughput degradation

    # Diesel generator (quadratic fuel cost a*p^2 + b*p)
    p_diesel_max = 200.0   # kW
    fuel_a = 0.0005
    fuel_b = 0.12

    # Grid tie-line
    p_grid_max = 250.0     # kW (import/export limit)
    sell_ratio = 0.5       # export price = sell_ratio * buy price
    c_demand = 0.6         # $/kW charge on the daily peak import (demand charge)

    # Penalties (used only for cost accounting / shaping)
    c_shed = 5.0           # $/kWh unserved energy
    c_curt = 0.05          # $/kWh curtailed PV


def _tou_price(hour):
    """Time-of-use electricity buy price ($/kWh)."""
    if 10 <= hour < 15 or 18 <= hour < 21:
        return 0.30        # peak
    if 7 <= hour < 10 or 15 <= hour < 18 or 21 <= hour < 23:
        return 0.18        # shoulder
    return 0.08            # valley


def _pv_profile(hour):
    """Normalised clear-sky PV shape in [0, 1] (bell around noon)."""
    if 6 <= hour <= 18:
        return float(np.sin(np.pi * (hour - 6) / 12.0) ** 2)
    return 0.0


def _load_profile(hour):
    """Normalised load shape in [0, 1] with morning/evening peaks."""
    base = 0.55
    base += 0.25 * np.exp(-((hour - 8) ** 2) / 6.0)    # morning peak
    base += 0.35 * np.exp(-((hour - 19) ** 2) / 6.0)   # evening peak
    return float(min(base, 1.0))


class MicrogridEnv:
    """Gym-style microgrid dispatch environment.

    Parameters
    ----------
    safety_layer : callable or None
        If provided, called as ``p_b, p_g, info = safety_layer(...)`` to
        project the raw action onto the feasible set (guaranteeing zero
        hard-constraint violations). If None, the raw action is applied and
        any physical infeasibility is counted as a violation.
    uncertainty : float
        Std of multiplicative noise on PV and load (renewable uncertainty).
    seed : int
    """

    def __init__(self, cfg=None, safety_layer=None, uncertainty=0.15, seed=0):
        self.cfg = cfg or MicrogridConfig()
        self.safety = safety_layer
        self.sigma = uncertainty
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 8
        self.act_dim = 2
        self._build_day()
        self.reset()

    def _build_day(self):
        c = self.cfg
        self.hours = np.arange(c.T)
        self.price = np.array([_tou_price(h) for h in self.hours])
        self.pv_clear = np.array([_pv_profile(h) for h in self.hours]) * c.pv_cap
        self.load_nom = np.array([_load_profile(h) for h in self.hours]) * c.load_base

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        # Per-episode stochastic realisation of PV and load (uncertainty).
        self.pv = np.clip(self.pv_clear * (1 + self.sigma * self.rng.standard_normal(c.T)), 0, None)
        self.load = np.clip(self.load_nom * (1 + self.sigma * self.rng.standard_normal(c.T)), 0, None)
        self.t = 0
        self.soc = c.soc_init
        self.viol_count = 0
        self.viol_energy = 0.0
        self.peak_import = 0.0
        return self._obs()

    def _obs(self):
        c = self.cfg
        h = self.hours[self.t]
        net = self.load[self.t] - self.pv[self.t]
        return np.array([
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
            self.load[self.t] / c.load_base,
            self.pv[self.t] / c.pv_cap,
            (self.soc - c.soc_min) / (c.soc_max - c.soc_min),
            self.price[self.t] / 0.30,
            net / c.load_base,
            self.peak_import / c.p_grid_max,
        ], dtype=np.float32)

    def _soc_feasible_power(self):
        """Max discharge (+) and max charge (-) power respecting SOC bounds."""
        c = self.cfg
        # Energy that can still be drawn / absorbed this step.
        e_dis = (self.soc - c.soc_min) * c.e_cap          # kWh available
        e_chg = (c.soc_max - self.soc) * c.e_cap          # kWh headroom
        p_dis_max = min(c.p_batt_max, e_dis * c.eta_d / c.dt)
        p_chg_max = min(c.p_batt_max, e_chg / c.eta_c / c.dt)
        return p_dis_max, p_chg_max                        # both >= 0

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # Map raw action in [-1,1] to physical command ranges (box).
        p_batt_cmd = a[0] * c.p_batt_max                   # + discharge, - charge
        p_diesel_cmd = (a[1] + 1.0) / 2.0 * c.p_diesel_max  # [0, p_diesel_max]

        p_dis_max, p_chg_max = self._soc_feasible_power()
        load, pv = self.load[self.t], self.pv[self.t]

        if self.safety is not None:
            # Feasibility projection layer -> guaranteed-feasible action.
            p_batt, p_diesel, p_grid, curtail, shed = self.safety(
                p_batt_cmd, p_diesel_cmd, load, pv, p_dis_max, p_chg_max, c)
            step_viol = 0
            step_viol_e = 0.0
        else:
            # Raw action applied; physical limits enforced only to keep the
            # simulation valid, and the breach is counted as a violation.
            p_batt = float(np.clip(p_batt_cmd, -p_chg_max, p_dis_max))
            p_diesel = float(np.clip(p_diesel_cmd, 0.0, c.p_diesel_max))
            # SOC violation = how much the command exceeded the feasible range.
            soc_breach = max(0.0, p_batt_cmd - p_dis_max) + max(0.0, -p_batt_cmd - p_chg_max)
            p_grid = load - pv - p_batt - p_diesel
            curtail = max(0.0, -p_grid - c.p_grid_max)     # forced export curtailment
            shed = max(0.0, p_grid - c.p_grid_max)         # forced load shedding
            p_grid = float(np.clip(p_grid, -c.p_grid_max, c.p_grid_max))
            step_viol_e = soc_breach * c.dt + (curtail + shed) * c.dt
            step_viol = int(soc_breach > 1e-6 or curtail > 1e-6 or shed > 1e-6)

        # SOC update (feasible for both branches by construction / clipping).
        if p_batt >= 0:                                    # discharge
            self.soc -= (p_batt / c.eta_d) * c.dt / c.e_cap
        else:                                              # charge
            self.soc += (-p_batt * c.eta_c) * c.dt / c.e_cap
        self.soc = float(np.clip(self.soc, 0.0, 1.0))

        # Operating cost.
        fuel = c.fuel_a * p_diesel ** 2 + c.fuel_b * p_diesel
        if p_grid >= 0:
            grid_cost = self.price[self.t] * p_grid * c.dt
        else:
            grid_cost = -self.price[self.t] * c.sell_ratio * (-p_grid) * c.dt
        deg = c.c_deg * abs(p_batt) * c.dt
        pen = c.c_curt * curtail * c.dt + c.c_shed * shed * c.dt
        cost = fuel * c.dt + grid_cost + deg + pen

        # Demand charge billed incrementally: whenever a step sets a new daily
        # peak import, the marginal peak cost is charged immediately. Summed
        # over the day this equals c_demand * (daily peak import), but the
        # dense per-step signal gives the policy correct credit for peak-setting
        # actions (enabling peak shaving).
        new_import = max(0.0, p_grid)
        if new_import > self.peak_import:
            cost += c.c_demand * (new_import - self.peak_import)
            self.peak_import = new_import

        self.viol_count += step_viol
        self.viol_energy += step_viol_e
        reward = -cost / 100.0                              # scaled for RL

        info = {"cost": cost, "p_batt": p_batt, "p_diesel": p_diesel,
                "p_grid": p_grid, "pv": pv, "load": load, "soc": self.soc,
                "curtail": curtail, "shed": shed, "viol": step_viol}

        self.t += 1
        done = self.t >= c.T
        obs = self._obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, done, info
