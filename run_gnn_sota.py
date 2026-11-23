"""SOTA-baseline comparison experiments (real numbers, single script).

Implements and measures the four additional baselines requested for the
SOTA table, plus the topology-migration stress test and the adversarial
FDI stress test. All numbers land in results/gnn_sota.json.

New methods
-----------
* GraphPPO-Lagrangian : same GNN encoder, on-policy PPO, NO projection layer;
  constraint violations handled by a Lagrangian multiplier updated by dual
  ascent on the observed violation rate (classic CMDP treatment).
* GAT-SAC             : attention-weighted message passing (no projection;
  soft penalty) -- isolates the value of attention vs plain GCN.
* Padded MLP-SAC      : the flat MLP policy given padded/masked inputs so it
  CAN technically run on other sizes; shows size-generalisation failure is
  architectural, not merely an input-width technicality.
* Indep-DDPG          : one independent DDPG agent per controllable bus
  (fully decentralised, no communication).
* Graph-MADDPG        : per-bus deterministic agents that average neighbour
  actions via the adjacency (one-hop communication).

Stress tests
------------
* mesh migration  : mixed-size GNN (radial training) evaluated zero-shot on
  weakly meshed feeders with loop_frac in {0.1, 0.3} (10-bus).
* FDI / load jump : 32-bus system under 50% pulse load steps + biased
  sensor attack on 15% of nodes.

Usage:
  python run_gnn_sota.py --eps 500 --days 20 [--quick]
  python run_gnn_sota.py --load-agent results/agent_mixed.pt ...
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

from grid_env import make_feeder, make_meshed_feeder, FeederEnv, FeederConfig
from gnn_policy import (normalize_adj, GNNEncoder, GraphActor, GraphCritic,
                        GraphValue, GraphSAC, GraphDDPG)
import run_gnn as RG
import run_gnn_mixed as RM

RES = RG.RES
NF = FeederConfig.NF
N_MAX = 10
TRAIN_SIZES = RM.TRAIN_SIZES
pad = RM.pad


def build_pool():
    return [make_feeder(s, 300 + s * 10 + i) for s in TRAIN_SIZES for i in range(3)]


def eval_10_feeders():
    return [make_feeder(10, 900 + 100 + i) for i in range(4)]


# --------------------------------------------------------------------------
# Generic padded off-policy loop shared by all replay agents below.
# --------------------------------------------------------------------------
def off_policy_train(agent, pool, episodes, mode, seed=0, lag=None, log_curve=False):
    """SAC/DDPG-style loop. ``lag`` optionally couples a Lagrangian multiplier:
    each violation step bumps lambda; the reward adds -lag * violation."""
    rng = np.random.default_rng(seed)
    warmup, steps, curve = 400, 0, []
    for ep in range(episodes):
        topo = pool[int(rng.integers(len(pool)))]
        env = FeederEnv(topo, safety=True, seed=4000 + ep, constraint_mode=mode)
        X = env.reset(seed=4000 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        Xp, Ap, mp = pad(X, Ahat, mask)
        done, ep_r = False, 0.0
        while not done:
            a_full = (rng.uniform(-1, 1, (N_MAX, 2)).astype(np.float32)
                      if steps < warmup else agent.act(Xp, Ap, mp))
            X2, r, done, info = env.step(a_full[:env.n])
            if lag is not None:
                r -= lag[0] * float(info.get("box_breach", 0.0) > 1e-6) * 0.1
            Xp2, Ap2, mp2 = pad(X2, Ahat, mask)
            agent.buf.add(Xp, Ap, a_full, r, Xp2, Ap2, float(done), mp)
            Xp, ep_r = Xp2, ep_r + r
            steps += 1
            if steps >= warmup:
                agent.update()
                if lag is not None and (steps % 50 == 0):
                    # dual ascent: raise lambda while violations persist
                    rate = env.box_viol / max(1, env.t)
                    lag[0] *= float(np.exp(0.02 * (rate - 0.0)))
                    lag[0] = float(np.clip(lag[0], 0.05, 50.0))
        if log_curve:
            curve.append(ep_r)
    return (agent, np.array(curve)) if log_curve else agent


# --------------------------------------------------------------------------
# GAT encoder (attention over neighbours), same interface as GNNEncoder.
# --------------------------------------------------------------------------
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.ws = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.at = nn.ModuleList([nn.Linear(2 * dims[i + 1], 1) for i in range(layers)])

    def forward(self, X, Ahat):
        # Attention logits from current neighbour embeddings; the normalised
        # adjacency Ahat acts as the mask/aggregate basis. We approximate
        # standard GAT with one softmax normalisation over the Ahat support.
        H = X
        for lin, at in zip(self.ws, self.at):
            Z = lin(H)                                   # [B,n,H]
            pair = torch.cat([Z.unsqueeze(2).expand(-1, -1, Z.size(1), -1),
                              Z.unsqueeze(1).expand(-1, Z.size(1), -1, -1)], -1)
            e = at(pair).squeeze(-1)                     # [B,n,n] logits
            e = e - 1e9 * (1.0 - (Ahat > 0).float())     # mask to neighbourhood
            w = torch.softmax(e, dim=-1)                 # attention weights
            H = torch.relu(torch.bmm(w, Z))
        return H


class GraphSACWith(GAT_ := object):
    pass


def make_gat_sac(seed=0, layers=3, hidden=64):
    """SAC whose actor/critic encoders are the GAT above (soft-penalty mode)."""
    enc_actor, enc_critic = GATEncoder(NF, hidden, layers), GATEncoder(NF, hidden, layers)
    agent = GraphSAC(NF, N_MAX, layers=layers, seed=seed)
    # swap encoders, keeping everything else
    actor_enc = agent.actor.enc
    for new in (enc_actor,):
        agent.actor.enc = new
        agent.q1.enc = enc_critic
        agent.q2.enc = GATEncoder(NF, hidden, layers)
        agent.q1t.enc = GATEncoder(NF, hidden, layers)
        agent.q2t.enc = GATEncoder(NF, hidden, layers)
        agent.q1t.load_state_dict(agent.q1.state_dict())
        agent.q2t.load_state_dict(agent.q2.state_dict())
    return agent


# --------------------------------------------------------------------------
# PPO with Lagrangian (CMDP baseline): no projection layer.
# --------------------------------------------------------------------------
class GraphPPOLag:
    """Clipped PPO + Lagrangian on the step-violation indicator."""

    def __init__(self, in_dim, hidden=64, layers=3, gamma=0.99, lam=0.95,
                 clip=0.2, lr=3e-4, epochs=6, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor = GraphActor(in_dim, hidden, layers)
        self.critic = GraphValue(in_dim, hidden, layers)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam_gae, self.clip, self.epochs = gamma, lam, clip, epochs
        self.buf = []
        self.lmbd = 0.5            # Lagrange multiplier on violation indicator
        self.viol_ema = 0.0

    def act(self, X, Ahat, mask, deterministic=False):
        Xt = torch.as_tensor(X[None], dtype=torch.float32)
        At = torch.as_tensor(Ahat[None], dtype=torch.float32)
        Mt = torch.as_tensor(mask[None], dtype=torch.float32)
        with torch.no_grad():
            a, logp = self.actor(Xt, At, Mt, deterministic=deterministic)
            v = self.critic(Xt, At, Mt)
        self._last = (float(logp[0]), float(v[0]))
        return a[0].numpy().astype(np.float32)

    def store(self, X, A, m, a, r, done):
        lp, v = self._last
        self.buf.append([X, A, m, a, r, float(done), lp, v])

    def finish(self):
        if not self.buf:
            return
        r = np.array([b[4] for b in self.buf], np.float32)
        dones = np.array([b[5] for b in self.buf], np.float32)
        v = np.array([b[7] for b in self.buf], np.float32)
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            vnext = 0.0 if (t == len(r) - 1 or dones[t]) else v[t + 1]
            delta = r[t] + self.gamma * vnext * (1 - dones[t]) - v[t]
            gae = delta + self.gamma * self.lam_gae * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        X = torch.as_tensor(np.stack([b[0] for b in self.buf]))
        A = torch.as_tensor(np.stack([b[1] for b in self.buf]))
        m = torch.as_tensor(np.stack([b[2] for b in self.buf]))
        a = torch.as_tensor(np.stack([b[3] for b in self.buf]))
        lp_old = torch.as_tensor(np.array([b[6] for b in self.buf], np.float32))
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)
        for _ in range(self.epochs):
            logp = self.actor.eval_logp(X, A, m, a)
            ratio = torch.exp(logp - lp_old)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = nn.functional.mse_loss(self.critic(X, A, m), ret_t)
            loss = pi_loss + 0.5 * v_loss
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.buf = []


def ppo_lag_train(agent, pool, episodes, seed=0, log_curve=False):
    """PPO training with constraint_mode='soft' env and a Lagrangian penalty."""
    rng = np.random.default_rng(seed)
    curve = []
    for ep in range(episodes):
        topo = pool[int(rng.integers(len(pool)))]
        env = FeederEnv(topo, safety=True, seed=6000 + ep, constraint_mode="soft")
        X = env.reset(seed=6000 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        Xp, Ap, mp = pad(X, Ahat, mask)
        done, ep_r = False, 0.0
        while not done:
            a_full = agent.act(Xp, Ap, mp)
            X2, r, done, info = env.step(a_full[:env.n])
            r = r - agent.lmbd * float(info.get("box_breach", 0.0) > 1e-6)
            agent.store(Xp, Ap, mp, a_full, r, done)
            Xp, ep_r = pad(X2, Ahat, mask)[0], ep_r + r
        # dual ascent on the daily violation indicator
        rate = env.box_viol / max(1, env.cfg.T)
        agent.viol_ema = 0.95 * agent.viol_ema + 0.05 * rate
        agent.lmbd = float(np.clip(agent.lmbd + 0.05 * (agent.viol_ema - 0.01),
                                   0.05, 25.0))
        if (ep + 1) % 8 == 0:
            agent.finish()
        curve.append(ep_r)
    agent.finish()
    return agent, np.array(curve)


# --------------------------------------------------------------------------
# Decentralised / multi-agent baselines.
# --------------------------------------------------------------------------
class IndepDDPG:
    """Parameter-shared independent DDPG over node views: every controllable
    bus runs the SAME small MLP policy but sees ONLY its own features -- no
    adjacency, no communication. This is the standard parameter-shared
    independent-RL baseline (per-bus agents without coordination)."""

    def __init__(self, in_dim, n_train, hidden=32, gamma=0.99, tau=0.005,
                 lr=3e-4, noise=0.1, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.gamma, self.tau, self.noise = gamma, tau, noise
        self.buf = GraphBufferLite(in_dim, n_train)
        self.a_mu = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2), nn.Tanh())
        self.a_mu_t = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 2), nn.Tanh())
        self.a_mu_t.load_state_dict(self.a_mu.state_dict())
        self.q = nn.Sequential(nn.Linear(in_dim + 2, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))
        self.qt = nn.Sequential(nn.Linear(in_dim + 2, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
        self.qt.load_state_dict(self.q.state_dict())
        self.opt_pi = torch.optim.Adam(self.a_mu.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)

    def act(self, X, Ahat, mask, deterministic=False):
        with torch.no_grad():
            a = self.a_mu(torch.as_tensor(X)).numpy()
        if not deterministic:
            a = np.clip(a + self.noise * np.random.standard_normal(a.shape), -1, 1)
        return a.astype(np.float32)

    def update(self, bs=128):
        X, a, r, X2, d = self.buf.sample(bs)
        with torch.no_grad():
            a2 = self.a_mu_t(X2)
            y = r + self.gamma * (1 - d) * self.qt(torch.cat([X2, a2], -1)).squeeze(-1)
        q_loss = nn.functional.mse_loss(self.q(torch.cat([X, a], -1)).squeeze(-1), y)
        self.opt_q.zero_grad(); q_loss.backward(); self.opt_q.step()
        ap = self.a_mu(X)
        pi_loss = -self.q(torch.cat([X, ap], -1)).mean()
        self.opt_pi.zero_grad(); pi_loss.backward(); self.opt_pi.step()
        with torch.no_grad():
            for net, nett in ((self.a_mu, self.a_mu_t), (self.q, self.qt)):
                for p, pt in zip(net.parameters(), nett.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)


class GraphBufferLite:
    """Per-node transition buffer for IndepDDPG / GraphMADDPG (node-wise)."""

    def __init__(self, dim, size=60000):
        self.X = np.zeros((size, dim), np.float32)
        self.a = np.zeros((size, 2), np.float32)
        self.r = np.zeros(size, np.float32)
        self.X2 = np.zeros((size, dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.ptr = self.n = 0
        self.max = size

    def add_nodes(self, X, a, r, X2, d, mask):
        idx = np.where(mask > 0)[0]
        for j in idx:
            i = self.ptr
            self.X[i], self.a[i], self.r[i] = X[j], a[j], r
            self.X2[i], self.d[i] = X2[j], d
            self.ptr = (i + 1) % self.max
            self.n = min(self.n + 1, self.max)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, size=bs)
        t = lambda z: torch.as_tensor(z[idx])
        return t(self.X), t(self.a), t(self.r), t(self.X2), t(self.d)


class GraphMADDPG:
    """Per-bus deterministic agents whose actor input is its own features
    concatenated with the mean of one-hop neighbour features (the only
    communication allowed). Shared Q over pooled node views."""

    def __init__(self, in_dim, hidden=32, gamma=0.99, tau=0.005, lr=3e-4,
                 noise=0.1, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        F2 = in_dim * 2
        self.actor = nn.Sequential(nn.Linear(F2, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 2), nn.Tanh())
        self.actor_t = nn.Sequential(nn.Linear(F2, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 2), nn.Tanh())
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.qf = nn.Sequential(nn.Linear(F2 + 2, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
        self.qf_t = nn.Sequential(nn.Linear(F2 + 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.qf_t.load_state_dict(self.qf.state_dict())
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(self.qf.parameters(), lr=lr)
        self.buf = GraphBufferLite(in_dim * 2, 60000)
        self.gamma, self.tau, self.noise = gamma, tau, noise

    def act(self, X, Ahat, mask, deterministic=False):
        with torch.no_grad():
            Xt = torch.as_tensor(X)
            # one-hop mean of neighbour features via the *binary* adjacency
            A = (Ahat > 0).astype(np.float32)
            deg = np.clip(A.sum(-1, keepdims=True), 1, None)
            Xn = (A @ X) / deg
            Xin = np.concatenate([X, Xn], -1)
            a = self.actor(torch.as_tensor(Xin)).numpy()
        if not deterministic:
            a = np.clip(a + self.noise * np.random.standard_normal(a.shape), -1, 1)
        return a.astype(np.float32)

    def add(self, X, Ahat, a, r, X2, d, mask):
        A = (Ahat > 0).astype(np.float32)
        deg = np.clip(A.sum(-1, keepdims=True), 1, None)
        Xin = np.concatenate([X, (A @ X) / deg], -1)
        Xin2 = np.concatenate([X2, (A @ X2) / deg], -1)
        self.buf.add_nodes(Xin, a, r, Xin2, d, mask)

    def update(self, bs=128):
        X, a, r, X2, d = self.buf.sample(bs)
        with torch.no_grad():
            a2 = self.actor_t(X2)
            y = r + self.gamma * (1 - d) * self.qf_t(torch.cat([X2, a2], -1)).squeeze(-1)
        q_loss = nn.functional.mse_loss(self.qf(torch.cat([X, a], -1)).squeeze(-1), y)
        self.opt_q.zero_grad(); q_loss.backward(); self.opt_q.step()
        ap = self.actor(X)
        pi_loss = -self.qf(torch.cat([X, ap], -1)).mean()
        self.opt_pi.zero_grad(); pi_loss.backward(); self.opt_pi.step()
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.qf.parameters(), self.qf_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)


class PaddedMLPSAC:
    """Flat MLP-SAC that pads/masks observations to N_MAX so it can at least
    RUN on any size; structure-blind, so expect poor transfer."""

    def __init__(self, n_max=N_MAX, seed=0):
        from cvar_sac import SAC
        obs_dim, act_dim = n_max * NF, n_max * 2
        self.agent = SAC(obs_dim, act_dim, cvar_lambda=0.0, auto_alpha=True, seed=seed)
        self.n_max = n_max

    def act(self, X, Ahat, mask, deterministic=False):
        o = np.zeros(self.n_max * NF, np.float32)
        o[:X.size] = X.flatten()
        a_flat = self.agent.act(o, deterministic=deterministic)
        return a_flat.reshape(self.n_max, 2)[:X.shape[0]]

    def train(self, pool, episodes, seed=0, log_curve=False):
        from grid_env import FeederEnv as FE
        rng = np.random.default_rng(seed)
        warmup, steps, curve = 400, 0, []
        for ep in range(episodes):
            topo = pool[int(rng.integers(len(pool)))]
            env = FE(topo, safety=True, seed=8000 + ep, constraint_mode="soft")
            X = env.reset(seed=8000 + ep)
            done, ep_r = False, 0.0
            while not done:
                o = np.zeros(self.n_max * NF, np.float32)
                o[:X.size] = X.flatten()
                a_full = (rng.uniform(-1, 1, self.n_max * 2).astype(np.float32)
                          if steps < warmup else self.agent.act(o))
                X2, r, done, info = env.step(a_full.reshape(self.n_max, 2)[:env.n])
                o2 = np.zeros(self.n_max * NF, np.float32)
                o2[:X2.size] = X2.flatten()
                self.agent.buf.add(o, a_full, r, o2, float(done))
                X, ep_r = X2, ep_r + r
                steps += 1
                if steps >= warmup:
                    self.agent.update()
            curve.append(ep_r)
        return np.array(curve)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def eval_agent(agent, topos, days, mode="projection", nmax=None, noise_fn=None):
    nmax = nmax or max(t["n_bus"] for t in topos)
    costs, vviol, box = [], [], 0
    tot_steps = tot_bus = 0
    for topo in topos:
        mask = topo["controllable"].astype(np.float32)
        for d in range(days):
            env = FeederEnv(topo, safety=True, seed=90000 + d, constraint_mode=mode)
            X = env.reset(seed=90000 + d)
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            Xp, Ap, mp = pad(X, Ahat, mask, nmax)
            done, tot = False, 0.0
            while not done:
                a = agent.act(Xp, Ap, mp, deterministic=True)
                X2, r, done, info = env.step(a[:env.n])
                tot += info["cost"]
                Xp, Ap, mp = pad(X2, Ahat, mask, nmax)
            costs.append(tot); vviol.append(env.vviol)
            box += env.box_viol
            tot_steps += env.cfg.T; tot_bus += env.cfg.T * env.n
    return (float(np.mean(costs)), float(np.mean(vviol)),
            100.0 * box / max(1, tot_steps), box)


def train_all(pool, episodes, seed_base=0):
    out = {}
    t0 = time.time()
    print("[1/7] SAC full (projection) ...", flush=True)
    sac = GraphSAC(NF, N_MAX, layers=3, seed=seed_base)
    _, curve = off_policy_train(sac, pool, episodes, "projection",
                                seed=seed_base, log_curve=True)
    out["sac"] = (sac, curve)
    print("    %.0fs" % (time.time() - t0), flush=True)

    print("[2/7] GAT-SAC (soft penalty) ...", flush=True)
    gat = make_gat_sac(seed=seed_base + 1)
    _, curve = off_policy_train(gat, pool, episodes, "soft", seed=seed_base + 1,
                                log_curve=True)
    out["gat"] = (gat, curve)
    print("    %.0fs" % (time.time() - t0), flush=True)

    print("[3/7] PPO-Lagrangian (no projection) ...", flush=True)
    ppo = GraphPPOLag(NF, layers=3, seed=seed_base + 2)
    _, curve = ppo_lag_train(ppo, pool, episodes, seed=seed_base + 2, log_curve=True)
    out["ppolag"] = (ppo, curve)
    print("    %.0fs" % (time.time() - t0), flush=True)

    print("[4/7] Padded MLP-SAC (soft) ...", flush=True)
    pm = PaddedMLPSAC(seed=seed_base + 3)
    curve = pm.train(pool, episodes, seed=seed_base + 3, log_curve=True)
    out["pmlp"] = (pm, curve)
    print("    %.0fs" % (time.time() - t0), flush=True)

    print("[5/7] Indep-DDPG (per-bus, no comm) ...", flush=True)
    idp = train_indep(pool, episodes, seed=seed_base + 4)
    out["idp"] = (idp, None)
    print("    %.0fs" % (time.time() - t0), flush=True)

    print("[6/7] Graph-MADDPG (one-hop comm) ...", flush=True)
    mad = train_maddpg(pool, episodes, seed=seed_base + 5)
    out["mad"] = (mad, None)
    print("    %.0fs" % (time.time() - t0), flush=True)

    print("[7/7] DDPG (deterministic, projection) ...", flush=True)
    dd = GraphDDPG(NF, N_MAX, layers=3, seed=seed_base + 6)
    _, curve = off_policy_train(dd, pool, episodes, "projection",
                                seed=seed_base + 6, log_curve=True)
    out["ddpg"] = (dd, curve)
    print("    %.0fs total" % (time.time() - t0), flush=True)
    return out


def train_indep(pool, episodes, seed=0):
    ag = IndepDDPG(NF, 60000, seed=seed)
    rng = np.random.default_rng(seed)
    warmup, steps = 400, 0
    for ep in range(episodes):
        topo = pool[int(rng.integers(len(pool)))]
        env = FeederEnv(topo, safety=True, seed=9500 + ep, constraint_mode="soft")
        X = env.reset(seed=9500 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        done = False
        while not done:
            if steps < warmup:
                a = rng.uniform(-1, 1, (env.n, 2)).astype(np.float32)
            else:
                a = ag.act(X, Ahat, mask)
            X2, r, done, info = env.step(a)
            ag.buf.add_nodes(X, a, r, X2, float(done), mask)
            X = X2; steps += 1
            if steps >= warmup:
                ag.update()
    return ag


def train_maddpg(pool, episodes, seed=0):
    ag = GraphMADDPG(NF, seed=seed)
    rng = np.random.default_rng(seed)
    warmup, steps = 400, 0
    for ep in range(episodes):
        topo = pool[int(rng.integers(len(pool)))]
        env = FeederEnv(topo, safety=True, seed=9700 + ep, constraint_mode="soft")
        X = env.reset(seed=9700 + ep)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        done = False
        while not done:
            if steps < warmup:
                a = rng.uniform(-1, 1, (env.n, 2)).astype(np.float32)
            else:
                a = ag.act(X, Ahat, mask)
            X2, r, done, info = env.step(a)
            ag.add(X, Ahat, a, r, X2, float(done), mask)
            X = X2; steps += 1
            if steps >= warmup:
                ag.update()
    return ag


# --------------------------------------------------------------------------
# Stress tests: mesh migration + FDI/load-jump
# --------------------------------------------------------------------------
def eval_mesh(agent, loop_frac, days):
    topos = [make_meshed_feeder(10, 3100 + 10 * int(loop_frac * 100) + i,
                                loop_frac=loop_frac) for i in range(4)]
    return eval_agent(agent, topos, days, "projection")


def perturb_env_noise(env, rng, mode):
    """Wrap the env's profiles with load-jump pulses and/or FDI sensor bias.
    Implemented by post-hoc modification of the stored profiles."""
    T, n = env.load.shape
    if "jump" in mode:
        jumps = (rng.random((T, n)) < 0.08) * 0.5      # 50% pulses, ~8% of hours
        env.load = env.load * (1 + jumps)
    if "fdi" in mode:
        m = (rng.random(n) < 0.15).astype(float)        # 15% of nodes attacked
        env.load = env.load * (1 + 0.3 * m[None, :] * (rng.random((T, n)) - 0.5) * 2)
        env.pv = env.pv * (1 + 0.3 * m[None, :] * (rng.random((T, n)) - 0.5) * 2)


def eval_fdi(agent, topos, days, mode):
    costs, vviol, box = [], [], 0
    rng = np.random.default_rng(4321)
    for topo in topos:
        mask = topo["controllable"].astype(np.float32)
        for d in range(days):
            env = FeederEnv(topo, safety=True, uncertainty=0.2, seed=60000 + d,
                            constraint_mode="projection")
            env.reset(seed=60000 + d)
            perturb_env_noise(env, rng, mode)
            # NOTE: the policy still reads the TRUE features here; to model
            # the sensor attack properly we feed biased observations.
            Ahat = normalize_adj(env.adjacency()).astype(np.float32)
            nb = topo["n_bus"]
            Xp, Ap, mp = pad(env._node_features(), Ahat, mask, nb)
            m = (rng.random(nb) < 0.15).astype(np.float32)
            done, tot = False, 0.0
            while not done:
                Xobs = Xp.copy()
                Xobs[..., 0] *= (1 + 0.3 * m * (rng.random(Xobs.shape[0]) - 0.5) * 2)
                Xobs[..., 1] *= (1 + 0.3 * m * (rng.random(Xobs.shape[0]) - 0.5) * 2)
                a = agent.act(Xobs, Ap, mp, deterministic=True)
                X2, r, done, info = env.step(a[:env.n])
                tot += info["cost"]
                Xp, Ap, mp = pad(env._node_features() if not done else Xp,
                                 Ahat, mask, nb)
            costs.append(tot); vviol.append(env.vviol); box += env.box_viol
    return float(np.mean(costs)), float(np.mean(vviol)), box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=500)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="only run stress tests with a freshly trained SAC")
    args = ap.parse_args()
    if args.quick:
        args.eps, args.days = 40, 2

    pool = build_pool()
    e10 = eval_10_feeders()
    out = {}

    if not args.skip_train:
        trained = train_all(pool, args.eps)
        print("[eval] 10-bus comparison ...", flush=True)
        comp = {}
        for name, (ag, _) in trained.items():
            mode = "soft" if name in ("gat", "pmlp", "ppolag", "idp", "mad") else "projection"
            c, v, boxpct, _ = eval_agent(ag, e10, args.days, mode)
            comp[name] = {"cost": c, "vvolt_pct": v, "box_pct": boxpct}
            print(f"  {name:7s} cost {c:8.1f} volt% {v:5.2f} box% {boxpct:5.2f}",
                  flush=True)
        out["comparison"] = comp

        # convergence speed: episodes to reach 90% of final-100-episode mean return
        conv = {}
        for name, (_, curve) in trained.items():
            if curve is None or len(curve) < 50:
                conv[name] = None
                continue
            c = np.asarray(curve, float)
            final = c[-100:].mean()
            tgt = final + 0.10 * abs(final)
            above = np.where(c <= tgt)[0]
            conv[name] = int(above[0]) if len(above) else len(c)
        out["convergence_ep"] = conv

    # ---- stress: mesh migration (uses the projection SAC) ----
    print("[stress] mesh migration ...", flush=True)
    sac = GraphSAC(NF, N_MAX, layers=3, seed=0)
    off_policy_train(sac, pool, args.eps if not args.quick else 40, "projection", seed=0)
    mesh = {}
    for lf in (0.0, 0.1, 0.3):
        c, v, boxpct, _ = eval_mesh(sac, lf, args.days)
        mesh[str(lf)] = {"cost": c, "vvolt": v, "box_pct": boxpct}
        print(f"  loop {lf:.1f}: cost {c:.1f} vviol {v:.2f} box% {boxpct:.2f}",
              flush=True)
    out["mesh"] = mesh

    # ---- stress: FDI + load jump on 32-bus ----
    print("[stress] FDI / load jump (32-bus) ...", flush=True)
    t32 = [make_feeder(32, 7000 + 32 + i) for i in range(3)]
    c0, v0, _, _ = eval_agent(sac, t32, args.days, "projection")
    c1, v1, b1 = eval_fdi(sac, t32, args.days, "jump+fdi")
    out["fdi32"] = {"normal": {"cost": c0, "vvolt": v0},
                    "attack": {"cost": c1, "vvolt": v1, "box": b1}}
    print(f"  normal {c0:.1f}/vv {v0:.1f} -> attack {c1:.1f}/vv {v1:.1f}", flush=True)

    # rule-based under the same attack
    r0 = RG.eval_rule(t32, args.days)
    costs, vviol = [], []
    rng = np.random.default_rng(4321)
    for topo in t32:
        for d in range(args.days):
            env = FeederEnv(topo, safety=True, uncertainty=0.2, seed=60000 + d,
                            constraint_mode="projection")
            env.reset(seed=60000 + d)
            perturb_env_noise(env, rng, "jump+fdi")
            done, tot = False, 0.0
            while not done:
                _, r, done, info = env.step(RG.rule_feeder(env))
                tot += info["cost"]
            costs.append(tot); vviol.append(env.vviol)
    out["fdi32"]["rule"] = {"normal": {"cost": r0[0], "vvolt": r0[1]},
                            "attack": {"cost": float(np.mean(costs)),
                                       "vvolt": float(np.mean(vviol))}}
    print(f"  rule  normal {r0[0]:.1f} -> attack {np.mean(costs):.1f}", flush=True)

    # latency of the GNN for the SOTA table (10-bus & 32-bus)
    lat = {}
    for size in (10, 32):
        topo = make_feeder(size, 123 + size)
        env = FeederEnv(topo, safety=True, seed=1)
        X = env.reset(seed=1)
        Ahat = normalize_adj(env.adjacency()).astype(np.float32)
        mask = topo["controllable"].astype(np.float32)
        Xp, Ap, mp = pad(X, Ahat, mask, size)
        sac.act(Xp, Ap, mp, deterministic=True)
        t0 = time.perf_counter()
        for _ in range(200):
            sac.act(Xp, Ap, mp, deterministic=True)
        lat[str(size)] = (time.perf_counter() - t0) / 200 * 1000.0
    out["latency_ms"] = lat

    with open(os.path.join(RES, "gnn_sota.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] saved gnn_sota.json")


if __name__ == "__main__":
    main()
