"""Soft Actor-Critic with optional CVaR risk-aware reward shaping.

The CVaR term follows the Rockafellar-Uryasev surrogate: we track a running
estimate of the alpha-quantile (VaR) of step costs, nu, and subtract a tail
penalty  (lambda / (1 - alpha)) * max(cost - nu, 0)  from the reward. This
drives the policy away from the expensive (1 - alpha) tail of outcomes, i.e.
it optimises a CVaR-regularised objective rather than the plain expectation.

Combined with the feasibility projection layer (safety_layer.py), the full
method ("Safe + CVaR SAC") targets low mean cost, low tail cost, and zero
hard-constraint violations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False, with_logp=True):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = torch.exp(log_std)
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(std)
        a = torch.tanh(u)
        logp = None
        if with_logp:
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(u).sum(-1)
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=100000):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.ptr, self.n, self.max = 0, 0, size

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.max
        self.n = min(self.n + 1, self.max)

    def sample(self, bs=256):
        idx = np.random.randint(0, self.n, size=bs)
        t = lambda x: torch.as_tensor(x[idx])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class SAC:
    """SAC agent with optional CVaR reward shaping."""

    def __init__(self, obs_dim, act_dim, cvar_lambda=0.0, cvar_alpha=0.9,
                 gamma=0.99, tau=0.005, lr=3e-4, alpha=0.2, auto_alpha=True,
                 seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor = GaussianActor(obs_dim, act_dim)
        self.q1 = QNet(obs_dim, act_dim)
        self.q2 = QNet(obs_dim, act_dim)
        self.q1t = QNet(obs_dim, act_dim)
        self.q2t = QNet(obs_dim, act_dim)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) +
                                      list(self.q2.parameters()), lr=lr)
        self.buf = ReplayBuffer(obs_dim, act_dim)
        self.gamma, self.tau = gamma, tau
        # Entropy temperature (optionally auto-tuned).
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        self.cvar_lambda, self.cvar_alpha = cvar_lambda, cvar_alpha
        self.ep_cost_hist = []    # recent episodic (daily) costs for VaR estimate
        self.nu = 0.0             # running VaR (alpha-quantile of daily cost)
        self.kappa = 100.0        # reward scale (matches env)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32),
                              deterministic=deterministic, with_logp=False)
        return a.numpy()

    def cvar_penalty(self, episode_cost):
        """CVaR (Rockafellar-Uryasev) tail penalty on the *daily* cost.

        Returns a non-negative penalty (in reward units) added to the terminal
        transition, driving the policy away from the expensive (1-alpha) tail
        of the daily-cost distribution. Updates the running VaR estimate.
        """
        if self.cvar_lambda <= 0:
            return 0.0
        if len(self.ep_cost_hist) >= 20:
            self.nu = float(np.quantile(self.ep_cost_hist, self.cvar_alpha))
            tail = max(0.0, episode_cost - self.nu)
            pen = self.cvar_lambda / (1 - self.cvar_alpha) * tail / self.kappa
        else:
            pen = 0.0
        self.ep_cost_hist.append(episode_cost)
        if len(self.ep_cost_hist) > 500:
            self.ep_cost_hist.pop(0)
        return pen

    def update(self, bs=256):
        o, a, r, o2, d = self.buf.sample(bs)
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q_t = torch.min(self.q1t(o2, a2), self.q2t(o2, a2)) - self.alpha * logp2
            target = r + self.gamma * (1 - d) * q_t
        q1_loss = F.mse_loss(self.q1(o, a), target)
        q2_loss = F.mse_loss(self.q2(o, a), target)
        self.q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_opt.step()

        ap, logp = self.actor(o)
        q_pi = torch.min(self.q1(o, ap), self.q2(o, ap))
        pi_loss = (self.alpha * logp - q_pi).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.a_opt.zero_grad()
            alpha_loss.backward()
            self.a_opt.step()
            self.alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
