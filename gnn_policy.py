"""Graph neural network actor-critic for topology-generalising dispatch.

The policy is a message-passing GNN over the feeder graph, so the SAME weights
act on feeders of different connectivity and size. A per-node action head emits
the [battery, DG] command for every bus; only controllable buses are used by
the environment. The critic encodes the state graph, conditions on the joint
action, mean-pools over buses and outputs a scalar Q.

Because every weight matrix is node-count independent, a model trained on
n-bus feeders can be evaluated zero-shot on feeders with a different number of
buses -- which an MLP policy (fixed input width) cannot do.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A):
    """Symmetric normalisation D^-1/2 (A) D^-1/2 of an adjacency with self-loops."""
    d = A.sum(-1)
    dinv = np.power(np.clip(d, 1e-8, None), -0.5)
    return (dinv[:, None] * A) * dinv[None, :]


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.ws = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])

    def forward(self, X, Ahat):
        # X: [B,n,F]  Ahat: [B,n,n]
        H = X
        for lin in self.ws:
            H = torch.relu(torch.bmm(Ahat, lin(H)))
        return H                                   # [B,n,hidden]


class GraphActor(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, act_dim=2):
        super().__init__()
        self.enc = GNNEncoder(in_dim, hidden, layers)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, X, Ahat, mask, deterministic=False, with_logp=True):
        H = self.enc(X, Ahat)
        mu = self.mu(H)
        log_std = torch.clamp(self.log_std(H), -20, 2)
        std = torch.exp(log_std)
        u = mu if deterministic else mu + std * torch.randn_like(std)
        a = torch.tanh(u)                          # [B,n,act_dim]
        logp = None
        if with_logp:
            dist = torch.distributions.Normal(mu, std)
            lp = dist.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u)))
            lp = lp.sum(-1)                        # [B,n]
            logp = (lp * mask).sum(-1)            # sum over controllable buses
        return a, logp

    def eval_logp(self, X, Ahat, mask, a):
        """Log-prob of a given squashed action under the current policy."""
        H = self.enc(X, Ahat)
        mu = self.mu(H)
        std = torch.exp(torch.clamp(self.log_std(H), -20, 2))
        u = torch.atanh(torch.clamp(a, -0.999, 0.999))
        dist = torch.distributions.Normal(mu, std)
        lp = dist.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u)))
        return (lp.sum(-1) * mask).sum(-1)


class GraphCritic(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, act_dim=2):
        super().__init__()
        self.enc = GNNEncoder(in_dim, hidden, layers)
        self.f = nn.Sequential(nn.Linear(hidden + act_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 1)

    def forward(self, X, Ahat, a, mask):
        H = self.enc(X, Ahat)                      # [B,n,hidden]
        h = self.f(torch.cat([H, a], -1))         # [B,n,hidden]
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.out(pooled).squeeze(-1)        # [B]


class GraphBuffer:
    def __init__(self, n, F, size=60000):
        self.X = np.zeros((size, n, F), np.float32)
        self.A = np.zeros((size, n, n), np.float32)
        self.a = np.zeros((size, n, 2), np.float32)
        self.r = np.zeros(size, np.float32)
        self.X2 = np.zeros((size, n, F), np.float32)
        self.A2 = np.zeros((size, n, n), np.float32)
        self.d = np.zeros(size, np.float32)
        self.m = np.zeros((size, n), np.float32)
        self.ptr = self.n = 0
        self.max = size

    def add(self, X, A, a, r, X2, A2, d, m):
        i = self.ptr
        self.X[i], self.A[i], self.a[i], self.r[i] = X, A, a, r
        self.X2[i], self.A2[i], self.d[i], self.m[i] = X2, A2, d, m
        self.ptr = (i + 1) % self.max
        self.n = min(self.n + 1, self.max)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, size=bs)
        t = lambda z: torch.as_tensor(z[idx])
        return (t(self.X), t(self.A), t(self.a), t(self.r),
                t(self.X2), t(self.A2), t(self.d), t(self.m))


class GraphSAC:
    def __init__(self, in_dim, n_train, hidden=64, layers=2, gamma=0.99,
                 tau=0.005, lr=3e-4, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        mk_a = lambda: GraphActor(in_dim, hidden, layers)
        mk_q = lambda: GraphCritic(in_dim, hidden, layers)
        self.actor = mk_a()
        self.q1, self.q2 = mk_q(), mk_q()
        self.q1t, self.q2t = mk_q(), mk_q()
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) +
                                      list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha = 1.0
        self.gamma, self.tau = gamma, tau
        self.buf = GraphBuffer(n_train, in_dim)

    def act(self, X, Ahat, mask, deterministic=False):
        with torch.no_grad():
            Xt = torch.as_tensor(X[None], dtype=torch.float32)
            At = torch.as_tensor(Ahat[None], dtype=torch.float32)
            Mt = torch.as_tensor(mask[None], dtype=torch.float32)
            a, _ = self.actor(Xt, At, Mt, deterministic=deterministic, with_logp=False)
        return a[0].numpy()

    def update(self, bs=128):
        X, A, a, r, X2, A2, d, m = self.buf.sample(bs)
        target_entropy = -2.0 * m.sum(-1).mean()
        with torch.no_grad():
            a2, logp2 = self.actor(X2, A2, m)
            q_t = torch.min(self.q1t(X2, A2, a2, m), self.q2t(X2, A2, a2, m)) - self.alpha * logp2
            target = r + self.gamma * (1 - d) * q_t
        q1_loss = F.mse_loss(self.q1(X, A, a, m), target)
        q2_loss = F.mse_loss(self.q2(X, A, a, m), target)
        self.q_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_opt.step()

        ap, logp = self.actor(X, A, m)
        q_pi = torch.min(self.q1(X, A, ap, m), self.q2(X, A, ap, m))
        pi_loss = (self.alpha * logp - q_pi).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
        self.a_opt.zero_grad()
        alpha_loss.backward()
        self.a_opt.step()
        self.alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)


class GraphDDPG:
    """Deterministic graph actor-critic (DDPG) with the same GNN encoder."""

    def __init__(self, in_dim, n_train, hidden=64, layers=3, gamma=0.99,
                 tau=0.005, lr=3e-4, noise=0.1, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor = GraphActor(in_dim, hidden, layers)
        self.actor_t = GraphActor(in_dim, hidden, layers)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q = GraphCritic(in_dim, hidden, layers)
        self.qt = GraphCritic(in_dim, hidden, layers)
        self.qt.load_state_dict(self.q.state_dict())
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = GraphBuffer(n_train, in_dim)
        self.gamma, self.tau, self.noise = gamma, tau, noise

    def act(self, X, Ahat, mask, deterministic=False):
        with torch.no_grad():
            Xt = torch.as_tensor(X[None], dtype=torch.float32)
            At = torch.as_tensor(Ahat[None], dtype=torch.float32)
            Mt = torch.as_tensor(mask[None], dtype=torch.float32)
            a, _ = self.actor(Xt, At, Mt, deterministic=True, with_logp=False)
            a = a[0].numpy()
        if not deterministic:
            a = np.clip(a + self.noise * np.random.standard_normal(a.shape), -1, 1)
        return a.astype(np.float32)

    def update(self, bs=128):
        X, A, a, r, X2, A2, d, m = self.buf.sample(bs)
        with torch.no_grad():
            a2, _ = self.actor_t(X2, A2, m, deterministic=True, with_logp=False)
            y = r + self.gamma * (1 - d) * self.qt(X2, A2, a2, m)
        q_loss = F.mse_loss(self.q(X, A, a, m), y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        ap, _ = self.actor(X, A, m, deterministic=True, with_logp=False)
        pi_loss = -self.q(X, A, ap, m).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
        with torch.no_grad():
            for net, nett in ((self.q, self.qt), (self.actor, self.actor_t)):
                for p, pt in zip(net.parameters(), nett.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)


class GraphValue(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=3):
        super().__init__()
        self.enc = GNNEncoder(in_dim, hidden, layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, X, Ahat, mask):
        H = self.enc(X, Ahat)
        pooled = (H * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


class GraphPPO:
    """On-policy clipped PPO with the same GNN encoder (state-value critic)."""

    def __init__(self, in_dim, hidden=64, layers=3, gamma=0.99, lam=0.95,
                 clip=0.2, lr=3e-4, epochs=6, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor = GraphActor(in_dim, hidden, layers)
        self.critic = GraphValue(in_dim, hidden, layers)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam, self.clip, self.epochs = gamma, lam, clip, epochs
        self.buf = []

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
        """GAE + returns, then PPO epochs over the collected rollout."""
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
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
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
            v_loss = F.mse_loss(self.critic(X, A, m), ret_t)
            loss = pi_loss + 0.5 * v_loss
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.buf = []
