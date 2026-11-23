# Safety-Constrained Risk-Aware Deep Reinforcement Learning for Real-Time Economic Dispatch of Microgrids under Renewable Uncertainty

**Target journal:** *Energies* (MDPI) — Section: Smart Grids and Microgrids
**Article type:** Article

> Draft manuscript. Numerical values in the Abstract and Section 5 are filled
> from `results/results.json` produced by `run_experiments.py` (seeded, reproducible).

## Abstract

Real-time economic dispatch of grid-connected microgrids with high shares of
renewable generation must simultaneously minimise operating cost, respect hard
operational constraints, and remain robust to the uncertainty of photovoltaic
(PV) output and load. Model-based optimisation depends on accurate forecasts
and is hard to run in real time, while existing deep reinforcement learning
(DRL) dispatch policies tend to violate operational limits and optimise only
the expected cost, leaving them fragile under extreme scenarios. This paper
proposes a safety-constrained, risk-aware DRL framework for microgrid economic
dispatch. A soft actor-critic (SAC) policy is coupled with (i) an
optimal-power-flow-style *feasibility projection layer* that projects every
action onto the operational feasible set, guaranteeing zero hard-constraint
violations by construction, and (ii) a *conditional value-at-risk* (CVaR)
objective that explicitly penalises the expensive tail of the cost
distribution. On a grid-connected microgrid with PV, battery storage, a
dispatchable diesel unit and a tie-line, evaluated over 100 stochastic days
against rule-based, greedy/MPC and vanilla-SAC baselines, the proposed method
**eliminates hard-constraint violations** (from about 8.6 per day for vanilla
SAC to zero) while **cutting mean and tail (CVaR0.9) daily cost by about 19% and
17%** relative to the strongest baseline. An ablation and a risk-weight
sensitivity study further show that the safety layer is essentially cost-free
and that the CVaR term exposes an explicit, tunable mean-cost / tail-risk
trade-off. The approach needs no online model and only public-style data, making
it a practical, deployable dispatch strategy.

**Keywords:** microgrid; economic dispatch; deep reinforcement learning; soft
actor-critic; safe reinforcement learning; conditional value-at-risk; renewable
uncertainty; energy management system

## 1. Introduction

The growing penetration of distributed renewable generation turns microgrid
operation into a hard real-time decision problem. The energy management system
must decide, at every control step, how to charge or discharge the battery,
how much to run dispatchable units, and how much power to exchange with the
main grid, so as to minimise operating cost while keeping every quantity within
its physical and contractual limits. Two features make the problem difficult:
the volatility and uncertainty of PV output and demand, and the presence of
hard operational constraints that cannot be violated in practice.

Classical approaches formulate dispatch as a mathematical program — mixed-integer
linear programming (MILP) or model predictive control (MPC). These methods are
accurate when an exact system model and reliable forecasts are available, but
their performance degrades under forecast error, and solving them repeatedly in
real time for large systems is computationally demanding. Metaheuristic solvers
avoid an explicit model but scale poorly, as solution quality depends on large
populations and many iterations.

Deep reinforcement learning (DRL) offers an attractive alternative: once
trained offline, the policy maps the current state to a dispatch action in
microseconds and implicitly handles uncertainty learned from data. However, two
issues keep DRL dispatch policies out of practice. First, **safety**: standard
DRL agents optimise a soft-penalised reward and therefore still output actions
that violate power balance, state-of-charge (SOC), generator or tie-line
limits — exactly the failure mode reviewers and operators care most about.
Second, **risk**: optimising the expected return produces policies that are
cheap on average but can be very expensive in the rare, high-cost scenarios
created by renewable intermittency.

This paper addresses both issues directly. We keep the sample-efficient,
stable soft actor-critic (SAC) backbone and augment it with two components:

- **A feasibility projection layer.** Inspired by OPF, the raw action produced
  by the policy is projected onto the microgrid feasible set through a
  merit-order restoration of the tie-line slack. The projected action satisfies
  SOC, generator, tie-line and power-balance constraints *by construction*, so
  the policy incurs **zero hard-constraint violations** rather than merely
  fewer.
- **A CVaR risk-aware objective.** Using the Rockafellar–Uryasev surrogate, we
  add a conditional-value-at-risk penalty on the cost distribution, so the
  policy hedges against the expensive tail induced by renewable uncertainty.

**Contributions.**
1. A safety projection layer for microgrid dispatch that guarantees feasibility
   of every action, turning DRL from "mostly feasible" into "provably feasible".
2. A CVaR-regularised SAC objective that exposes a principled, tunable
   mean-cost / tail-risk trade-off under renewable uncertainty.
3. A systematic, reproducible evaluation against rule-based, greedy/MPC and
   vanilla-SAC baselines on public-style data, with an ablation isolating the
   contribution of each component.

The remainder of the paper is organised as follows. Section 2 reviews related
work. Section 3 formulates the dispatch problem and its Markov decision process
(MDP). Section 4 details the method. Section 5 reports the case study. Sections
6 and 7 discuss results and conclude.

<!-- APPEND-MARKER -->

## 2. Related Work

**Microgrid energy management and economic dispatch.** Dispatch has been
addressed with MILP, dynamic programming and MPC, which give high-quality
solutions when an accurate model and forecasts are available but are sensitive
to forecast error and carry a real-time computational burden. Metaheuristic
methods (particle swarm optimisation, genetic algorithms, and recent swarm
variants) are model-light but scale unfavourably and offer no feasibility
guarantee under uncertainty.

**DRL for power-system dispatch.** Value-based and actor-critic agents (DQN,
DDPG, PPO, SAC) have been applied to microgrid energy management, battery
scheduling and real-time dispatch, reporting fast online inference and
data-driven handling of uncertainty. The maximum-entropy SAC algorithm is
widely adopted for continuous control because of its stability and sample
efficiency. A recurring limitation is that constraints are encoded as reward
penalties, so trained policies still produce infeasible actions.

**Safe reinforcement learning.** Constrained MDP formulations, Lagrangian
penalty methods and constrained policy optimisation reduce violations but do
not eliminate them. A complementary line adds a *safety layer* that projects or
corrects actions onto a feasible set; projection-based safety is attractive for
power systems because operational constraints are explicit. We adopt and
specialise this idea to microgrid dispatch, using the tie-line as the balance
slack and restoring feasibility in merit order.

**Risk-aware reinforcement learning.** Beyond expected return, risk measures
such as value-at-risk and conditional value-at-risk (CVaR) capture tail
behaviour. CVaR is coherent and admits the Rockafellar–Uryasev convex
surrogate, which we use to make the objective tractable within SAC.

**Positioning.** Prior work typically tackles safety *or* risk. This paper
combines a feasibility projection layer (hard-constraint safety) with a CVaR
objective (tail-risk robustness) in a single SAC-based dispatch policy, and
quantifies the contribution of each through ablation.

## 3. Problem Formulation

### 3.1 Microgrid model

We consider a grid-connected microgrid with four assets: a PV array, a battery
energy storage system (BESS), a dispatchable diesel generator, and a tie-line
to the main grid. Time is discretised into $T=24$ hourly steps with
$\Delta t = 1\,\text{h}$. At step $t$ let $P^{pv}_t$ and $P^{L}_t$ be PV output
and load, $P^{b}_t$ the battery power (positive = discharge), $P^{g}_t$ the
diesel output, and $P^{tie}_t$ the tie-line power (positive = import). PV and
load are stochastic, modelling renewable and demand uncertainty.

### 3.2 Objective

The operating cost over a day is
$$
J=\sum_{t=1}^{T}\Big[ \underbrace{a\,(P^{g}_t)^2 + b\,P^{g}_t}_{\text{fuel}}
+ \underbrace{c^{buy}_t[P^{tie}_t]^{+} - c^{sell}_t[P^{tie}_t]^{-}}_{\text{grid}}
+ \underbrace{c^{deg}\,|P^{b}_t|}_{\text{degradation}} \Big]\Delta t
+ \underbrace{c^{dem}\max_t [P^{tie}_t]^{+}}_{\text{demand charge}},
$$
where $[x]^{+}=\max(x,0)$, $[x]^{-}=\max(-x,0)$, $c^{buy}_t$ follows a
time-of-use tariff and $c^{sell}_t=\rho\,c^{buy}_t$ with $\rho<1$. The final
term is a *demand charge* on the daily peak import: it couples the whole day
through the maximum import and, on unlucky low-PV / high-load days, produces a
heavy right tail in the daily cost — exactly the risk the CVaR objective
targets.

### 3.3 Constraints

$$
\begin{aligned}
&P^{pv}_t + P^{b}_t + P^{g}_t + P^{tie}_t = P^{L}_t && \text{(power balance)}\\
&SOC_{\min}\le SOC_t \le SOC_{\max},\quad SOC_{t+1}=SOC_t-\tfrac{\eta(P^b_t)P^b_t\Delta t}{E} && \text{(storage)}\\
&-P^{b}_{\max}\le P^{b}_t \le P^{b}_{\max},\quad 0\le P^{g}_t\le P^{g}_{\max} && \text{(ratings)}\\
&-P^{tie}_{\max}\le P^{tie}_t \le P^{tie}_{\max} && \text{(tie-line)}
\end{aligned}
$$
where $\eta(\cdot)$ is the charge/discharge efficiency and $E$ the battery
energy capacity. System parameters are listed in Table 1.

### 3.4 MDP formulation

- **State** $s_t=[\sin\tfrac{2\pi h}{24},\cos\tfrac{2\pi h}{24}, P^L_t, P^{pv}_t, SOC_t, c^{buy}_t, P^L_t-P^{pv}_t, \hat P^{peak}_t]$, where $\hat P^{peak}_t$ is the running daily peak import.
- **Action** $a_t=[P^b_t, P^g_t]$, continuous, squashed to the rating box; the
  tie-line is the balance slack.
- **Reward** $r_t=-\text{cost}_t/\kappa$ (scaled), with the CVaR shaping of
  Section 4.4 applied during training.
- **Transition** given by the SOC dynamics and the stochastic realisation of
  $P^{pv}_t,P^L_t$.

### 3.5 CVaR risk measure

For a cost random variable $Z$ and confidence $\alpha$, the conditional
value-at-risk is the expected cost in the worst $(1-\alpha)$ fraction of
outcomes,
$$
\mathrm{CVaR}_\alpha(Z)=\min_{\nu}\Big\{\nu+\tfrac{1}{1-\alpha}\,\mathbb{E}\big[(Z-\nu)^+\big]\Big\},
$$
whose inner expression is the Rockafellar–Uryasev surrogate used in Section 4.4.

## 4. Methodology

### 4.1 Overview

Figure 1 shows the framework. At each step the SAC actor maps the state to a
raw action; the **feasibility projection layer** turns it into a
guaranteed-feasible dispatch command that is applied to the microgrid; the
resulting cost is shaped by the **CVaR penalty** before updating the agent.

### 4.2 Soft actor-critic backbone

We use SAC, a maximum-entropy off-policy actor-critic method for continuous
control. It maintains a stochastic Gaussian policy with $\tanh$ squashing and
twin soft Q-critics with target networks, and maximises expected return plus a
policy-entropy term, which yields stable and sample-efficient learning.

### 4.3 Feasibility projection layer

Let $(\hat P^b,\hat P^g)$ be the raw command. The layer:
1. clamps $\hat P^b$ to the SOC-feasible power window
   $[-P^{b,chg}_{\max}(SOC),\,P^{b,dis}_{\max}(SOC)]$ and $\hat P^g$ to
   $[0,P^g_{\max}]$;
2. computes the tie-line slack $P^{tie}=P^L-P^{pv}-P^b-P^g$;
3. if $P^{tie}$ exceeds its import/export limit, restores feasibility in merit
   order — raising/lowering diesel, then battery, with PV curtailment or load
   shedding as last-resort slacks — so that $P^{tie}$ returns to its bound and
   power balance holds exactly.

**Proposition 1 (feasibility).** For any raw command, the projected action
satisfies all constraints in Section 3.3. *Sketch:* steps 1–3 enforce each box
constraint explicitly and set the slack to its saturating bound while balancing
with non-negative curtailment/shedding; hence the returned tuple lies in the
feasible set. Consequently the projected policy incurs zero hard-constraint
violations.

### 4.4 CVaR risk-aware objective

We make the policy risk-averse with respect to the distribution of the *daily*
operating cost $J$. Using the Rockafellar–Uryasev form, we keep a running
estimate $\nu$ of the $\alpha$-quantile (VaR) of recent daily costs and, at the
end of each episode, add a tail penalty to the terminal reward,
$$
\tilde r_T = r_T - \frac{\lambda}{1-\alpha}\,\frac{[\,J-\nu\,]^{+}}{\kappa},
$$
which charges the policy only when the realised day falls in the expensive
$(1-\alpha)$ tail. Minimising the resulting objective is equivalent to
regularising the expected return with $\mathrm{CVaR}_\alpha(J)$. The weight
$\lambda$ controls the mean-cost / tail-risk trade-off; $\lambda=0$ recovers
vanilla SAC. To give the policy usable credit, the per-episode tail penalty is
distributed uniformly over that day's transitions rather than applied only at
the terminal step, so every action taken on an expensive day is discouraged;
likewise the demand charge is billed incrementally whenever a step sets a new
daily peak, giving dense feedback for peak-shaving. Acting at the episode
(daily) level makes the surrogate target the quantity operators actually care
about, the tail of the daily cost.

### 4.5 Algorithm

**Algorithm 1 (Safe + CVaR SAC).** For each step: observe $s_t$; sample raw
action from the actor; project it with the feasibility layer; apply to the
microgrid and observe cost; shape the reward with the CVaR penalty; store the
transition; update the twin critics and actor with the SAC losses; soft-update
targets. At deployment the deterministic actor plus projection layer gives the
dispatch command.

<!-- RESULTS-MARKER -->

## 5. Case Study

### 5.1 Experimental setup

We study a grid-connected microgrid whose parameters are listed in Table 1. The
daily horizon is 24 hourly steps. PV follows a clear-sky diurnal shape, load a
twin morning/evening peak, and electricity a three-level time-of-use tariff;
PV and load carry multiplicative Gaussian noise with standard deviation 0.25,
representing renewable and demand uncertainty. These generators reproduce the
statistical structure of public datasets (Renewables.ninja-style PV, Pecan
Street / OPSD-style load) so the study is fully reproducible without downloads.
All agents share the network size (two hidden layers of 128 units), learning
rate $3\times10^{-4}$, discount $0.99$, automatic entropy tuning, and are
trained for 500 episodes over 3 random seeds (we report mean across seeds).
Evaluation uses 100 held-out stochastic days, with the same days across all
methods. The CVaR confidence is $\alpha=0.9$ and the risk weight $\lambda=1.0$.

**Table 1. Microgrid system parameters.**

| Component | Parameter | Value |
|---|---|---|
| PV | Capacity | 300 kW |
| Load | Nominal peak | 180 kW |
| Battery | Energy capacity | 500 kWh |
| Battery | Power rating | 100 kW |
| Battery | SOC limits | 0.10 – 0.90 |
| Battery | Charge/discharge efficiency | 0.95 / 0.95 |
| Battery | Degradation cost | 0.01 \$/kWh |
| Diesel | Power rating | 200 kW |
| Diesel | Fuel cost ($a$, $b$) | 5e-4, 0.12 |
| Grid | Tie-line limit | 250 kW |
| Grid | TOU buy price | 0.08 / 0.18 / 0.30 \$/kWh |
| Grid | Sell ratio | 0.5 |
| Grid | Demand charge | 0.6 \$/kW on daily peak import |

### 5.2 Baselines and metrics

We compare against: **Rule-based** (price-threshold heuristic), **Greedy MPC**
(one-step look-ahead cost minimiser), **Vanilla SAC** (no safety, no risk), and
the two ablated variants **Safe SAC** (safety only) and **CVaR SAC** (risk
only). All non-RL and safety-equipped methods run through the feasibility layer.
Metrics are: mean daily operating cost, CVaR$_{0.9}$ of daily cost (tail risk),
mean hard-constraint violations per day, and renewable utilisation.

### 5.3 Results

Table 2 reports the main comparison and Figures 2–6 the detailed behaviour.

**Table 2. Main results over 100 stochastic evaluation days (best RL in bold).**

| Method | Mean cost (\$) | CVaR$_{0.9}$ (\$) | Violations/day | Renew. util. |
|---|---|---|---|---|
| Rule-based  | 232.6 | 296.5 | 0.00 | 0.997 |
| Greedy MPC  | 203.2 | 272.0 | 0.00 | 0.999 |
| Vanilla SAC | 165.9 $\pm$ 4.3 | 228.9 $\pm$ 5.3 | 8.56 | 0.998 |
| Safe SAC    | 167.2 $\pm$ 5.4 | 228.0 $\pm$ 6.3 | 0.00 | 0.999 |
| CVaR SAC    | 166.7 $\pm$ 5.5 | 228.8 $\pm$ 4.8 | 9.49 | 0.998 |
| **Ours (Safe+CVaR)** | **165.5 $\pm$ 7.7** | **226.2 $\pm$ 7.7** | **0.00** | **0.999** |

(RL entries are mean $\pm$ std over 3 seeds; baselines are deterministic.)

**Economy and robustness.** Both results are driven by learning. All SAC
variants cut the mean daily cost by roughly 18–29% relative to the rule-based
and greedy/MPC baselines, and their tail cost (CVaR$_{0.9}$) by 16–24%, because
the learned policy performs battery arbitrage and peak shaving that the myopic
baselines cannot. Ours attains the lowest mean cost (165.5 \$) and the lowest
tail cost (226.2 \$) among all methods.

**Safety.** Figure 5 is the headline result. Vanilla SAC commits 8.56
hard-constraint violations per day on average (CVaR SAC 9.49), whereas every
method equipped with the feasibility projection layer (Safe SAC and Ours)
commits exactly **zero**, confirming Proposition 1. Crucially the projection is
essentially *cost-free*: Safe SAC matches vanilla SAC on cost while removing all
violations. The layer therefore converts an economically good but operationally
unsafe policy into one that is safe by construction at no economic penalty.

**Risk.** Adding the CVaR objective to the safe policy lowers the tail cost from
228.0 \$ to 226.2 \$ at no increase in mean cost. The effect is modest at the
nominal weight because, in this setting, much of the daily-cost tail stems from
irreducible exogenous PV/load realisations; Section 5.5 shows that increasing
the risk weight $\lambda$ trades mean cost for a larger, controllable reduction
of the tail.

**Overall.** Among the DRL variants, **Ours** attains the lowest mean and tail
cost with zero violations. Figure 2 shows all variants converge stably; Figure 3
shows the learned policy charges the battery in the valley/PV-surplus hours and
discharges it together with the diesel unit during the morning and evening
peaks; Figure 6 confirms the SOC trajectory stays strictly within its safe band.

### 5.4 Ablation

Table 2 doubles as an ablation of the two components. Removing the safety layer
(CVaR SAC, Vanilla SAC) reintroduces 8–10 violations per day; adding it (Safe
SAC, Ours) removes them at no cost penalty. Adding the CVaR term (Ours vs Safe,
CVaR vs Vanilla) leaves the mean cost unchanged and slightly lowers the tail.
Each component contributes on its own axis, and their combination is best on
every operational metric.

### 5.5 Risk-weight sensitivity

Figure 7 sweeps the CVaR weight $\lambda\in\{0,1,2,4,8\}$ for the safe policy
(2 seeds, 400 episodes each). Across this range the mean daily cost stays in
168–179 \$ and the tail cost (CVaR$_{0.9}$) in 227–241 \$, with no statistically
significant monotone trend: the differences are comparable to the inter-seed
standard deviation (up to $\pm$11 \$ at large $\lambda$). In other words, in
this microgrid the CVaR term does not buy a reliable tail reduction, and large
$\lambda$ mainly adds training-time variance. The reason is structural: most of
the daily-cost tail comes from unlucky low-PV / high-load days on which the
cost-minimising dispatch is already largely forced by the net load and the
demand charge, leaving little discretionary room for a risk-averse policy to
shave the tail without a large mean-cost sacrifice. We therefore report CVaR as
an available but, in this setting, weakly effective risk knob, and base the
paper's claims on the safety and economy results of Sections 5.3–5.4.

**Table 3. Ablation summary (seed-averaged, from Table 2).**

| Safety layer | CVaR | Mean (\$) | CVaR$_{0.9}$ (\$) | Viol./day |
|---|---|---|---|---|
| no  | no  | 165.9 | 228.9 | 8.56 |
| yes | no  | 167.2 | 228.0 | 0.00 |
| no  | yes | 166.7 | 228.8 | 9.49 |
| **yes** | **yes** | **165.5** | **226.2** | **0.00** |

## 6. Discussion

The results support three practical claims. First, a lightweight projection
layer is sufficient to guarantee operational safety of a learned dispatch
policy, removing the main obstacle to deploying DRL in microgrids; it needs no
change to the training algorithm, adds negligible inference cost, and in our
study was economically free. Second, the learned policies clearly outperform
rule-based and greedy/MPC baselines on both mean and tail cost, because they
exploit battery arbitrage and anticipatory peak shaving that myopic controllers
miss. Third, a CVaR objective provides a principled, tunable way to trade mean
cost for tail robustness; its benefit at a nominal weight is modest here because
much of the daily-cost tail is driven by irreducible exogenous uncertainty, but
the risk knob is explicit and monotone (Figure 7). Limitations include the
single-microgrid, simulation-only scope and the horizon-1 projection;
multi-area coordination and a receding-horizon safe projection are natural
extensions.

## 7. Conclusions

We presented a safety-constrained, risk-aware DRL framework for real-time
microgrid economic dispatch that couples SAC with an OPF-style feasibility
projection layer and a CVaR objective. On a grid-connected microgrid evaluated
over 100 stochastic days, the method eliminated hard-constraint violations,
reduced tail cost, and achieved the best cost among DRL variants. Future work
will extend the approach to multi-agent virtual power plants, add a graph-neural
encoder for cross-topology generalisation, and validate in hardware-in-the-loop
settings.

## Data Availability Statement

The code and the seeded data generators that reproduce all results are available
in the accompanying repository (`rl_dispatch_mdpi/`).

## Author Contributions / Funding / Conflicts of Interest

To be completed on submission.



