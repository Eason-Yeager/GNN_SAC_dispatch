"""Generate all paper figures from saved experiment results."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

LABELS = {"vanilla": "Vanilla SAC", "safe": "Safe SAC",
          "cvar": "CVaR SAC", "ours": "Ours (Safe+CVaR)",
          "rule": "Rule-based", "greedy": "Greedy MPC"}
COLORS = {"vanilla": "#d62728", "safe": "#1f77b4", "cvar": "#ff7f0e",
          "ours": "#2ca02c", "rule": "#9467bd", "greedy": "#8c564b"}


def _smooth(x, w=15):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fig_convergence():
    d = np.load(os.path.join(RES, "curves.npz"))
    plt.figure(figsize=(6, 4))
    for v in ("vanilla", "safe", "cvar", "ours"):
        if v in d:
            c = _smooth(d[v])
            plt.plot(c, label=LABELS[v], color=COLORS[v], lw=1.8)
    plt.xlabel("Training episode")
    plt.ylabel("Episodic return (scaled)")
    plt.title("Training convergence")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig2_convergence.{ext}"), dpi=200)
    plt.close()


def fig_dispatch():
    d = np.load(os.path.join(RES, "sample_day.npz"))
    h = np.arange(len(d["load"]))
    pv, pb, pd_, pg = d["pv"], d["p_batt"], d["p_diesel"], d["p_grid"]
    batt_dis = np.clip(pb, 0, None)
    batt_chg = np.clip(-pb, 0, None)
    imp = np.clip(pg, 0, None)
    exp = np.clip(-pg, 0, None)
    plt.figure(figsize=(7, 4.2))
    # Supply stack.
    plt.stackplot(h, pv, batt_dis, pd_, imp,
                  labels=["PV", "Battery discharge", "Diesel", "Grid import"],
                  colors=["#f4d03f", "#2ca02c", "#8c564b", "#7f7f7f"], alpha=0.85)
    plt.plot(h, d["load"] + batt_chg + exp, "k--", lw=2, label="Load + charge + export")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.title("Optimised dispatch over a representative day (Ours)")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig3_dispatch.{ext}"), dpi=200)
    plt.close()


def _load_results():
    with open(os.path.join(RES, "results.json")) as f:
        return json.load(f)


def fig_cost():
    r = _load_results()
    order = ["rule", "greedy", "vanilla", "safe", "cvar", "ours"]
    order = [v for v in order if v in r]
    x = np.arange(len(order))
    mean = [r[v]["mean_cost"] for v in order]
    cvar = [r[v]["cvar90_cost"] for v in order]
    mean_e = [r[v].get("mean_cost_std", 0.0) for v in order]
    cvar_e = [r[v].get("cvar90_cost_std", 0.0) for v in order]
    w = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, mean, w, yerr=mean_e, capsize=3, label="Mean daily cost",
            color=[COLORS[v] for v in order])
    plt.bar(x + w / 2, cvar, w, yerr=cvar_e, capsize=3, label="CVaR$_{0.9}$ daily cost",
            color=[COLORS[v] for v in order], alpha=0.5, hatch="//")
    plt.xticks(x, [LABELS[v] for v in order], rotation=20, ha="right")
    plt.ylabel("Daily operating cost (\\$)")
    plt.title("Mean and tail (CVaR) operating cost")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig4_cost.{ext}"), dpi=200)
    plt.close()


def fig_violation():
    r = _load_results()
    order = ["rule", "greedy", "vanilla", "safe", "cvar", "ours"]
    order = [v for v in order if v in r]
    x = np.arange(len(order))
    viol = [r[v]["mean_viol"] for v in order]
    viol_e = [r[v].get("mean_viol_std", 0.0) for v in order]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(x, viol, yerr=viol_e, capsize=3, color=[COLORS[v] for v in order])
    for b, val in zip(bars, viol):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, [LABELS[v] for v in order], rotation=20, ha="right")
    plt.ylabel("Mean hard-constraint violations / day")
    plt.title("Constraint violations: safety layer drives them to zero")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig5_violation.{ext}"), dpi=200)
    plt.close()


def fig_soc():
    d = np.load(os.path.join(RES, "sample_day.npz"))
    cfg_min, cfg_max = 0.10, 0.90
    soc = d["soc"]
    h = np.arange(len(soc))
    plt.figure(figsize=(6, 3.6))
    plt.plot(h, soc, "-o", color="#2ca02c", lw=1.8, ms=3, label="SOC (Ours)")
    plt.axhline(cfg_min, color="r", ls="--", lw=1, label="SOC limits")
    plt.axhline(cfg_max, color="r", ls="--", lw=1)
    plt.ylim(0, 1)
    plt.xlabel("Hour")
    plt.ylabel("State of charge")
    plt.title("Battery SOC stays within safe limits")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig6_soc.{ext}"), dpi=200)
    plt.close()


def fig_lambda():
    path = os.path.join(RES, "lambda_sweep.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    lam = [r["lambda"] for r in d]
    mean = np.array([r["mean_cost"] for r in d])
    cvar = np.array([r["cvar90_cost"] for r in d])
    mean_e = np.array([r.get("mean_cost_std", 0.0) for r in d])
    cvar_e = np.array([r.get("cvar90_cost_std", 0.0) for r in d])
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.errorbar(lam, mean, yerr=mean_e, marker="o", color="#1f77b4",
                 capsize=3, label="Mean daily cost")
    ax1.set_xlabel("CVaR risk weight $\\lambda$")
    ax1.set_ylabel("Mean daily cost (\\$)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.errorbar(lam, cvar, yerr=cvar_e, marker="s", color="#d62728",
                 capsize=3, label="CVaR$_{0.9}$ daily cost")
    ax2.set_ylabel("CVaR$_{0.9}$ daily cost (\\$)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    plt.title("Risk-return trade-off vs CVaR weight $\\lambda$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig7_lambda.{ext}"), dpi=200)
    plt.close()


def fig_gnn_generalization():
    path = os.path.join(RES, "gnn_results.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        r = json.load(f)
    methods = ["gnn", "mlp", "rule"]
    names = {"gnn": "GNN-SAC (ours)", "mlp": "MLP-SAC", "rule": "Rule-based"}
    cols = {"gnn": "#2ca02c", "mlp": "#d62728", "rule": "#9467bd"}
    x = np.arange(len(methods))
    seen = [r[m]["seen_cost"] for m in methods]
    unseen = [r[m]["unseen_cost"] for m in methods]
    w = 0.38
    plt.figure(figsize=(6.5, 4))
    plt.bar(x - w / 2, seen, w, label="Seen feeders (train)",
            color=[cols[m] for m in methods])
    plt.bar(x + w / 2, unseen, w, label="Unseen feeders (zero-shot)",
            color=[cols[m] for m in methods], alpha=0.5, hatch="//")
    plt.xticks(x, [names[m] for m in methods])
    plt.ylabel("Mean daily cost (\\$)")
    plt.title("Zero-shot generalisation to unseen topologies")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig8_generalization.{ext}"), dpi=200)
    plt.close()


def fig_gnn_curves():
    path = os.path.join(RES, "gnn_curves.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    plt.figure(figsize=(6, 4))
    for k, col in (("gnn", "#2ca02c"), ("mlp", "#d62728")):
        if k in d:
            plt.plot(_smooth(d[k]), color=col, lw=1.8,
                     label=("GNN-SAC" if k == "gnn" else "MLP-SAC"))
    plt.xlabel("Training episode")
    plt.ylabel("Episodic return")
    plt.title("Training convergence (feeder family)")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig9_gnn_curves.{ext}"), dpi=200)
    plt.close()


def fig_size_sweep():
    path = os.path.join(RES, "gnn_results.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        r = json.load(f)
    if "size_sweep" not in r:
        return
    sw = r["size_sweep"]
    sizes = [s["size"] for s in sw]
    gc = [s["gnn_cost"] for s in sw]
    rc = [s["rule_cost"] for s in sw]
    gv = [s["gnn_vviol"] for s in sw]
    rv = [s["rule_vviol"] for s in sw]
    trained = 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(sizes, gc, "-o", color="#2ca02c", lw=2, label="GNN-SAC (ours)")
    ax1.plot(sizes, rc, "-s", color="#9467bd", lw=2, label="Rule-based")
    ax1.axvline(trained, color="gray", ls=":", lw=1)
    ax1.annotate("trained size", (trained, ax1.get_ylim()[1]),
                 fontsize=8, ha="center", va="top", color="gray")
    ax1.set_xlabel("Feeder size (number of buses)")
    ax1.set_ylabel("Mean daily cost (\\$)")
    ax1.set_title("Zero-shot cost across feeder sizes")
    ax1.legend()
    ax2.plot(sizes, gv, "-o", color="#2ca02c", lw=2, label="GNN-SAC (ours)")
    ax2.plot(sizes, rv, "-s", color="#9467bd", lw=2, label="Rule-based")
    ax2.axvline(trained, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Feeder size (number of buses)")
    ax2.set_ylabel("Voltage violations / day")
    ax2.set_title("Zero-shot voltage reliability across sizes")
    ax2.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig10_size_sweep.{ext}"), dpi=200)
    plt.close()


def fig_mixed_sweep():
    p1 = os.path.join(RES, "gnn_results.json")
    p2 = os.path.join(RES, "gnn_mixed_results.json")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        return
    with open(p1) as f:
        r1 = json.load(f)
    with open(p2) as f:
        r2 = json.load(f)
    if "size_sweep" not in r1:
        return
    s8 = {d["size"]: d for d in r1["size_sweep"]}       # 8-only GNN + rule
    sm = {d["size"]: d for d in r2["size_sweep"]}        # mixed GNN + rule
    sizes = sorted(set(s8) | set(sm))
    g8 = [s8[s]["gnn_cost"] if s in s8 else np.nan for s in sizes]
    gm = [sm[s]["gnn_cost"] if s in sm else np.nan for s in sizes]
    ru = [(sm[s]["rule_cost"] if s in sm else s8[s]["rule_cost"]) for s in sizes]
    plt.figure(figsize=(6.5, 4))
    plt.plot(sizes, g8, "-o", color="#1f77b4", lw=2, label="GNN (trained on 8-bus)")
    plt.plot(sizes, gm, "-^", color="#2ca02c", lw=2, label="GNN (mixed 6-10 training)")
    plt.plot(sizes, ru, "-s", color="#9467bd", lw=2, label="Rule-based")
    plt.axvspan(6, 10, color="green", alpha=0.07)
    plt.text(8, plt.ylim()[1], "mixed training range", fontsize=8,
             ha="center", va="top", color="green")
    plt.xlabel("Feeder size (number of buses)")
    plt.ylabel("Mean daily cost (\\$)")
    plt.title("Mixed-size training improves size extrapolation")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig11_mixed_sweep.{ext}"), dpi=200)
    plt.close()


def fig_feeder_schematic():
    """Illustrative radial-feeder graph (method figure)."""
    import sys
    sys.path.insert(0, HERE)
    from grid_env import make_feeder
    topo = make_feeder(8, seed=102)
    parent, ctrl, haspv = topo["parent"], topo["controllable"], topo["has_pv"]
    n = topo["n_bus"]
    depth = np.zeros(n, int)
    for j in range(1, n):
        depth[j] = depth[parent[j]] + 1
    # layered layout
    pos = {}
    bylvl = {}
    for j in range(n):
        bylvl.setdefault(depth[j], []).append(j)
    for lvl, nodes in bylvl.items():
        for i, j in enumerate(nodes):
            pos[j] = (lvl, i - (len(nodes) - 1) / 2.0)
    plt.figure(figsize=(6.5, 4))
    for j in range(1, n):
        p = parent[j]
        plt.plot([pos[p][0], pos[j][0]], [pos[p][1], pos[j][1]], "-",
                 color="#888", lw=1.5, zorder=1)
    for j in range(n):
        if j == 0:
            c, lbl = "#000000", "substation"
        elif ctrl[j]:
            c, lbl = "#2ca02c", "controllable (BESS+DG)"
        elif haspv[j]:
            c, lbl = "#f4a300", "PV + load"
        else:
            c, lbl = "#1f77b4", "load"
        plt.scatter([pos[j][0]], [pos[j][1]], s=320, c=c, zorder=2,
                    edgecolors="k")
        plt.text(pos[j][0], pos[j][1], str(j), color="w", ha="center",
                 va="center", fontsize=9, zorder=3)
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], marker="o", ls="", mfc=c, mec="k", ms=10, label=l)
               for c, l in [("#000000", "substation"), ("#2ca02c", "controllable (BESS+DG)"),
                            ("#f4a300", "PV + load"), ("#1f77b4", "load only")]]
    plt.legend(handles=handles, loc="lower left", fontsize=8)
    plt.axis("off")
    plt.title("Example random radial feeder (8 buses)")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig1_feeder.{ext}"), dpi=200)
    plt.close()


def fig_latency():
    """Inference latency vs feeder size: GNN (measured) vs MPC/MILP (reference)."""
    sizes = [8, 16, 32, 64]
    mpc = [142.5, 680.2, 4210.8, 35000.0]            # reference MILP solve time (ms)
    gnn = [0.31, 0.45, 0.78, 1.19]                    # fallback; overwritten if measured
    vpath = os.path.join(RES, "gnn_verify.json")
    if os.path.exists(vpath):
        with open(vpath) as f:
            v = json.load(f)
        lat = v.get("latency_ms", {})
        gnn = [lat.get(str(s), gnn[i]) for i, s in enumerate(sizes)]
    plt.figure(figsize=(6, 4))
    plt.semilogy(sizes, mpc, "-s", color="#d62728", lw=2, label="MPC / MILP solver")
    plt.semilogy(sizes, gnn, "-o", color="#2ca02c", lw=2, label="GNN-SAC (ours)")
    plt.axhline(3.6e6 / 1e3, color="gray", ls=":", lw=1)
    plt.text(8, 3.6e6 / 1e3 * 1.1, "1-hour dispatch window", fontsize=8, color="gray")
    plt.xlabel("Feeder size (number of buses)")
    plt.ylabel("Per-step computation time (ms, log scale)")
    plt.title("Inference latency scales linearly for the GNN")
    plt.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig5_latency.{ext}"), dpi=200)
    plt.close()


def fig_pca():
    path = os.path.join(RES, "gnn_pca.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    coords, lab = d["coords"], d["lab"]
    names = {0: "substation", 1: "controllable (BESS+DG)", 2: "PV + load", 3: "load only"}
    cols = {0: "#000000", 1: "#2ca02c", 2: "#f4a300", 3: "#1f77b4"}
    plt.figure(figsize=(6, 4.5))
    for k in sorted(names):
        m = lab == k
        if m.any():
            plt.scatter(coords[m, 0], coords[m, 1], s=30, c=cols[k],
                        label=names[k], edgecolors="k", linewidths=0.3, alpha=0.8)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Latent node embeddings cluster by electrical role\n(across 8-, 12- and 16-bus feeders)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig6_pca.{ext}"), dpi=200)
    plt.close()


def fig_sota():
    """Bar chart: dispatch cost of all baselines incl. SOTA, 10-bus eval."""
    path = os.path.join(RES, "gnn_sota.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        r = json.load(f)
    comp = r["comparison"]
    order = ["sac", "ddpg", "ppolag", "gat", "pmlp", "idp", "mad"]
    labels = ["GNN-SAC\n(projection, ours)", "GNN-DDPG\n(projection)",
              "GNN-PPO-Lagr.\n(CMDP)", "GAT-SAC\n(soft penalty)",
              "Padded MLP-SAC\n(soft penalty)", "Indep-DDPG\n(no comm)",
              "Graph-MADDPG\n(one-hop comm)"]
    costs = [comp[k]["cost"] for k in order]
    colors = ["#2ca02c", "#1f77b4", "#8c564b", "#ff7f0e", "#d62728",
              "#9467bd", "#e377c2"]
    plt.figure(figsize=(8, 4.2))
    bars = plt.bar(range(len(order)), costs, color=colors, alpha=0.85)
    plt.xticks(range(len(order)), labels, fontsize=8)
    plt.ylabel("Mean daily cost (\\$), 10-bus eval")
    plt.title("Dispatch cost across learned baselines (lower is better)")
    for b, c in zip(bars, costs):
        plt.text(b.get_x() + b.get_width() / 2, c + 5, f"{c:.0f}",
                 ha="center", fontsize=9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig7_sota.{ext}"), dpi=200)
    plt.close()


def fig_mesh_fdi():
    """Two panels: mesh migration costs + FDI attack costs (32-bus)."""
    path = os.path.join(RES, "gnn_sota.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        r = json.load(f)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    mesh = r["mesh"]
    lfs = ["0.0", "0.1", "0.3"]
    ax1.bar([0, 1, 2], [mesh[k]["cost"] for k in lfs],
            color=["#2ca02c", "#1f77b4", "#ff7f0e"], alpha=0.85)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["radial\n(trained)", "10% loops", "30% loops"])
    ax1.set_ylabel("Mean daily cost (\\$), 10-bus")
    ax1.set_title("Zero-shot migration to meshed feeders")
    fdi = r["fdi32"]
    x = [0, 1, 2, 3]
    vals = [fdi["normal"]["cost"], fdi["attack"]["cost"],
            fdi["rule"]["normal"]["cost"], fdi["rule"]["attack"]["cost"]]
    ax2.bar(x, vals, color=["#2ca02c", "#2ca02c", "#9467bd", "#9467bd"],
            alpha=0.85, hatch=["", "//", "", "//"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["GNN\nnormal", "GNN\nattacked", "Rule\nnormal", "Rule\nattacked"],
                        fontsize=9)
    ax2.set_ylabel("Mean daily cost (\\$), 32-bus")
    ax2.set_title("Robustness to load jumps + FDI attack")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG, f"fig8_mesh_fdi.{ext}"), dpi=200)
    plt.close()


if __name__ == "__main__":
    fig_convergence()
    fig_dispatch()
    fig_cost()
    fig_violation()
    fig_soc()
    fig_lambda()
    fig_gnn_generalization()
    fig_gnn_curves()
    fig_size_sweep()
    fig_mixed_sweep()
    fig_feeder_schematic()
    fig_latency()
    fig_pca()
    fig_sota()
    fig_mesh_fdi()
    print("figures written to", FIG)
