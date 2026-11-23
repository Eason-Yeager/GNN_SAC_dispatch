"""OPF-style feasibility projection layer.

Given the raw (battery, diesel) command produced by the RL policy, this layer
projects it onto the microgrid feasible set so that every hard constraint is
satisfied by construction:

    * battery power within the SOC-feasible range and rating,
    * diesel generator within [0, p_max],
    * tie-line power within +/- p_grid_max  (grid is the balance slack),
    * instantaneous power balance.

Resources are adjusted in a merit order (cheap/flexible first) to restore
feasibility with minimal deviation from the commanded action. Load shedding /
PV curtailment are last-resort slacks and are ~0 for a well-sized system.

Proposition. For any command the returned action lies in the feasible set,
hence the projected policy incurs zero hard-constraint violations.
"""


def project(p_batt_cmd, p_diesel_cmd, load, pv, p_dis_max, p_chg_max, cfg):
    # 1) Clamp to SOC-feasible battery range and diesel box.
    p_batt = float(min(max(p_batt_cmd, -p_chg_max), p_dis_max))
    p_diesel = float(min(max(p_diesel_cmd, 0.0), cfg.p_diesel_max))
    curtail = 0.0
    shed = 0.0

    # 2) Tie-line (slack) power; restore its limits via merit-order adjustment.
    grid = load - pv - p_batt - p_diesel
    if grid > cfg.p_grid_max:
        # Import exceeds limit -> raise local generation.
        deficit = grid - cfg.p_grid_max
        inc = min(deficit, cfg.p_diesel_max - p_diesel)      # diesel up
        p_diesel += inc
        deficit -= inc
        inc = min(deficit, p_dis_max - p_batt)               # battery discharge up
        p_batt += inc
        deficit -= inc
        shed = max(0.0, deficit)                             # last resort
        grid = cfg.p_grid_max
    elif grid < -cfg.p_grid_max:
        # Export exceeds limit -> cut generation / absorb.
        surplus = -grid - cfg.p_grid_max
        dec = min(surplus, p_diesel)                         # diesel down
        p_diesel -= dec
        surplus -= dec
        dec = min(surplus, p_batt + p_chg_max)               # charge more
        p_batt -= dec
        surplus -= dec
        curtail = max(0.0, surplus)                          # last resort
        grid = -cfg.p_grid_max

    return p_batt, p_diesel, float(grid), curtail, shed
