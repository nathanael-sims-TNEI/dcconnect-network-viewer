"""
Consistency checks, dispatch and load flows for a pandapower network built by ``pandapower_network_model_from_file.py``.
"""

from __future__ import annotations
import copy
import logging
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top

LOG = logging.getLogger(__name__)


# Columns that must be populated for the power flow to be well defined
REQUIRED_COLUMNS = {
    "line": ["from_bus", "to_bus", "length_km", "r_ohm_per_km", "x_ohm_per_km",
             "c_nf_per_km", "max_i_ka"],
    "trafo": ["hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent",
              "vkr_percent", "pfe_kw", "i0_percent", "shift_degree"],
    "trafo3w": ["hv_bus", "mv_bus", "lv_bus", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                "sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vk_hv_percent", "vk_mv_percent",
                "vk_lv_percent", "vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent",
                "pfe_kw", "i0_percent"],
    "impedance": ["from_bus", "to_bus", "rft_pu", "xft_pu", "sn_mva"],
    "load": ["bus", "p_mw", "q_mvar"],
    "gen": ["bus", "p_mw", "vm_pu"],
    "sgen": ["bus", "p_mw", "q_mvar"],
    "shunt": ["bus", "p_mw", "q_mvar", "vn_kv"],
}


def element_counts(net):
    tables = ["bus", "line", "trafo", "trafo3w", "dcline", "impedance", "switch",
              "load", "gen", "sgen", "shunt", "ext_grid"]
    return {t: len(net[t]) for t in tables}


def parameter_checks(net):
    issues = {}

    if (net.bus.vn_kv <= 0).any():
        issues["buses with non-positive vn_kv"] = int((net.bus.vn_kv <= 0).sum())

    for table, cols in [("line", ["from_bus", "to_bus"]),
                        ("trafo", ["hv_bus", "lv_bus"]),
                        ("trafo3w", ["hv_bus", "mv_bus", "lv_bus"]),
                        ("impedance", ["from_bus", "to_bus"]),
                        ("dcline", ["from_bus", "to_bus"]),
                        ("load", ["bus"]), ("gen", ["bus"]),
                        ("sgen", ["bus"]), ("shunt", ["bus"]), ("ext_grid", ["bus"])]:
        df = net[table]
        for col in cols:
            missing = ~df[col].isin(net.bus.index)
            if missing.any():
                issues[f"{table}.{col} referencing a missing bus"] = int(missing.sum())

    if len(net.line):
        for col, limit in [("x_ohm_per_km", 0), ("length_km", 0), ("max_i_ka", 0)]:
            bad = net.line[col] <= limit
            if bad.any():
                issues[f"lines with {col} <= {limit}"] = int(bad.sum())
        same_bus = net.line.from_bus == net.line.to_bus
        if same_bus.any():
            issues["lines connecting a bus to itself"] = int(same_bus.sum())

    if len(net.trafo):
        bad = net.trafo.vkr_percent >= net.trafo.vk_percent
        if bad.any():
            issues["two-winding trafos with vkr >= vk"] = int(bad.sum())
        if (net.trafo.sn_mva <= 0).any():
            issues["two-winding trafos with sn_mva <= 0"] = int((net.trafo.sn_mva <= 0).sum())

    if len(net.trafo3w):
        for side in ("hv", "mv", "lv"):
            bad = net.trafo3w[f"vkr_{side}_percent"] >= net.trafo3w[f"vk_{side}_percent"]
            if bad.any():
                issues[f"three-winding trafos with vkr_{side} >= vk_{side}"] = int(bad.sum())

    if len(net.impedance):
        bad = (net.impedance.xft_pu.abs() < 1e-9) & (net.impedance.rft_pu.abs() < 1e-9)
        if bad.any():
            issues["impedances with zero R and X"] = int(bad.sum())

    for table, cols in REQUIRED_COLUMNS.items():
        df = net[table]
        nan_cols = [c for c in cols if c in df.columns and df[c].isna().any()]
        if nan_cols:
            issues[f"{table} rows with missing {', '.join(nan_cols)}"] = int(
                df[nan_cols].isna().any(axis=1).sum()
            )

    return issues


def topology_checks(net):
    graph = top.create_nxgraph(net, respect_switches=True)
    islands = list(top.connected_components(graph))
    unsupplied = top.unsupplied_buses(net)
    isolated = [c for c in islands if len(c) == 1]
    return {
        "connected islands": len(islands),
        "largest island (buses)": max((len(c) for c in islands), default=0),
        "single-bus islands": len(isolated),
        "buses not connected to a slack": len(unsupplied),
    }


def power_balance(net):
    return {
        "total demand (MW)": float(net.load.p_mw.sum()),
        "synchronous generation (MW)": float(net.gen.loc[net.gen.in_service, "p_mw"].sum()),
        "static generation (MW)": float(net.sgen.loc[net.sgen.in_service, "p_mw"].sum()),
        "out of service generators": int((~net.gen.in_service).sum()
                                        + (~net.sgen.in_service).sum()),
        "shunt reactive (MVAr, +ve absorbing)": float(net.shunt.q_mvar.sum()),
    }


def capacity_weights(table):
    usable = table["in_service"] & (table["max_p_mw"].fillna(0.0) > 0)
    return table["sn_mva"].where(usable, 0.0).fillna(0.0)


def dispatch_by_capacity(net, share=1.0):
    gen_weights = capacity_weights(net.gen)
    sgen_weights = capacity_weights(net.sgen)
    total = gen_weights.sum() + sgen_weights.sum()

    net.load["scaling"] = share
    net.gen["scaling"] = 1.0
    net.sgen["scaling"] = 1.0
    net.gen["slack_weight"] = gen_weights
    if total <= 0:
        return 0.0

    demand = float(net.load.loc[net.load.in_service, "p_mw"].sum()) * share
    net.gen["p_mw"] = gen_weights / total * demand
    net.sgen["p_mw"] = sgen_weights / total * demand
    return demand


def run_dc_power_flow(net):
    # The same capacity-proportional dispatch as the AC run, at full peak demand, so the
    # two load flows are comparable. The network as built carries every generator at its
    # rated output (about twice the demand), which the slack machine would otherwise have
    # to absorb, and that surplus is what produced widespread DC overloads.
    solved = copy.deepcopy(net)
    demand = dispatch_by_capacity(solved, 1.0)
    try:
        pp.rundcpp(solved)
    except Exception as exc:
        return {"converged": False, "error": f"{type(exc).__name__}: {exc}"}, solved
    slack_mw = float(solved.res_ext_grid.p_mw.sum())
    if len(solved.gen):
        is_slack = solved.gen.slack.fillna(False).astype(bool)
        slack_mw += float(solved.res_gen.loc[is_slack, "p_mw"].sum()
                          - (solved.gen.loc[is_slack, "p_mw"]
                             * solved.gen.loc[is_slack, "scaling"]).sum())
    return {
        "converged": True,
        "highest converged share of peak demand": 1.0,
        "generation dispatched (MW)": demand,
        "slack pick-up (MW)": slack_mw,
        "lines above 100% loading": int((solved.res_line.loading_percent.abs() > 100).sum()),
        "trafos above 100% loading": int((solved.res_trafo.loading_percent.abs() > 100).sum()
                                         + (solved.res_trafo3w.loading_percent.abs() > 100).sum()),
    }, solved


# Demand steps of 5%, from 10% of peak up to full peak. Coarser steps make the
# Newton solve give up early on operating points that are themselves solvable.
DEMAND_STEPS = tuple(round(i / 20, 2) for i in range(2, 21))


def run_ac_power_flow(net, steps=DEMAND_STEPS):
    solved = copy.deepcopy(net)
    dispatch_by_capacity(solved, 0.0)
    options = dict(calculate_voltage_angles=True, max_iteration=100,
                   distributed_slack=bool(solved.gen["slack_weight"].sum() > 0))
    try:
        pp.runpp(solved, **options)
    except NotImplementedError:
        # a distributed slack needs a single supplied zone; fall back if not
        options["distributed_slack"] = False
        pp.runpp(solved, **options)
    except Exception as exc:
        return ({"converged": False,
                 "error": f"unloaded network: {type(exc).__name__}: {exc}"}, solved)

    reached = 0.0
    for step in steps:
        dispatch_by_capacity(solved, step)
        try:
            pp.runpp(solved, init="results", **options)
        except Exception:
            # a failed solve leaves empty results, so re-solve at the last operating point that converged to keep dispatch and results consistent
            dispatch_by_capacity(solved, reached)
            pp.runpp(solved, **options)
            break
        reached = step

    dispatched = float((solved.gen["p_mw"] * solved.gen["scaling"]).sum()
                       + (solved.sgen["p_mw"] * solved.sgen["scaling"]).sum())
    slack_mw = float(solved.res_ext_grid["p_mw"].sum() + solved.res_gen["p_mw"].sum()
                     - (solved.gen["p_mw"] * solved.gen["scaling"]).sum())
    return {
        "converged at full demand": reached >= max(steps),
        "highest converged share of peak demand": reached,
        "generation dispatched (MW)": dispatched,
        "slack pick-up, i.e. losses (MW)": slack_mw,
        "slack shared over": (f"{int((solved.gen['slack_weight'] > 0).sum())} synchronous machines"
                              if options["distributed_slack"] else "one machine"),
        "buses solved": int(solved.res_bus.vm_pu.notna().sum()),
        "min vm_pu": float(solved.res_bus.vm_pu.min()),
        "max vm_pu": float(solved.res_bus.vm_pu.max()),
        "lines above 100% loading": int((solved.res_line.loading_percent > 100).sum()),
        "trafos above 100% loading": int((solved.res_trafo.loading_percent > 100).sum()
                                         + (solved.res_trafo3w.loading_percent > 100).sum()),
    }, solved


# N-1 contingency analysis
N1_VOLTAGE_LEVELS_KV = (400.0, 275.0)
BRANCH_TABLES = ("line", "trafo", "trafo3w")
BRANCH_LABELS = {"line": "Line", "trafo": "Transformer", "trafo3w": "Transformer"}
_FLOW_COLUMN = {"line": "p_from_mw", "trafo": "p_hv_mw", "trafo3w": "p_hv_mw"}


def n1_outage_cases(net, voltage_levels_kv=N1_VOLTAGE_LEVELS_KV, from_buses=None):
    """In-service lines and transformers whose from/HV bus is at one of the given voltage
    levels and, when ``from_buses`` is given, one of those buses."""
    cases = []
    for table, bus_column in (("line", "from_bus"), ("trafo", "hv_bus"), ("trafo3w", "hv_bus")):
        df = net[table]
        if not len(df):
            continue
        kv = net.bus.vn_kv.reindex(df[bus_column]).round(1).values
        live = df.in_service.values & np.isin(kv, voltage_levels_kv)
        if from_buses is not None:
            live &= df[bus_column].isin(from_buses).values
        cases += [(table, int(idx)) for idx in df.index[live]]
    return cases


def run_n1_contingencies(solved, mode="AC", voltage_levels_kv=N1_VOLTAGE_LEVELS_KV,
                         progress=None, cases=None, stop=None):
    """
    Outage every case from ``n1_outage_cases`` in turn on an already solved network and
    record, for every branch, its highest post-contingency loading and the outage that
    caused it, and for every bus its lowest and highest post-contingency voltage with
    their causing outages.

    ``solved`` is the network returned by ``run_ac_power_flow`` or ``run_dc_power_flow``:
    its dispatch is kept unchanged, and AC cases start from its solved voltages. ``mode``
    selects ``runpp`` ("AC") or ``rundcpp`` ("DC"); the DC load flow has no voltage
    solution so ``buses`` is None in that case. ``progress(done, total, label)`` is called
    after each case. ``cases`` is an explicit list of ``(table, index)`` outages to run
    instead of those from ``n1_outage_cases``. Buses islanded by an outage have no result
    and are ignored. ``stop()`` is checked before each case: when it returns True the run
    ends there, the result covers the cases run so far and ``stopped`` is True.
    """
    net = copy.deepcopy(solved)
    if cases is None:
        cases = n1_outage_cases(net, voltage_levels_kv)
    ac = mode == "AC"
    options = dict(calculate_voltage_angles=True, max_iteration=100,
                   distributed_slack=bool(net.gen["slack_weight"].sum() > 0)) if ac else {}
    base_res_bus = net.res_bus.copy()

    tables = [t for t in BRANCH_TABLES if len(net[t])]
    worst = {t: {"loading": np.full(len(net[t]), np.nan), "p_mw": np.full(len(net[t]), np.nan),
                 "cause": np.full(len(net[t]), -1)} for t in tables}
    n_bus = len(net.bus)
    vm_min, vm_max = np.full(n_bus, np.nan), np.full(n_bus, np.nan)
    vm_min_cause, vm_max_cause = np.full(n_bus, -1), np.full(n_bus, -1)

    labels, unsolved, attempted = [], [], 0
    for number, (table, idx) in enumerate(cases):
        if stop is not None and stop():
            break
        attempted = number + 1
        label = f"{BRANCH_LABELS[table]} {net[table].at[idx, 'name']}"
        labels.append(label)
        net[table].at[idx, "in_service"] = False
        try:
            if ac:
                net.res_bus = base_res_bus.copy()
                pp.runpp(net, init="results", **options)
            else:
                pp.rundcpp(net)
        except Exception as exc:
            unsolved.append({"outage": label, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            net[table].at[idx, "in_service"] = True
            if progress is not None:
                progress(number + 1, len(cases), label)

        for t in tables:
            results = net[f"res_{t}"]
            loading = results["loading_percent"].abs().to_numpy(dtype=float)
            higher = loading > np.nan_to_num(worst[t]["loading"], nan=-1.0)
            worst[t]["loading"][higher] = loading[higher]
            worst[t]["p_mw"][higher] = results[_FLOW_COLUMN[t]].to_numpy(dtype=float)[higher]
            worst[t]["cause"][higher] = number
        if ac:
            vm = net.res_bus["vm_pu"].to_numpy(dtype=float)
            lower = vm < np.nan_to_num(vm_min, nan=np.inf)
            higher = vm > np.nan_to_num(vm_max, nan=-np.inf)
            vm_min[lower], vm_min_cause[lower] = vm[lower], number
            vm_max[higher], vm_max_cause[higher] = vm[higher], number

    def cause_names(numbers):
        return np.array([labels[n] if n >= 0 else None for n in numbers], dtype=object)

    branches = {}
    for t in tables:
        branches[t] = pd.DataFrame({
            "max_loading_percent": worst[t]["loading"],
            "p_mw": worst[t]["p_mw"],
            "cause": cause_names(worst[t]["cause"]),
        }, index=net[t].index)
    buses = None
    if ac:
        buses = pd.DataFrame({
            "min_vm_pu": vm_min, "min_cause": cause_names(vm_min_cause),
            "max_vm_pu": vm_max, "max_cause": cause_names(vm_max_cause),
        }, index=net.bus.index)

    return {
        "mode": mode,
        "method": "load flow",
        "voltage levels (kV)": list(voltage_levels_kv),
        "cases": attempted,
        "requested": len(cases),
        "stopped": attempted < len(cases),
        "unsolved": unsolved,
        "branches": branches,
        "buses": buses,
    }


def run_demand_step(solved, load_index, step_mw, mode="AC"):
    """
    Add ``step_mw`` to one load of an already solved network, at the operating point it was
    solved at, meet the extra demand by re-dispatching generation in proportion to rated
    capacity (the same rule as the base load flows) and re-solve. Returns each branch's new
    loading and flow so the change against the base case can be read off; ``solved`` is left
    unchanged. The base solution may sit below peak demand (the AC ramp), so the step is
    applied in as-built terms scaled by that share, making it exactly ``step_mw`` at the
    operating point.
    """
    net = copy.deepcopy(solved)
    share = float(net.load.at[load_index, "scaling"]) or 1.0
    net.load.at[load_index, "p_mw"] += step_mw / share
    dispatch_by_capacity(net, share)
    label = f"{net.load.at[load_index, 'name']}"
    try:
        if mode == "AC":
            pp.runpp(net, init="results", calculate_voltage_angles=True, max_iteration=100,
                     distributed_slack=bool(net.gen["slack_weight"].sum() > 0))
        else:
            pp.rundcpp(net)
    except Exception as exc:
        return {"converged": False, "mode": mode, "load": label, "step_mw": step_mw,
                "error": f"{type(exc).__name__}: {exc}", "branches": {}, "buses": None}
    branches = {}
    for table in BRANCH_TABLES:
        if len(net[table]):
            results = net[f"res_{table}"]
            branches[table] = pd.DataFrame({
                "loading_percent": results["loading_percent"].abs(),
                "p_mw": results[_FLOW_COLUMN[table]],
            }, index=net[table].index)
    buses = net.res_bus[["vm_pu"]].copy() if mode == "AC" else None
    return {"converged": True, "mode": mode, "load": label, "step_mw": step_mw,
            "branches": branches, "buses": buses}


def report(net, run_load_flow=True):
    results = {
        "counts": element_counts(net),
        "issues": parameter_checks(net),
        "topology": topology_checks(net),
        "balance": power_balance(net),
    }

    print("\n=== Element counts ===")
    for key, value in results["counts"].items():
        print(f"  {key:<12} {value:>8,}")

    print("\n=== Parameter checks ===")
    if results["issues"]:
        for key, value in results["issues"].items():
            print(f"  ISSUE: {key} ({value})")
    else:
        print("  no invalid parameters found")

    print("\n=== Topology ===")
    for key, value in results["topology"].items():
        print(f"  {key:<40} {value:>8,}")

    print("\n=== Power balance (as built) ===")
    for key, value in results["balance"].items():
        print(f"  {key:<40} {value:>12,.1f}")

    if run_load_flow:
        print("\n=== DC load flow (generation dispatched by rated capacity) ===")
        results["dc_power_flow"], _ = run_dc_power_flow(net)
        for key, value in results["dc_power_flow"].items():
            print(f"  {key:<40} {value}")

        print("\n=== AC load flow (generation dispatched by rated capacity) ===")
        results["ac_power_flow"], _ = run_ac_power_flow(net)
        for key, value in results["ac_power_flow"].items():
            print(f"  {key:<40} {value}")

    return results
