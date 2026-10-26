"""
Presents a pandapower network, with its load-flow results, to the viewer.
"""

import json
import os
import pandas as pd
import pandapower as pp

import n1_screening
import network_checks as pp_checks

MODEL_EXTENSIONS = {".json": "pandapower"}
DATA_CENTRE_TYPE = "DC"  # the ``type`` of a data centre in the Demand Data sheet


def load_model(path):
    kind = MODEL_EXTENSIONS[os.path.splitext(path)[1].lower()]
    assert kind == "pandapower", kind
    return PandapowerModel(path)


class PandapowerModel:
    """A pandapower network, solved in place by ``solve``."""
    label = "pandapower"
    checks = pp_checks

    def __init__(self, path):
        self.net = pp.from_json(path)
        self.summary = None

    def solve(self, mode):
        run = self.checks.run_ac_power_flow if mode == "AC" else self.checks.run_dc_power_flow
        self.summary, self.net = run(self.net)

    @property
    def solved(self):
        return len(self.net.res_bus) > 0 and self.net.res_bus["vm_pu"].notna().any()

    def n1_cases(self, voltage_levels_kv=None, from_buses=None):
        """In-service circuits and transformers whose from/HV bus is at one of the voltage
        levels (default 400kV and 275kV) and, when ``from_buses`` is given, one of those."""
        if voltage_levels_kv is None:
            voltage_levels_kv = self.checks.N1_VOLTAGE_LEVELS_KV
        return self.checks.n1_outage_cases(self.net, voltage_levels_kv, from_buses)

    def n1_case_count(self, voltage_levels_kv=None, from_buses=None):
        return len(self.n1_cases(voltage_levels_kv, from_buses))

    def run_n1(self, mode, progress, full_resolve=False, resolve_above=None, stop=None,
               voltage_levels_kv=None, from_buses=None):
        """N-1 screen of the ``n1_cases`` outages with line outage distribution factors, or
        a re-solve of every outage with the selected load flow. With ``resolve_above`` (a
        loading in percent) only the outages the screen finds loading some element above it
        are re-solved and the two results combined. Three-winding transformer outages, which
        the linear screen cannot handle, are re-solved either way. ``stop()`` ends a
        re-solve early."""
        if voltage_levels_kv is None:
            voltage_levels_kv = self.checks.N1_VOLTAGE_LEVELS_KV
        cases = self.n1_cases(voltage_levels_kv, from_buses)
        if full_resolve and resolve_above is None:
            result = self.checks.run_n1_contingencies(self.net, mode, progress=progress,
                                                      cases=cases, stop=stop)
        else:
            result = n1_screening.run_n1_lodf_screen(
                self.net, cases, mode=mode, progress=progress,
                resolve=lambda subset: self.checks.run_n1_contingencies(
                    self.net, mode, cases=subset, stop=stop))
            if full_resolve:
                subset = n1_screening.cases_loading_above(result, cases, resolve_above)
                resolved = self.checks.run_n1_contingencies(self.net, mode, progress=progress,
                                                            cases=subset, stop=stop)
                result = n1_screening.combine_with_resolved(result, resolved, resolve_above)
        result["voltage levels (kV)"] = list(voltage_levels_kv)
        return result

    def counts(self):
        net = self.net
        return {"buses": len(net.bus), "circuits": len(net.line),
                "transformers": len(net.trafo) + len(net.trafo3w)}

    def loads(self):
        loads = self.net.load
        return pd.DataFrame({"name": loads["name"], "bus": loads["bus"], "p_mw": loads["p_mw"],
                             "demand_type": loads["demand_type"].astype(str).str.strip(),
                             "in_service": loads["in_service"]}, index=loads.index)

    def data_centres(self):
        """Loads typed as data centres: name, bus, as-built peak demand and service state."""
        loads = self.loads()
        return loads[loads["demand_type"] == DATA_CENTRE_TYPE].drop(columns="demand_type")

    def generators(self):
        """Synchronous (gen) and static (sgen) generators with their capacity and, once
        solved, dispatch. Dynamic compensation sits in the gen table with no capacity and
        is left out (see ``compensation``)."""
        net = self.net
        frames = []
        for table, model in (("gen", "Synchronous"), ("sgen", "Static")):
            df, results = net[table], net[f"res_{table}"]
            if not len(df):
                continue
            capacity = df["max_p_mw"].where(df["max_p_mw"].notna(), df["p_mw"])
            frame = pd.DataFrame({
                "name": df["name"], "bus": df["bus"], "capacity_mw": capacity,
                "plant_type": df["plant_type"] if "plant_type" in df else "",
                "model": model, "in_service": df["in_service"]}, index=df.index)
            frame["dispatch_mw"] = (results["p_mw"].reindex(df.index)
                                    if self.solved and len(results) else pd.NA)
            frames.append(frame[capacity.fillna(0.0) > 0])
        return pd.concat(frames, ignore_index=True)

    def run_demand_step(self, load_id, step_mw, mode):
        return self.checks.run_demand_step(self.net, load_id, step_mw, mode)

    def buses(self):
        net = self.net
        buses = net.bus[["name", "vn_kv", "site", "site_name", "zone", "owner",
                         "in_service"]].copy()
        coords = net.bus["geo"].dropna().map(lambda s: json.loads(s)["coordinates"])
        buses["lon"] = coords.map(lambda c: c[0])
        buses["lat"] = coords.map(lambda c: c[1])
        buses["vm_pu"] = net.res_bus["vm_pu"] if self.solved else pd.NA
        return buses

    def branches(self):
        """Lines and transformers as one frame; ``table`` and ``element`` keep the
        pandapower table and index of each row so per-table results can be joined on."""
        net = self.net
        frames = []
        specs = [
            ("line", "Line", "from_bus", "to_bus", "winter_mva", "p_from_mw", "circuit_type",
             lambda df: df["circuit_type"].astype(str) + ", " + df["length_km"].round(1).astype(str) + "km"),
            ("trafo", "Transformer", "hv_bus", "lv_bus", "sn_mva", "p_hv_mw", "tx_type",
             lambda df: df["tx_type"].astype(str) + ", two winding"),
            ("trafo3w", "Transformer", "hv_bus", "mv_bus", "sn_hv_mva", "p_hv_mw", "tx_type",
             lambda df: df["tx_type"].astype(str) + ", three winding"),
        ]
        for table, kind, from_col, to_col, rating_col, flow_col, type_col, detail in specs:
            df, results = net[table], net[f"res_{table}"]
            if not len(df):
                continue
            frame = pd.DataFrame({
                "kind": kind, "table": table, "element": df.index, "name": df["name"],
                "from_bus": df[from_col], "to_bus": df[to_col], "detail": detail(df),
                "subtype": df[type_col].astype(str),
                "length_km": df["length_km"] if table == "line" else pd.NA,
                "rating_mva": df[rating_col], "in_service": df["in_service"],
            }, index=df.index)
            frame["loading"] = results["loading_percent"].abs() if len(results) else pd.NA
            frame["p_mw"] = results[flow_col] if len(results) else pd.NA
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def bus_injections(self):
        """Demand and generation (MW) per bus at the operating point shown: the solved
        results, which include the slack contribution, or the network as built."""
        net = self.net

        def by_bus(table, results):
            live = table[table["in_service"]]
            if results is not None and len(results) and results["p_mw"].notna().any():
                mw = results["p_mw"].reindex(live.index)
            else:
                mw = live["p_mw"] * live["scaling"]
            return pd.Series(mw.values, index=live["bus"].values).groupby(level=0).sum()

        generation = pd.concat([by_bus(net.gen, net.res_gen), by_bus(net.sgen, net.res_sgen)])
        if len(net.res_ext_grid):
            generation = pd.concat([generation, pd.Series(net.res_ext_grid["p_mw"].values,
                                                          index=net.ext_grid["bus"].values)])
        return pd.DataFrame({
            "demand_mw": by_bus(net.load, net.res_load),
            "generation_mw": generation.groupby(level=0).sum(),
        }).reindex(net.bus.index).fillna(0.0)

    def compensation(self):
        """Fixed shunts and the dynamic devices (SVC, STATCOM, synchronous compensators)
        held in the gen table with no active power capacity; MVAr absorbed and generated
        are both positive."""
        net = self.net
        shunts, q = net.shunt, net.shunt["q_mvar"]  # +ve absorbing
        fixed = pd.DataFrame({
            "name": shunts["name"], "bus": shunts["bus"], "device": shunts["device_type"],
            "absorb_mvar": q.clip(lower=0.0), "generate_mvar": (-q).clip(lower=0.0),
            "in_service": shunts["in_service"]})
        gens = net.gen[net.gen["max_p_mw"].fillna(0.0) <= 0]
        dynamic = pd.DataFrame({
            "name": gens["name"], "bus": gens["bus"], "device": gens["type"],
            "absorb_mvar": (-gens["min_q_mvar"]).clip(lower=0.0),
            "generate_mvar": gens["max_q_mvar"].clip(lower=0.0),
            "in_service": gens["in_service"]})
        return pd.concat([fixed, dynamic], ignore_index=True)

