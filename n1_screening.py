"""N-1 thermal screening of a solved pandapower or PyPSA network with line outage distribution factors."""

import copy

import numpy as np
import pandapower as pp
import pandas as pd

from linearised_network import LinearisedNetwork

CHUNK = 256


def _per_component_share(weights, lin):
    share = np.asarray(weights, dtype=float).copy()
    for component in np.unique(lin.component):
        members = lin.component == component
        total = share[members].sum()
        if total > 0:
            share[members] /= total
        else:
            share[members] = 0.0
            refs = lin.slack_index[members[lin.slack_index]]
            share[refs] = 1.0 / max(len(refs), 1)
    return share


class _Source:
    labels = {}

    def __init__(self, solved, mode, slack_weights):
        self.solved = solved
        self.mode = mode
        self.lin = self._linearise(solved)
        self.p0, self.q0, self.v0, self.p1, self.q1, self.v1 = self._flows()
        self.rating = self._ratings()
        self.slack_share = _per_component_share(
            self._slack_weights(slack_weights), self.lin)

    def _linearise(self, solved):
        raise NotImplementedError

    def _flows(self):
        raise NotImplementedError

    def _ratings(self):
        raise NotImplementedError

    def _slack_weights(self, given):
        raise NotImplementedError

    def tables(self):
        raise NotImplementedError

    def index(self, table):
        raise NotImplementedError

    def label(self, table, idx):
        raise NotImplementedError

    def _assemble(self, per_table_row):
        n_rows = len(self.lin.ppc_to_internal_row)
        full = np.zeros(n_rows)
        for key, values in per_table_row.items():
            full[self.lin.ppc_branch_indices[key]] = values
        return np.nan_to_num(full[self.lin.ppc_to_internal_row >= 0])


class _PandapowerSource(_Source):
    labels = {"line": "Line", "trafo": "Transformer", "trafo3w": "Transformer"}

    def _linearise(self, solved):
        net = copy.deepcopy(solved)
        pp.rundcpp(net, numba=False)
        return LinearisedNetwork(net)

    def _flows(self):
        net, dc = self.solved, self.mode == "DC"
        p0, q0, v0, p1, q1, v1 = ({} for _ in range(6))

        def take(key, results, p_a, q_a, v_a, p_b, q_b, v_b, sign=1.0):
            p0[key] = sign * results[p_a].to_numpy(dtype=float)
            p1[key] = sign * results[p_b].to_numpy(dtype=float)
            if dc:
                q0[key] = q1[key] = np.zeros(len(results))
                v0[key] = v1[key] = np.ones(len(results))
            else:
                q0[key] = sign * results[q_a].to_numpy(dtype=float)
                q1[key] = sign * results[q_b].to_numpy(dtype=float)
                v0[key] = results[v_a].to_numpy(dtype=float)
                v1[key] = results[v_b].to_numpy(dtype=float)

        if "line" in self.lin.ppc_branch_indices and len(net.line):
            take("line", net.res_line, "p_from_mw", "q_from_mvar", "vm_from_pu",
                 "p_to_mw", "q_to_mvar", "vm_to_pu")
        if "trafo" in self.lin.ppc_branch_indices and len(net.trafo):
            take("trafo", net.res_trafo, "p_hv_mw", "q_hv_mvar", "vm_hv_pu",
                 "p_lv_mw", "q_lv_mvar", "vm_lv_pu")
        if "trafo3w" in self.lin.ppc_branch_indices and len(net.trafo3w):
            # MV and LV branches run star point -> terminal, opposite to their results
            for winding, sign in (("hv", 1.0), ("mv", -1.0), ("lv", -1.0)):
                take(f"trafo3w_{winding}", net.res_trafo3w, f"p_{winding}_mw",
                     f"q_{winding}_mvar", f"vm_{winding}_pu", f"p_{winding}_mw",
                     f"q_{winding}_mvar", f"vm_{winding}_pu", sign)
                p1[f"trafo3w_{winding}"] = -p0[f"trafo3w_{winding}"]
        return tuple(self._assemble(d) for d in (p0, q0, v0, p1, q1, v1))

    def _ratings(self):
        net = self.solved
        rating = {}
        if "line" in self.lin.ppc_branch_indices and len(net.line):
            kv = net.bus["vn_kv"].reindex(net.line["from_bus"]).to_numpy(dtype=float)
            rating["line"] = (np.sqrt(3) * kv * net.line["max_i_ka"].to_numpy(dtype=float)
                              * net.line["df"].to_numpy(dtype=float)
                              * net.line["parallel"].to_numpy(dtype=float))
        if "trafo" in self.lin.ppc_branch_indices and len(net.trafo):
            rating["trafo"] = (net.trafo["sn_mva"].to_numpy(dtype=float)
                               * net.trafo["df"].to_numpy(dtype=float)
                               * net.trafo["parallel"].to_numpy(dtype=float))
        if "trafo3w" in self.lin.ppc_branch_indices and len(net.trafo3w):
            for winding in ("hv", "mv", "lv"):
                rating[f"trafo3w_{winding}"] = net.trafo3w[f"sn_{winding}_mva"].to_numpy(dtype=float)
        full = np.full(len(self.lin.ppc_to_internal_row), np.nan)
        for key, values in rating.items():
            full[self.lin.ppc_branch_indices[key]] = values
        return full[self.lin.ppc_to_internal_row >= 0]

    def _slack_weights(self, given):
        net = self.solved
        bus_lookup = self.lin.network["_pd2ppc_lookups"]["bus"]
        weights = np.zeros(self.lin.pv_pq_mask.size)
        if given is not None:
            nodes = bus_lookup[np.asarray(given.index, dtype=int)]
            np.add.at(weights, nodes[nodes >= 0], given.to_numpy(dtype=float)[nodes >= 0])
        elif net["_options"].get("distributed_slack", False):
            for table in ("gen", "ext_grid"):
                frame = net[table]
                if not len(frame) or "slack_weight" not in frame.columns:
                    continue
                live = frame[frame["in_service"].astype(bool) & (frame["slack_weight"].fillna(0) > 0)]
                nodes = bus_lookup[live["bus"].to_numpy(dtype=int)]
                np.add.at(weights, nodes[nodes >= 0],
                          live["slack_weight"].to_numpy(dtype=float)[nodes >= 0])
        return weights

    def tables(self):
        return [t for t in self.labels if len(self.solved[t])]

    def index(self, table):
        return self.solved[table].index

    def label(self, table, idx):
        return f"{self.labels[table]} {self.solved[table].at[idx, 'name']}"


class _PypsaSource(_Source):
    labels = {"lines": "Line", "transformers": "Transformer"}

    def _linearise(self, solved):
        return LinearisedNetwork.from_pypsa(solved.copy())

    def _flows(self):
        n, snapshot = self.solved, self.solved.snapshots[0]
        vm = n.buses_t.v_mag_pu
        p0, q0, v0, p1, q1, v1 = ({} for _ in range(6))
        for table in self.lin.ppc_branch_indices:
            static, dynamic = n.c[table].static, n.c[table].dynamic
            p0[table] = dynamic["p0"].loc[snapshot].reindex(static.index).to_numpy(dtype=float)
            p1[table] = dynamic["p1"].loc[snapshot].reindex(static.index).to_numpy(dtype=float)
            if self.mode == "DC" or dynamic["q0"].empty or vm.empty:
                q0[table] = q1[table] = np.zeros(len(static))
                v0[table] = v1[table] = np.ones(len(static))
            else:
                q0[table] = dynamic["q0"].loc[snapshot].reindex(static.index).to_numpy(dtype=float)
                q1[table] = dynamic["q1"].loc[snapshot].reindex(static.index).to_numpy(dtype=float)
                v0[table] = vm.loc[snapshot].reindex(static["bus0"]).to_numpy(dtype=float)
                v1[table] = vm.loc[snapshot].reindex(static["bus1"]).to_numpy(dtype=float)
        return tuple(self._assemble(d) for d in (p0, q0, v0, p1, q1, v1))

    def _ratings(self):
        full = np.full(len(self.lin.ppc_to_internal_row), np.nan)
        for table, rows in self.lin.ppc_branch_indices.items():
            full[rows] = self.solved.c[table].static["s_nom"].to_numpy(dtype=float)
        return full[self.lin.ppc_to_internal_row >= 0]

    def _slack_weights(self, given):
        weights = np.zeros(self.lin.pv_pq_mask.size)
        if given is not None:
            position = pd.Series(np.arange(len(self.solved.buses)), index=self.solved.buses.index)
            nodes = position.reindex(given.index).to_numpy()
            valid = ~pd.isna(nodes)
            np.add.at(weights, nodes[valid].astype(int), given.to_numpy(dtype=float)[valid])
        return weights

    def tables(self):
        return [t for t in self.labels if len(self.solved.c[t].static)]

    def index(self, table):
        return self.solved.c[table].static.index

    def label(self, table, idx):
        return f"{self.labels[table]} {idx}"


def _source_for(solved, mode, slack_weights):
    if isinstance(solved, pp.pandapowerNet):
        return _PandapowerSource(solved, mode, slack_weights)
    if type(solved).__name__ == "Network" and hasattr(solved, "sub_networks"):
        return _PypsaSource(solved, mode, slack_weights)
    raise TypeError(f"not a pandapower or PyPSA network: {type(solved)!r}")


# An outage only counts as loading an element where it raises the loading by more than this
# over the intact network, so an overload already present intact does not flag every outage.
MATERIAL_RISE_POINTS = 1.0


class _Worst:
    def __init__(self, source, case_count):
        lin = source.lin
        self.source = source
        self.rows = {t: lin.element_branch_rows(t) for t in source.tables()}
        self.loading = {t: np.full(len(r), np.nan) for t, r in self.rows.items()}
        self.p_mw = {t: np.full(len(r), np.nan) for t, r in self.rows.items()}
        self.cause = {t: np.full(len(r), -1) for t, r in self.rows.items()}
        self.peak = np.full(case_count, np.nan)  # per outage, the highest loading it raises
        with np.errstate(invalid="ignore", divide="ignore"):
            intact = (np.maximum(np.hypot(source.p0, source.q0) / source.v0,
                                 np.hypot(source.p1, source.q1) / source.v1)
                      / source.rating * 100.0)
        self.intact = {t: self._element_loading(intact[:, None], rows)[:, 0]
                       for t, rows in self.rows.items()}

    @staticmethod
    def _element_loading(loading, rows):
        """Each element's loading (its most loaded winding), -1 where it has none."""
        safe_rows = np.where(rows >= 0, rows, 0)
        winding_loading = np.nan_to_num(loading[safe_rows], nan=-1.0)
        winding_loading[rows < 0] = -1.0
        return winding_loading.max(axis=1)

    def _note_peaks(self, table, post_loading, case_numbers):
        raised = post_loading > self.intact[table][:, None] + MATERIAL_RISE_POINTS
        if raised.any():
            peak = np.where(raised, post_loading, -np.inf).max(axis=0)
            peak = np.where(np.isfinite(peak), peak, np.nan)
            self.peak[case_numbers] = np.fmax(self.peak[case_numbers], peak)

    def update(self, delta_p, case_numbers, no_result=None):
        s = self.source
        with np.errstate(invalid="ignore", divide="ignore"):
            end0 = np.hypot(s.p0[:, None] + delta_p, s.q0[:, None]) / s.v0[:, None]
            end1 = np.hypot(s.p1[:, None] - delta_p, s.q1[:, None]) / s.v1[:, None]
            loading = np.maximum(end0, end1) / s.rating[:, None] * 100.0
        if no_result is not None:
            loading[no_result] = np.nan
        post_p = s.p0[:, None] + delta_p
        for table, rows in self.rows.items():
            safe_rows = np.where(rows >= 0, rows, 0)
            element_loading = self._element_loading(loading, rows)
            self._note_peaks(table, element_loading, case_numbers)
            case = np.argmax(element_loading, axis=1)
            best = element_loading[np.arange(len(rows)), case]
            better = best > np.nan_to_num(self.loading[table], nan=-1.0)
            self.loading[table][better] = best[better]
            self.p_mw[table][better] = post_p[safe_rows[better, 0], case[better]]
            self.cause[table][better] = case_numbers[case[better]]

    def merge(self, branches, labels):
        for table, frame in branches.items():
            if table not in self.loading:
                continue
            frame = frame.reindex(self.source.index(table))
            other = np.nan_to_num(frame["max_loading_percent"].to_numpy(dtype=float), nan=-1.0)
            better = other > np.nan_to_num(self.loading[table], nan=-1.0)
            self.loading[table][better] = other[better]
            self.p_mw[table][better] = frame["p_mw"].to_numpy(dtype=float)[better]
            for cause in frame.loc[better, "cause"].unique():
                if cause not in labels:
                    labels.append(cause)
            self.cause[table][better] = [labels.index(c) for c in frame.loc[better, "cause"]]
            # Re-solved results only give each element's worst outage, so an outage's peak is
            # the highest loading it is recorded as raising.
            raised = better & (other > self.intact[table] + MATERIAL_RISE_POINTS)
            caused = pd.Series(other[raised], index=self.cause[table][raised])
            if len(caused):
                worst = caused.groupby(level=0).max()
                numbers = worst.index.to_numpy()
                in_range = numbers < len(self.peak)
                self.peak[numbers[in_range]] = np.fmax(self.peak[numbers[in_range]],
                                                       worst.to_numpy()[in_range])

    def frames(self, labels):
        out = {}
        for table in self.loading:
            cause = np.array([labels[n] if n >= 0 else None for n in self.cause[table]],
                             dtype=object)
            out[table] = pd.DataFrame({
                "max_loading_percent": self.loading[table],
                "p_mw": self.p_mw[table],
                "cause": cause,
            }, index=self.source.index(table))
        return out


def run_n1_lodf_screen(solved, cases, mode="DC", progress=None, resolve=None, slack_weights=None):
    def report(done, label):
        if progress is not None:
            progress(done, len(cases), label)

    report(0, "building the PTDF and LODF matrices")
    source = _source_for(solved, mode, slack_weights)
    lin = source.lin
    worst = _Worst(source, len(cases))
    p0 = source.p0

    labels = [source.label(table, idx) for table, idx in cases]
    rows_by_table = {table: lin.element_branch_rows(table) for table in source.tables()}
    positions = {table: pd.Index(source.index(table)) for table in source.tables()}
    unsolved, deferred = [], []
    single, single_numbers = [], []
    for number, (table, idx) in enumerate(cases):
        rows = rows_by_table[table][positions[table].get_loc(idx)]
        rows = rows[rows >= 0]
        if len(rows) == 1:
            single.append(rows[0])
            single_numbers.append(number)
        elif len(rows) == 0:
            unsolved.append({"outage": labels[number],
                             "reason": "not connected to the solved network"})
        else:
            deferred.append(number)
    single = np.array(single, dtype=int)
    single_numbers = np.array(single_numbers, dtype=int)

    is_bridge = lin.bridge_mask[single]
    connected, connected_numbers = single[~is_bridge], single_numbers[~is_bridge]
    done = 0
    for start in range(0, len(connected), CHUNK):
        k = connected[start:start + CHUNK]
        delta_p = lin.lodf[:, k] * p0[k][None, :]
        worst.update(delta_p, connected_numbers[start:start + CHUNK])
        done += len(k)
        report(done, "screening outages that keep the network connected")

    # Bridge outage: nodes with |PTDF| = 1 on the bridge form the island, which is dropped; the
    # node left behind loses the bridge flow and the slack of its part of the network picks it up.
    bridges, bridge_numbers = single[is_bridge], single_numbers[is_bridge]
    abs_grounded_incidence = abs(lin.grounded_incidence.T)
    grounded_share = source.slack_share[lin.pv_pq_mask]
    grounded_component = lin.component[lin.pv_pq_mask]
    for start in range(0, len(bridges), CHUNK):
        k = bridges[start:start + CHUNK]
        island = np.abs(lin.ptdf[k, :]) > 0.5
        island_branch = (abs_grounded_incidence @ island.T.astype(float)) >= 2 - 1e-9
        g_from, g_to = lin.grounded_index[lin.from_node[k]], lin.grounded_index[lin.to_node[k]]
        from_in_island = (g_from >= 0) & island[np.arange(len(k)), np.maximum(g_from, 0)]
        node_left = np.where(from_in_island, g_to, g_from)
        lost = np.where(from_in_island, -p0[k], p0[k])
        injections = np.zeros((lin.grounded_incidence.shape[0], len(k)))
        keep = node_left >= 0
        injections[node_left[keep], np.flatnonzero(keep)] += lost[keep]
        component = lin.component[np.where(from_in_island, lin.to_node[k], lin.from_node[k])]
        same_part = grounded_component[:, None] == component[None, :]
        share = grounded_share[:, None] * same_part * (~island.T)
        reference_share = np.array([
            source.slack_share[(lin.component == c) & ~lin.pv_pq_mask].sum() for c in component])
        share_total = share.sum(axis=0) + reference_share
        injections -= lost[None, :] * share / share_total[None, :]
        delta_p = lin.ptdf @ injections
        delta_p[k, np.arange(len(k))] = -p0[k]
        worst.update(delta_p, bridge_numbers[start:start + CHUNK], no_result=island_branch)
        done += len(k)
        report(done, "screening outages that island part of the network")

    resolved = 0
    if deferred:
        subset = [cases[n] for n in deferred]
        if resolve is None:
            unsolved += [{"outage": labels[n], "reason": "a multi-branch outage, which the "
                          "linear screen cannot handle"} for n in deferred]
        else:
            report(done, f"re-solving {len(subset):,} multi-branch outages")
            other = resolve(subset)
            worst.merge(other["branches"], labels)
            unsolved += other["unsolved"]
            resolved = len(subset) - len(other["unsolved"])
            done += len(subset)
    report(len(cases), "done")

    return {
        "mode": mode,
        "method": "LODF",
        "cases": len(cases),
        "islanding outages": int(is_bridge.sum()),
        "re-solved outages": resolved,
        "unsolved": unsolved,
        "branches": worst.frames(labels),
        "buses": None,
        # Per outage in the order of ``cases``: the highest post-outage loading among the
        # elements it raises by more than MATERIAL_RISE_POINTS, NaN where it raises none.
        "case peak loading": worst.peak,
    }


def cases_loading_above(screen, cases, threshold_percent):
    """The outages of a ``run_n1_lodf_screen`` result that raise some element's loading
    above ``threshold_percent``, in their original order."""
    peak = np.asarray(screen["case peak loading"], dtype=float)
    return [case for case, value in zip(cases, peak) if value > threshold_percent]


def combine_with_resolved(screen, resolved, threshold_percent):
    """A ``run_n1_lodf_screen`` result merged with a ``run_n1_contingencies`` result for a
    subset of its outages: per element the higher loading with its cause, the re-solved
    bus voltages, and both lists of unsolved outages."""
    branches = {}
    for table, frame in screen["branches"].items():
        other = resolved["branches"].get(table)
        if other is None:
            branches[table] = frame
            continue
        other = other.reindex(frame.index)
        better = (np.nan_to_num(other["max_loading_percent"].to_numpy(dtype=float), nan=-1.0)
                  > np.nan_to_num(frame["max_loading_percent"].to_numpy(dtype=float), nan=-1.0))
        merged = frame.copy()
        for column in ("max_loading_percent", "p_mw", "cause"):
            merged.loc[better, column] = other.loc[better, column].to_numpy()
        branches[table] = merged
    seen, unsolved = set(), []
    for entry in screen["unsolved"] + resolved["unsolved"]:
        if entry["outage"] not in seen:
            seen.add(entry["outage"])
            unsolved.append(entry)
    return {
        **screen,
        "method": "LODF + load flow",
        "re-solve threshold (%)": threshold_percent,
        "re-solve candidates": resolved["requested"],
        "re-solved outages": screen["re-solved outages"] + resolved["cases"],
        "stopped": resolved["stopped"],
        "unsolved": unsolved,
        "branches": branches,
        "buses": resolved["buses"],
    }
