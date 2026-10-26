"""PTDF-based analysis of a linearised Pandapower or PyPSA network."""

from functools import wraps
from time import time

import numpy as np
import pandapower as pn
import scipy.linalg
import scipy.sparse.csgraph
from scipy import sparse as sp

TIMING = False
BRIDGE_TOLERANCE = 1e-6
PYPSA_BRANCH_TABLES = ("lines", "transformers")


def _timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te - ts:2.4f} sec")
        return result

    if TIMING:
        return wrap

    else:
        return f


class LinearisedNetwork:
    """A class for linearised PTDF analysis of a Pandapower or PyPSA network"""

    def __init__(self, network: pn.pandapowerNet):
        """Initialise the LinearisedNetwork class.

        :param network: Pandapower network, with a solved DC load flow.
        """
        internal = network["_ppc"]["internal"]
        if "Bf" not in internal:
            raise ValueError("The network needs a solved DC load flow (pandapower.rundcpp) first.")

        branch_bus_susceptance = sp.csr_array(internal["Bf"])
        incidence = sp.csc_array(internal["Cft"].T)
        from_node = np.asarray(sp.csr_array(incidence.T).argmax(axis=1)).ravel()
        susceptance = np.asarray(
            branch_bus_susceptance[np.arange(incidence.shape[1]), from_node]
        ).ravel()

        branch_in_service = np.asarray(internal["branch_is"], dtype=bool)
        branch_indices = dict(network["_pd2ppc_lookups"]["branch"])
        windings = {}
        if "trafo3w" in branch_indices.keys():
            start, end = branch_indices["trafo3w"]
            n_trafo3w = (end - start) // 3
            windings["trafo3w"] = []
            for number, winding in enumerate(["hv", "mv", "lv"]):
                branch_indices[f"trafo3w_{winding}"] = (
                    start + n_trafo3w * number,
                    start + n_trafo3w * (number + 1),
                )
                windings["trafo3w"].append(f"trafo3w_{winding}")

        self._build(
            network=network,
            susceptance=susceptance,
            incidence=incidence,
            phase_shift_p=-1 * np.asarray(internal["Pbusinj"]).reshape(-1, 1),
            slack_index=np.asarray(internal["ref"], dtype=int),
            table_rows={
                key: np.arange(*value, dtype=int) for key, value in branch_indices.items()
            },
            table_to_internal_row=np.where(
                branch_in_service, np.cumsum(branch_in_service) - 1, -1
            ),
            windings=windings,
            aux_node_indices=network["_pd2ppc_lookups"]["aux"],
        )

    @classmethod
    def from_pypsa(cls, network) -> "LinearisedNetwork":
        """Initialise the LinearisedNetwork class from a PyPSA network.

        :param network: PyPSA network. Its dependent values and topology are recalculated.
        """
        network.calculate_dependent_values()
        network.determine_network_topology()
        for sub_network in network.sub_networks["obj"]:
            sub_network.find_slack_bus()

        buses = network.buses.index
        bus_position = {name: number for number, name in enumerate(buses)}
        tables = [t for t in PYPSA_BRANCH_TABLES if len(network.c[t].static)]
        frames = [network.c[t].static for t in tables]
        starts = np.cumsum([0] + [len(frame) for frame in frames])
        table_rows = {
            t: np.arange(starts[number], starts[number + 1], dtype=int)
            for number, t in enumerate(tables)
        }
        active = np.concatenate(
            [frame["active"].to_numpy(dtype=bool) for frame in frames]
        ) if frames else np.zeros(0, dtype=bool)
        reactance = np.concatenate(
            [frame["x_pu_eff"].to_numpy(dtype=float) for frame in frames]
        )[active]
        susceptance = np.divide(1.0, reactance, out=np.full_like(reactance, np.inf),
                                where=reactance != 0)
        bus0 = np.concatenate([frame["bus0"].map(bus_position).to_numpy(dtype=int)
                               for frame in frames])[active]
        bus1 = np.concatenate([frame["bus1"].map(bus_position).to_numpy(dtype=int)
                               for frame in frames])[active]
        n_branch = int(active.sum())
        rows = np.arange(n_branch)
        incidence = sp.csc_array(
            (np.r_[np.ones(n_branch), -np.ones(n_branch)], (np.r_[bus0, bus1], np.r_[rows, rows])),
            shape=(len(buses), n_branch),
        )
        shift = np.concatenate([
            np.deg2rad(frame["phase_shift"].to_numpy(dtype=float))
            if t == "transformers" else np.zeros(len(frame))
            for t, frame in zip(tables, frames)
        ])[active]
        phase_shift_p = (incidence @ (susceptance * shift)).reshape(-1, 1)
        slack_index = np.array(
            [bus_position[bus] for bus in network.sub_networks["slack_bus"]], dtype=int
        )

        self = cls.__new__(cls)
        self._build(
            network=network,
            susceptance=susceptance,
            incidence=incidence,
            phase_shift_p=phase_shift_p,
            slack_index=slack_index,
            table_rows=table_rows,
            table_to_internal_row=np.where(active, np.cumsum(active) - 1, -1),
            windings={},
            aux_node_indices={},
        )
        return self

    def _build(self, network, susceptance, incidence, phase_shift_p, slack_index, table_rows,
               table_to_internal_row, windings, aux_node_indices):
        self.network = network  #: the Pandapower or PyPSA network
        self.incidence = sp.csc_array(incidence)  #: the node / branch incidence matrix
        from_to = sp.csr_array(self.incidence.T)
        self.from_node = np.asarray(from_to.argmax(axis=1)).ravel()  #: from node of each branch
        self.to_node = np.asarray(from_to.argmin(axis=1)).ravel()  #: to node of each branch
        self.diag_susceptance = np.asarray(susceptance, dtype=float)  #: a vector of susceptances
        self.branch_susceptance = sp.diags_array(
            self.diag_susceptance, format="csc"
        )  #: a diagonal matrix of susceptances
        self.nodal_susceptance = (
            self.incidence @ self.branch_susceptance @ self.incidence.T
        )  #: the nodal susceptance matrix

        self.phase_shift_p = np.asarray(phase_shift_p, dtype=float).reshape(
            -1, 1
        )  #: an array of phase shifts, converted to power injections

        # extra indices
        self.aux_node_indices = aux_node_indices  #: indices of auxiliary busbars
        self.ppc_to_internal_row = np.asarray(
            table_to_internal_row, dtype=int
        )  #: internal branch row of each element-table row, -1 if not in the internal matrices
        self.ppc_branch_indices = {
            key: np.asarray(value, dtype=int) for key, value in table_rows.items()
        }  #: element-table rows of different types of branches
        self.branch_indices = {
            key: self.ppc_to_internal_row[value][self.ppc_to_internal_row[value] >= 0]
            for key, value in self.ppc_branch_indices.items()
        }  #: internal rows of different types of branches
        self.windings = dict(windings)  #: winding keys of tables whose elements span several branches
        self.component = scipy.sparse.csgraph.connected_components(
            abs(self.incidence) @ abs(self.incidence.T), directed=False
        )[1]  #: connected component of each node

        # slack indexing
        self.slack_index = np.asarray(slack_index, dtype=int)  #: indices of slack nodes
        self.pv_pq_mask = np.ones(
            self.phase_shift_p.shape[0], dtype=bool
        )  #: boolean mask of PV and PQ nodes
        self.pv_pq_mask[self.slack_index] = False
        self.grounded_index = np.where(
            self.pv_pq_mask, np.cumsum(self.pv_pq_mask) - 1, -1
        )  #: index of each node in the grounded matrices, -1 for the slack nodes

        # grounding
        self.grounded_incidence = self.incidence[
            self.pv_pq_mask
        ]  #: grounded incidence matrix, with the slack nodes removed
        self.grounded_nodal_susceptance = self.nodal_susceptance[self.pv_pq_mask][
            :, self.pv_pq_mask
        ]  #: grounded susceptance matrix, with the slack nodes removed

        self.angle = np.zeros(self.pv_pq_mask.size).reshape(
            -1, 1
        )  #: zero array of voltage angles

        # Injection vectors
        self.branch_unit = sp.eye_array(
            self.branch_susceptance.shape[1], format="csc"
        )  #: branch unit injection matrix
        self.grounded_unit_p = (
            self.grounded_incidence @ self.branch_unit
        )  #: grounded branch unit injection matrix

        # Inversion and PTDF
        self.inv_grounded_nodal_susceptance = scipy.linalg.solve(
            self.grounded_nodal_susceptance.toarray(),
            np.identity(self.grounded_nodal_susceptance.shape[0]),
            assume_a="sym",
        )  #: inverted grounded nodal susceptance matrix
        self.ptdf = (
            self.branch_susceptance.tocsr() @ self.grounded_incidence.T
        ) @ self.inv_grounded_nodal_susceptance  #: matrix of power transfer distribution factors

        self.bptdf = self.ptdf @ self.grounded_incidence.toarray()  #: matrix of branch PTDFS

        # Line outage distribution factors
        self_bptdf = np.diag(self.bptdf)
        self.bridge_mask = (
            np.abs(1.0 - self_bptdf) < BRIDGE_TOLERANCE
        )  #: boolean mask of branches which are a bridge between two parts of the system
        with np.errstate(divide="ignore", invalid="ignore"):
            outage_scaling = np.where(self.bridge_mask, 0.0, 1.0 / (1.0 - self_bptdf))
        self.lodf = (
            self.bptdf * outage_scaling[np.newaxis, :]
        )  #: matrix of line outage distribution factors
        np.fill_diagonal(self.lodf, -1.0)
        self.lodf[:, self.bridge_mask] = 0.0
        self.lodf[self.bridge_mask, :] = 0.0

        # Reactance distance
        self_reactance = np.diag(self.inv_grounded_nodal_susceptance)
        self.reactance_distance = (
            self_reactance[:, np.newaxis]
            + self_reactance[np.newaxis, :]
            - 2 * self.inv_grounded_nodal_susceptance
        )  #: total aggregate reactance between every pair of nodes

    def element_branch_rows(self, table: str) -> np.ndarray:
        """Internal branch rows of every element of a branch table.

        :param table: A Pandapower or PyPSA branch table name.
        :return: An array of shape (elements, windings), -1 where the element is not in service.
        """
        keys = self.windings.get(table, [table])
        if keys[0] not in self.ppc_branch_indices:
            return np.empty((0, len(keys)), dtype=int)
        return np.column_stack(
            [self.ppc_to_internal_row[self.ppc_branch_indices[key]] for key in keys]
        )

    def _solve_angles(
        self, inv_susceptance: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        return (
            inv_susceptance
            @ (p[self.pv_pq_mask] + self.phase_shift_p[self.pv_pq_mask])
            * 180
            / np.pi
        )

    def solve_intact_angles(self, p: np.ndarray) -> np.ndarray:
        """Solve for intact angles using the inverted grounded nodal susceptance matrix.

        :param p: array of nodal power injections
        """
        self.angle[self.pv_pq_mask] = self._solve_angles(
            self.inv_grounded_nodal_susceptance, p
        )
        return self.angle

    def _contingency_susceptances(self, outage_index):
        susceptance_outage = self.branch_susceptance.tolil()
        susceptance_outage[outage_index, outage_index] = 0
        susceptance_diff = -1 * sp.csc_array(
            np.array(np.diag(self.diag_susceptance[outage_index]))
        )
        return susceptance_outage.tocsc(), susceptance_diff

    @staticmethod
    def _low_rank_inversion(susceptance_diff, outage_identity):
        return sp.linalg.spsolve(susceptance_diff, outage_identity)

    def _outage_multiplication(self, outage_index):
        return (
            self.grounded_unit_p[:, outage_index].T
            @ self.inv_grounded_nodal_susceptance
        ) @ self.grounded_unit_p[:, outage_index]

    def _outage_inversion(self, susceptance_diff, outage_index, outage_identity):
        outage_inversion = sp.linalg.spsolve(
            sp.csc_array(
                self._low_rank_inversion(susceptance_diff, outage_identity)
                + self._outage_multiplication(outage_index)
            ),
            outage_identity,
        )
        if np.ndim(outage_inversion) <= 1:
            outage_inversion = np.asarray(outage_inversion).reshape(1, 1)
        return outage_inversion

    def _contingency_inversion(self, susceptance_diff, outage_index, outage_identity):
        return self.inv_grounded_nodal_susceptance - (
            (
                (
                    self.inv_grounded_nodal_susceptance
                    @ self.grounded_unit_p[:, outage_index]
                )
                @ self._outage_inversion(
                    susceptance_diff, outage_index, outage_identity
                )
            )
            @ (
                self.grounded_unit_p[:, outage_index].T
                @ self.inv_grounded_nodal_susceptance
            )
        )

    def contingency_ptdf(
        self, outage_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate PTDFs under a given contingency.

        :param outage_index: An array of indices which are disconnected during the contingency.
        :return: An adapted PTDF under the contingency.
        """
        branch_susceptance_outage, susceptance_diff = self._contingency_susceptances(
            outage_index
        )
        outage_identity = sp.csc_array(np.identity(n=susceptance_diff.shape[0]))
        inverted_grounded_nodal_susceptance_outage = self._contingency_inversion(
            susceptance_diff, outage_index, outage_identity
        )
        return (
            (branch_susceptance_outage.tocsr() @ self.grounded_incidence.T)
            @ inverted_grounded_nodal_susceptance_outage,
            inverted_grounded_nodal_susceptance_outage,
        )

    def aggregate_nodal_injections(self, p: np.ndarray) -> np.ndarray:
        pass
