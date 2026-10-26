"""
Smoke test for the Streamlit network viewer, on the newest pandapower network in ``models``.

Run with: python -m pytest tests -v
"""

import os
import re
import sys
import time
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import n1_screening  # noqa: E402
import network_viewer as viewer  # noqa: E402
import viewer_charts as charts  # noqa: E402

APP = os.path.join(_PROJECT_ROOT, "network_viewer.py")


@unittest.skipIf(not viewer.available_models(), "no network JSON in models/")
class NetworkViewerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = AppTest.from_file(APP, default_timeout=1800)
        cls.app.run()

    def _search_box(self):
        return next(s for s in self.app.selectbox if s.label == "Find a substation")

    def _model_box(self):
        return next(s for s in self.app.selectbox if s.label == "Network")

    def _load_flow(self):
        return next(r for r in self.app.radio if r.label == "Load flow")

    def _metric_labels(self):
        return [m.label for m in self.app.metric]

    def test_app_runs_without_error(self):
        self.assertEqual([e.value for e in self.app.exception], [])
        self.assertIn(viewer.APP_TITLE, "".join(m.value for m in self.app.markdown))
        self.assertEqual(self._metric_labels()[:4],
                         ["Demand", "Generation", "Data centres", "Violations"])
        self.assertEqual([t.label for t in self.app.tabs][:6],
                         ["Substation detail", "Network", "Demand", "Generation",
                          "Data centres", "Config"])
        for label in ["Scenario - year", "Scenario - FES", "Scenario - DC Connect"]:
            self.assertIn(label, "".join(m.value for m in self.app.markdown))

    def _list_search(self, key):
        return next(t for t in self.app.text_input if t.key == f"{key}_search")

    def _list_table(self, heading):
        # A selected substation's own demand and generation tables carry a kV column;
        # the Demand and Generation lists do not, and the data centre list's heading is unique.
        return next(df.value for df in self.app.dataframe
                    if heading in df.value.columns
                    and (heading == "Peak demand (MW)" or "kV" not in df.value.columns))

    def test_demand_generation_and_data_centre_lists_render(self):
        for heading in ["Demand (MW)", "Capacity (MW)", "Peak demand (MW)"]:
            self.assertGreaterEqual(len(self._list_table(heading).columns), 5, heading)
        centres = next(m for m in self.app.metric if m.label == "Data centres")
        self.assertEqual(len(self._list_table("Peak demand (MW)")),
                         int(centres.value.replace(",", "")))

    def test_list_search_and_sort_narrow_and_order_the_rows(self):
        demand = self._list_table("Demand (MW)")
        self.assertTrue(demand["Demand (MW)"].is_monotonic_decreasing)
        term = str(demand["Name"].iloc[0]).split()[0]
        self._list_search("demand").set_value(term).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        narrowed = self._list_table("Demand (MW)")
        self.assertLessEqual(len(narrowed), len(demand))
        self.assertIn(demand["Name"].iloc[0], narrowed["Name"].tolist())
        self.assertEqual(list(narrowed.columns), ["Name", "Node", "TO", "Demand (MW)", "In service"])
        self._list_search("demand").set_value("").run()
        sort = next(s for s in self.app.selectbox if s.key == "generation_sort")
        sort.set_value("Name A-Z").run()
        self.assertEqual([e.value for e in self.app.exception], [])
        names = self._list_table("Capacity (MW)")["Name"].astype(str).tolist()
        self.assertEqual(names, sorted(names))

    def test_list_filter_by_owner_keeps_only_that_owner(self):
        field = next(s for s in self.app.selectbox if s.key == "centres_filter")
        field.set_value("Transmission owner").run()
        self.assertEqual([e.value for e in self.app.exception], [])
        owner = next(s for s in self.app.selectbox if s.key == "centres_owner")
        owner.set_value(owner.options[0]).run()
        table = self._list_table("Peak demand (MW)")
        self.assertTrue(table["TO"].astype(str).str.contains(owner.options[0]).all())
        next(s for s in self.app.selectbox if s.key == "centres_filter").set_value("All").run()

    def test_network_tab_lists_every_element_kind(self):
        element = next(r for r in self.app.radio if r.key == "network_element")
        expected = {"Circuits": ["Name", "From", "To", "kV", "Type", "Length (km)"],
                    "Transformers": ["Name", "HV node", "LV node", "Type"],
                    "Nodes": ["Node", "kV", "Site"],
                    "Reactive compensation": ["Name", "Node", "kV", "Device", "Absorb (MVAr)",
                                              "Generate (MVAr)"]}
        for kind, columns in expected.items():
            next(r for r in self.app.radio if r.key == "network_element").set_value(kind).run()
            self.assertEqual([e.value for e in self.app.exception], [])
            table = next(df.value for df in self.app.dataframe
                         if list(df.value.columns)[:len(columns)] == columns)
            self.assertGreater(len(table), 0, kind)
        element = next(r for r in self.app.radio if r.key == "network_element")
        element.set_value("Circuits").run()
        circuits = next(df.value for df in self.app.dataframe
                        if list(df.value.columns)[:2] == ["Name", "From"])
        self.assertTrue(set(circuits["Type"].unique()) <= {"OHL", "Cable", "Composite",
                                                            "Zero Length"})

    def test_most_loaded_chart_bars_match_their_labels(self):
        # Parallel circuits share a name; Plotly would stack their bars on one category.
        circuits = pd.DataFrame({
            "name": ["A-B_Cable", "A-B_Cable", "C-D_OHL"], "kv": [400.0] * 3,
            "rating_mva": [1000.0] * 3, "p_mw": [920.0, 930.0, 500.0],
            "from_name": ["A", "A", "C"], "to_name": ["B", "B", "D"],
            "loading": [92.0, 93.0, 50.0]})
        fig = charts.most_loaded(circuits, viewer.LOADING_BANDS, 100.0)
        bar = next(t for t in fig.data if isinstance(t, go.Bar))
        self.assertEqual(len(set(bar.y)), 3)
        self.assertEqual(sorted(bar.x), [50.0, 92.0, 93.0])

    def test_n1_loading_charts_overlay_the_intact_results(self):
        circuits = pd.DataFrame({
            "name": ["A-B", "C-D", "E-F"], "kv": [400.0, 400.0, 275.0],
            "rating_mva": [1000.0] * 3, "p_mw": [600.0, 500.0, 400.0],
            "from_name": ["A", "C", "E"], "to_name": ["B", "D", "F"],
            "loading": [60.0, 50.0, 40.0], "n1_loading": [110.0, 70.0, 95.0],
            "n1_outage": ["X-Y", "A-B", "C-D"]})
        curve = charts.loading_cumulative(circuits, 100.0, "loading", "n1_loading")
        dashes = [t.line.dash for t in curve.data if isinstance(t, go.Scatter)]
        self.assertEqual(sorted(dashes), ["dash", "dash", "solid", "solid"])
        ranked = charts.most_loaded(circuits, viewer.LOADING_BANDS, 100.0, "n1_loading",
                                    intact_column="loading")
        bars = [t for t in ranked.data if isinstance(t, go.Bar)]
        self.assertEqual([b.name for b in bars], ["Intact network", "Worst N-1 outage"])
        self.assertEqual(list(bars[1].x), [70.0, 95.0, 110.0])  # ranked by N-1 loading
        self.assertEqual(list(bars[0].x), [50.0, 40.0, 60.0])  # the same circuits, intact
        self.assertEqual(ranked.layout.barmode, "group")
        # Neither the limit lines nor the bands carry a label.
        self.assertEqual(list(ranked.layout.annotations), [])
        self.assertEqual(list(curve.layout.annotations), [])
        self.assertEqual(len(curve.layout.shapes), 2)  # band and limit line
        buses = pd.DataFrame({
            "vn_kv": [400.0, 400.0, 275.0], "vm_pu": [1.00, 1.02, 0.98],
            "n1_min_vm_pu": [0.94, 1.00, 0.97], "n1_max_vm_pu": [1.03, 1.06, 1.01]})
        volts = charts.voltage_cumulative(buses, (0.95, 1.05), ["intact", "n1_low", "n1_high"])
        curves = [t for t in volts.data if t.hoverinfo != "skip"]  # not the legend-only traces
        self.assertEqual(sorted(t.line.dash for t in curves),
                         ["dash", "dash", "dot", "dot", "solid", "solid"])
        self.assertEqual(sorted(t.name for t in curves if "N-1" in t.name),
                         ["275kV N-1 highest", "275kV N-1 lowest", "400kV N-1 highest",
                          "400kV N-1 lowest"])
        # One legend entry per voltage level and one grey entry per post-outage series.
        self.assertEqual([t.name for t in volts.data if t.showlegend is not False],
                         ["400kV (2)", "275kV (1)", "N-1 lowest", "N-1 highest"])
        self.assertEqual(len(charts.voltage_cumulative(buses, (0.95, 1.05)).data), 2)

    def test_selective_resolve_picks_the_outages_that_raise_loading(self):
        cases = [("line", 1), ("line", 2), ("trafo", 1)]
        screen = {
            "method": "LODF", "cases": 3, "re-solved outages": 0,
            "unsolved": [{"outage": "Line C", "reason": "islanded"}],
            "case peak loading": np.array([95.0, np.nan, 80.0]),
            "branches": {"line": pd.DataFrame(
                {"max_loading_percent": [95.0, 70.0], "p_mw": [100.0, 50.0],
                 "cause": ["Line A", "Line B"]}, index=[1, 2])},
            "buses": None,
        }
        self.assertEqual(n1_screening.cases_loading_above(screen, cases, 90.0), [("line", 1)])
        self.assertEqual(n1_screening.cases_loading_above(screen, cases, 75.0),
                         [("line", 1), ("trafo", 1)])
        buses = pd.DataFrame({"min_vm_pu": [0.93]}, index=[7])
        resolved = {
            "cases": 1, "requested": 1, "stopped": False,
            "unsolved": [{"outage": "Line C", "reason": "islanded"}],
            "branches": {"line": pd.DataFrame(
                {"max_loading_percent": [90.0, 75.0], "p_mw": [96.0, 55.0],
                 "cause": ["Line A", "Line A"]}, index=[1, 2])},
            "buses": buses,
        }
        merged = n1_screening.combine_with_resolved(screen, resolved, 90.0)
        line = merged["branches"]["line"]
        # The higher loading wins per element, with its own cause and flow.
        self.assertEqual(list(line["max_loading_percent"]), [95.0, 75.0])
        self.assertEqual(list(line["cause"]), ["Line A", "Line A"])
        self.assertEqual(list(line["p_mw"]), [100.0, 55.0])
        self.assertIs(merged["buses"], buses)
        self.assertEqual(merged["method"], "LODF + load flow")
        self.assertEqual(len(merged["unsolved"]), 1)  # the same outage is listed once
        self.assertEqual((merged["re-solve candidates"], merged["re-solved outages"]), (1, 1))

    def test_n1_job_stops_early_and_keeps_the_outages_solved_so_far(self):
        class SlowModel:
            def n1_case_count(self, voltage_levels_kv=None, from_buses=None):
                return 50

            def run_n1(self, mode, progress, full_resolve=False, resolve_above=None, stop=None,
                       voltage_levels_kv=None, from_buses=None):
                done = 0
                for done in range(1, 51):
                    if stop():
                        done -= 1
                        break
                    time.sleep(0.01)
                    progress(done, 50, f"case {done}")
                return {"cases": done, "requested": 50, "stopped": done < 50,
                        "mode": mode, "full": full_resolve, "above": resolve_above}

        selection = viewer.N1Selection((400.0, 275.0), (), None)
        job = viewer.N1Job(("key",), SlowModel(), "AC", True, 90.0, selection)
        while job.done < 5:
            time.sleep(0.005)
        self.assertTrue(job.running)
        self.assertIn("N-1 contingency", job.text)
        job.stop()
        self.assertIn("Stopping", job.text)
        while job.running:
            time.sleep(0.005)
        self.assertIsNone(job.error)
        self.assertTrue(job.result["stopped"])
        self.assertGreaterEqual(job.result["cases"], 5)
        self.assertLess(job.result["cases"], 50)
        self.assertEqual((job.result["mode"], job.result["full"], job.result["above"]),
                         ("AC", True, 90.0))

    def test_sidebar_offers_the_selective_resolve(self):
        n1_box = next(c for c in self.app.checkbox if c.label.startswith("Run N-1"))
        if n1_box.value:  # left ticked by an earlier test on the shared app
            n1_box.uncheck().run()
        resolve = next(r for r in self.app.radio if r.label == "Re-solve with the AC load flow")
        self.assertEqual(resolve.options,
                         [viewer.RESOLVE_NONE, viewer.RESOLVE_ABOVE, viewer.RESOLVE_ALL])
        self.assertTrue(resolve.disabled)  # until N-1 is ticked on the AC load flow
        threshold = next(n for n in self.app.number_input if n.label == "Re-solve threshold (%)")
        self.assertEqual(threshold.value, 90)
        self.assertTrue(threshold.disabled)

    def test_selective_ac_resolve_runs_through_the_app(self):
        # A high threshold keeps the AC re-solve stage short, so the whole path (screen,
        # selection, worker thread, merge and summary) runs in a few minutes at most.
        box = lambda label: next(c for c in self.app.checkbox if c.label.startswith(label))
        resolve = lambda: next(r for r in self.app.radio
                               if r.label == "Re-solve with the AC load flow")
        self._load_flow().set_value("AC").run()
        box("Run N-1").check().run()
        self.assertFalse(resolve().disabled)  # the subset can be chosen straight away
        next(n for n in self.app.number_input
             if n.label == "Re-solve threshold (%)").set_value(200).run()
        resolve().set_value(viewer.RESOLVE_ABOVE).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        text = "".join(c.value for c in self.app.caption)
        match = re.search(r"outages screened with LODFs; ([\d,]+) of the ([\d,]+) that raise", text)
        self.assertIsNotNone(match, text)
        self.assertEqual(match.group(1), match.group(2))  # none skipped: no stop was pressed
        self.assertIn("N-1 outages run", self._metric_labels())
        self.assertFalse([w.value for w in self.app.warning if "stopped" in w.value])
        resolve().set_value(viewer.RESOLVE_NONE).run()
        next(n for n in self.app.number_input
             if n.label == "Re-solve threshold (%)").set_value(90).run()
        box("Run N-1").uncheck().run()
        self._load_flow().set_value("DC").run()
        self.assertEqual([e.value for e in self.app.exception], [])

    def test_every_list_offers_the_same_filters(self):
        for box in [s for s in self.app.selectbox if s.label == "Filter by"]:
            self.assertEqual(box.options, ["All", "FLOP zone", "Transmission owner",
                                           "In service"], box.key)
        # Widget handles go stale after each run, so they are re-fetched.
        key = lambda kind: "network_" + kind.split()[0].lower() + "_filter"
        box = lambda kind: next(s for s in self.app.selectbox if s.key == key(kind))
        for kind in next(r for r in self.app.radio if r.key == "network_element").options:
            next(r for r in self.app.radio if r.key == "network_element").set_value(kind).run()
            for label in box(kind).options[1:]:
                with self.subTest(element=kind, filter=label):
                    box(kind).set_value(label).run()
                    self.assertEqual([e.value for e in self.app.exception], [])
            box(kind).set_value("All").run()

    def test_elements_to_look_at_use_the_watch_threshold(self):
        limits = viewer.Limits(100.0, (0.95, 1.05))
        self.assertEqual(limits.watch, 90.0)
        self.assertEqual(viewer.Limits(80.0, (0.95, 1.05)).watch, 80.0)

    def test_limits_are_fixed(self):
        self.assertEqual((viewer.LIMITS.loading, viewer.LIMITS.vm), (100.0, (0.90, 1.05)))
        self.assertEqual([s.label for s in self.app.slider if "threshold" in s.label.lower()], [])
        self.assertEqual([s.label for s in self.app.slider if "voltage" in s.label.lower()], [])

    def test_about_page_sets_out_the_model_and_its_limits(self):
        page = next(r for r in self.app.radio if r.key == "page")
        page.set_value(viewer.ABOUT_PAGE).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        text = "".join(m.value for m in self.app.markdown)
        for phrase in ["Data sources", "Model capabilities", "Outputs and visualisation",
                       "NESO", "N-1", "NETS SQSS v2.14", "0.90pu to 1.05pu", "Short-term"]:
            self.assertIn(phrase, text)
        for phrase in ["PTDF", "LODF", "5% steps", "coupler"]:  # an overview, not a method
            self.assertNotIn(phrase, text)
        self.assertEqual([e.label for e in self.app.expander],
                         ["Network limits and assumptions"])
        self.assertFalse(self.app.expander[0].proto.expanded)  # collapsed
        self.assertNotIn("| Basis |", text)  # the SQSS source is a footnote to the value
        self.assertIn("\u00b9 NETS SQSS", text)
        self.assertNotIn(".json", text)
        self.assertNotIn(".nc", text)
        self.assertEqual([t.label for t in self.app.tabs], [])  # the viewer body is replaced
        next(r for r in self.app.radio if r.key == "page").set_value(viewer.NETWORK_PAGE).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        self.assertIn("Substation detail", [t.label for t in self.app.tabs])

    def test_config_tab_labels_the_scenario_cards(self):
        fes = next(s for s in self.app.selectbox if s.key == "config_fes")
        self.assertIsNone(fes.value)
        fes.set_value(fes.options[0]).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        text = "".join(m.value for m in self.app.markdown)
        self.assertIn(fes.options[0], text)
        year = next(s for s in self.app.slider if s.key == "config_year")
        self.assertEqual(year.value, viewer._model_year(self._model_box().value))

    def test_selecting_a_substation_shows_its_components(self):
        box = self._search_box()
        box.set_value(next(o for o in box.options if o != "(none)")).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        labels = [t.label for t in self.app.tabs]
        self.assertTrue(any(label.startswith("Nodes") for label in labels), labels)
        self.assertTrue(any(label.startswith("Branches") for label in labels), labels)

    def test_map_click_hand_off_lands_in_the_search_box(self):
        # AppTest cannot fire a Plotly click, so this exercises the click handler's hand-off:
        # the label is stashed in a pending key and copied into the search box on the rerun.
        box = self._search_box()
        target = next(o for o in box.options if o != "(none)")
        box.set_value("(none)").run()
        self.app.session_state["pending_substation_choice"] = target
        self.app.run()
        self.assertEqual([e.value for e in self.app.exception], [])
        self.assertEqual(self._search_box().value, target)
        self.assertNotIn("pending_substation_choice", self.app.session_state)
        labels = [t.label for t in self.app.tabs]
        self.assertTrue(any(label.startswith("Nodes") for label in labels), labels)
        self._search_box().set_value("(none)").run()

    def test_load_flow_modes_all_render_for_every_model_type(self):
        # The newest network of each file type.
        newest = {}
        for path in self._model_box().options:
            newest.setdefault(os.path.splitext(path)[1], path)
        for path in newest.values():
            self._model_box().set_value(path).run()
            for mode in ["DC", "None", "AC"]:
                with self.subTest(model=os.path.basename(path), mode=mode):
                    self._load_flow().set_value(mode).run()
                    self.assertEqual([e.value for e in self.app.exception], [])
                    self.assertIn("Data centres", self._metric_labels())

    def test_n1_screen_runs_for_every_model_type(self):
        # The LODF screen takes seconds, so it is run on the DC flows of each model type.
        newest = {}
        for path in self._model_box().options:
            newest.setdefault(os.path.splitext(path)[1], path)
        for path in newest.values():
            with self.subTest(model=os.path.basename(path)):
                self._model_box().set_value(path).run()
                self._load_flow().set_value("DC").run()
                n1_box = next(c for c in self.app.checkbox
                              if c.label.startswith("Run N-1"))
                n1_box.check().run()
                self.assertEqual([e.value for e in self.app.exception], [])
                labels = self._metric_labels()
                for label in ["N-1 outages run", "N-1 outages unsolved", "N-1 overloads"]:
                    self.assertIn(label, labels)
                run = next(m for m in self.app.metric if m.label == "N-1 outages run")
                self.assertGreater(int(run.value.replace(",", "")), 0)
                n1_box.uncheck().run()
                self.assertEqual([e.value for e in self.app.exception], [])

    def test_screening_a_data_centre_lists_the_branches_that_move(self):
        self._load_flow().set_value("DC").run()
        centres = next((s for s in self.app.selectbox if s.label == "Data centre"), None)
        if centres is None:
            self.skipTest("no data centre (type DC) load in the newest model")
        next(b for b in self.app.button if b.label == "Screen affected circuits").click().run()
        self.assertEqual([e.value for e in self.app.exception], [])
        labels = self._metric_labels()
        for label in ["Circuits", "Transformers", "Largest rise"]:
            self.assertIn(label, labels)
        # The screen table alone has the flow-change columns; a selected substation's
        # branch table also gains a Change (points) column.
        table = next(df.value for df in self.app.dataframe
                     if "Flow change (MW)" in df.value.columns)
        self.assertGreater(len(table), 0)
        self.assertTrue(table["Change (points)"].is_monotonic_decreasing)
        criterion = next(r for r in self.app.radio if r.label == "Affected by")
        criterion.set_value(criterion.options[1]).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        table = next(df.value for df in self.app.dataframe
                     if "Flow change (% of step)" in df.value.columns)
        self.assertTrue(table["Flow change (% of step)"].is_monotonic_decreasing)
        next(b for b in self.app.button if b.label == "Clear screen").click().run()
        self.assertEqual([e.value for e in self.app.exception], [])
        self.assertNotIn("Largest rise", self._metric_labels())

    def test_owner_filter_narrows_the_elements_to_look_at(self):
        owners = next(m for m in self.app.multiselect if m.label == "Transmission owner")
        unfiltered = [df.value.shape[0] for df in self.app.dataframe]
        owners.set_value([owners.options[0]]).run()
        self.assertEqual([e.value for e in self.app.exception], [])
        filtered = [df.value.shape[0] for df in self.app.dataframe]
        self.assertTrue(all(f <= u for f, u in zip(filtered, unfiltered)),
                        f"{filtered} should not exceed {unfiltered}")
        owners.set_value([]).run()

    def test_load_flow_status_line_is_short(self):
        self._load_flow().set_value("DC").run()
        captions = [c.value for c in self.app.caption]
        self.assertTrue(any(c.startswith("DC load flow: Successful at 100%")
                            or c.startswith("DC load flow: Unsuccessful") for c in captions),
                        captions)
        self.assertFalse(any(s.label == "FLOP zone" for s in self.app.selectbox))


if __name__ == "__main__":
    unittest.main()
