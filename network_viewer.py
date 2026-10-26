"""
Interactive Streamlit map of a pandapower GB transmission network model, with its load-flow results.
Run with: streamlit run network_viewer.py

Network models are read from the ``models`` folder next to this file.
"""

import glob
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageChops

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import model_adapters  # noqa: E402
import viewer_charts as charts  # noqa: E402

MODEL_DIR = os.path.join(_HERE, "models")
APP_TITLE = "DCConnect Network Visualiser"
APP_SUBTITLE = "GB transmission network capacity viewer for data centre connections"
LOGO_DIR = _HERE
LOGOS = ["national_grid_logo.png", "tnei_logo.png"]
LOGO_HEIGHT = 40
COASTLINE_FILE = os.path.join(LOGO_DIR, "coastline.geojson")
BOUNDARY_FILE = os.path.join(LOGO_DIR, "etys_boundaries_mar25.geojson")
ZONE_FILE = os.path.join(LOGO_DIR, "dcconnect_flop_zones.geojson")
LAND_COLOUR, COAST_COLOUR, SEA_COLOUR = "#F1EFE9", "#8C95A5", "#DCE8F2"

BRAND = charts.BRAND
LOADING_BANDS = [
    (5.0, "below 5%", "#008000"),
    (50.0, "5 - 50%", BRAND["light_blue"]),
    (80.0, "50 - 80%", BRAND["mid_blue"]),
    (100.0, "80 - 100%", BRAND["dark_blue"]),
    (float("inf"), "above 100%", charts.ALERT_COLOUR),
]
WATCH_LOADING = 90.0  # branches above this are listed under "Elements to look at"
LOADING_LIMIT = 100.0  # of the continuous winter rating; short-term ratings are not used
VM_LIMITS = (0.90, 1.05)  # -10%/+5% of nominal at every voltage level, see VOLTAGE_LIMIT_SOURCE
VOLTAGE_LIMIT_SOURCE = ("NETS SQSS v2.14 Table 6.2, Steady State Voltage Limits and "
                        "Requirements in Planning Timescales")
NETWORK_PAGE, ABOUT_PAGE = "Network", "About"
RESOLVE_NONE, RESOLVE_ABOVE, RESOLVE_ALL = ("None (LODF screen only)",
                                            "Outages above the loading threshold",
                                            "Every outage (slow)")
UNSOLVED_COLOUR = charts.LIGHT_GREY
SUBSTATION_COLOUR = BRAND["grey"]
PROBLEM_COLOUR = charts.ALERT_COLOUR
ARROW_COLOUR = BRAND["dark_blue"]
SELECTED_COLOUR = BRAND["light_blue"]
BOUNDARY_COLOUR = BRAND["mustard"]
ZONE_COLOUR = "#8E6BB5"
AFFECTED_COLOUR = charts.AFFECTED_COLOUR
DC_HEAT_SCALE = [[0.0, "#F3E7B5"], [0.5, BRAND["mustard"]], [1.0, "#7A5E10"]]
DC_HALOS = [(2.6, 0.10), (1.8, 0.18), (1.0, 0.80)]  # (size multiple, opacity), outer glow to core

GB_LON_RANGE = [-8.5, 2.5]
GB_LAT_RANGE = [49.5, 61.0]
SITE_HALF_SPAN_DEG = 0.4
MAP_ASPECT = 1.0  # width:height of the view box; the longitude span is derived from the latitude span
MAP_HEIGHT = 730  # map area plus MAP_LEGEND_MARGIN
MAP_LEGEND_MARGIN = 160  # room under the map for the legend columns
MAP_LEGEND_COLUMNS = {"legend": "Loading", "legend2": "Sites and Boundaries"}
MAP_HELP = ("Click a substation marker, or use the search box, to see its detail.<br>"
            "Hover a circuit for its rating and loading.<br>"
            "Arrow heads point in the direction of active power flow.<br>"
            "Gold glows are data centres sized by peak demand.")
SIDEBAR_WIDTH_PX = 260
MAIN_COLUMNS = [1.25, 1]  # map : tabs
N1_ROW_STYLE = f"background-color: {charts.TINTS['n1']}"
AFFECTED_ROW_STYLE = f"background-color: {charts.TINTS['affected']}"
MAX_ISSUE_ROWS = 200
DEFAULT_STEP_MW = 100
NO_SELECTION = "(none)"
UNAVAILABLE = "—"  # metric value when the analysis behind it has not been run
NOT_SET = "Not set"

# Scenario labels for the Config tab; they label the view only, the network shown is the file
# picked in the sidebar.
YEAR_RANGE = (2025, 2035)
DEFAULT_YEAR = 2030
FES_SCENARIOS = ["Holistic Transition", "Electric Engagement", "Hydrogen Evolution",
                 "Counterfactual"]
DC_SCENARIOS = ["Gold Rush"]
LIST_HEIGHT = 420

PAGE_CSS = f"""
<style>
.block-container {{padding-top: 4rem; padding-bottom: 2rem;}}
[data-testid="stSidebar"] {{width: {SIDEBAR_WIDTH_PX}px !important; min-width: {SIDEBAR_WIDTH_PX}px !important;}}
.app-header {{display: flex; flex-direction: column; justify-content: center;}}
[data-testid="stMarkdownContainer"] p.app-title {{font-size: 1.85rem; font-weight: 700; color: {BRAND["dark_blue"]};
             margin: 0; line-height: 1.15; letter-spacing: -0.01em;}}
[data-testid="stMarkdownContainer"] p.app-subtitle {{color: {BRAND["grey"]}; font-size: 0.95rem; margin: 0.15rem 0 0 0;}}
.app-rule {{border: 0; border-top: 3px solid {BRAND["teal"]}; margin: 0.6rem 0 1.0rem 0;}}
[data-testid="stMetricLabel"] p {{font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.03em;
                                  color: {BRAND["grey"]}; font-weight: 600;}}
[data-testid="stMetricValue"] {{font-size: 1.55rem; font-weight: 600; color: {BRAND["dark_blue"]};}}
[data-testid="stSidebar"] h2 {{font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
                               color: {BRAND["grey"]}; padding-top: 0; padding-bottom: 0.2rem;}}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{padding: 0.6rem 0.75rem 0.2rem 0.75rem;}}
h3 {{font-size: 1.1rem !important; color: {BRAND["dark_blue"]}; padding-bottom: 0.2rem !important;}}
.stTabs [data-baseweb="tab"] p {{font-weight: 600;}}
[data-testid="stMarkdownContainer"] p.panel-note {{color: {BRAND["grey"]}; font-size: 0.85rem;}}
[data-testid="stMarkdownContainer"] p.scenario-label {{font-size: 0.7rem; text-transform: uppercase;
             letter-spacing: 0.03em; color: {BRAND["grey"]}; font-weight: 600; margin: 0;}}
[data-testid="stMarkdownContainer"] p.scenario-value {{font-size: 1.2rem; font-weight: 700;
             text-transform: uppercase; margin: 0.1rem 0 0 0; line-height: 1.2;}}
[data-testid="stMarkdownContainer"] p.list-title {{font-size: 1.0rem; font-weight: 700;
             color: {BRAND["dark_blue"]}; margin: 0;}}
</style>
"""

CRITERIA = {
    "Loading rise (points of rating)": dict(
        column="delta", label="Affected if loading rises by at least (points)",
        limits=(1, 25), default=5,
        help="Rise in loading, in percentage points of rating."),
    "Flow rise (% of the added MW)": dict(
        column="delta_share", label="Affected if flow rises by at least (% of added MW)",
        limits=(1, 100), default=5,
        help="Rise in active power flow, as a share of the added MW."),
}
THERMAL_COLUMNS = {
    "kind": "Type", "name": "Name", "loading": "Loading (%)", "condition": "Condition",
    "outage": "Outage", "kv": "kV", "rating_mva": "Rating (MVA)", "p_mw": "Flow (MW)",
    "owner": "TO"}
VOLTAGE_COLUMNS = {
    "name": "Node", "vm_pu": "V (pu)", "condition": "Condition", "outage": "Outage",
    "vn_kv": "kV", "zone": "Zone", "owner": "TO"}
SCREEN_COLUMNS = {
    "kind": "Type", "name": "Name", "connects": "Connects", "kv": "kV",
    "rating_mva": "Rating (MVA)", "loading": "Before (%)", "step_loading": "After (%)",
    "delta": "Change (points)", "delta_mw": "Flow change (MW)",
    "delta_share": "Flow change (% of step)", "owner": "TO"}
LOAD_COLUMNS = {
    "name": "Name", "node_name": "Node", "vn_kv": "kV", "site": "Site", "zone": "Zone",
    "owner": "TO", "p_mw": "Demand (MW)", "demand_type": "Type", "in_service": "In service"}
GENERATOR_COLUMNS = {
    "name": "Name", "node_name": "Node", "vn_kv": "kV", "site": "Site", "zone": "Zone",
    "owner": "TO", "capacity_mw": "Capacity (MW)", "dispatch_mw": "Dispatched (MW)",
    "plant_type": "Plant type", "model": "Model", "in_service": "In service"}
DATA_CENTRE_COLUMNS = {
    "name": "Name", "node_name": "Node", "vn_kv": "kV", "site": "Site", "site_name": "Site name",
    "zone": "Zone", "owner": "TO", "p_mw": "Peak demand (MW)", "in_service": "In service"}
DEMAND_LIST_COLUMNS = {
    "name": "Name", "node_name": "Node", "owner": "TO", "p_mw": "Demand (MW)",
    "in_service": "In service"}
GENERATION_LIST_COLUMNS = {
    "name": "Name", "node_name": "Node", "capacity_mw": "Capacity (MW)",
    "dispatch_mw": "Dispatched (MW)", "plant_type": "Plant type", "in_service": "In service"}
COMPENSATION_COLUMNS = {
    "name": "Name", "node_name": "Node", "vn_kv": "kV", "device": "Device",
    "absorb_mvar": "Absorb (MVAr)", "generate_mvar": "Generate (MVAr)", "owner": "TO",
    "in_service": "In service"}
CIRCUIT_COLUMNS = {
    "name": "Name", "from_name": "From", "to_name": "To", "kv": "kV", "subtype": "Type",
    "length_km": "Length (km)", "rating_mva": "Rating (MVA)", "loading": "Loading (%)",
    "p_mw": "Flow (MW)", "zone": "Zone", "owner": "TO", "in_service": "In service"}
TRANSFORMER_COLUMNS = {
    "name": "Name", "from_name": "HV node", "to_name": "LV node", "subtype": "Type",
    "rating_mva": "Rating (MVA)", "loading": "Loading (%)", "p_mw": "Flow (MW)", "zone": "Zone",
    "owner": "TO", "in_service": "In service"}
NODE_COLUMNS = {
    "name": "Node", "vn_kv": "kV", "site": "Site", "site_name": "Site name", "zone": "Zone",
    "owner": "TO", "vm_pu": "V (pu)", "in_service": "In service"}
RESULT_COLUMNS = ("loading", "p_mw", "vm_pu", "dispatch_mw")  # need a load flow
LIST_FILTERS = {"FLOP zone": "zone", "Transmission owner": "owner", "In service": "in_service"}
SITE_COLUMNS = ("site", "site_name", "zone", "owner")  # redundant in a substation's own detail
ISSUE_COLUMN_CONFIG = {
    "Loading (%)": st.column_config.NumberColumn(width="small", format="%.1f"),
    "Demand (MW)": st.column_config.NumberColumn(format="%.1f"),
    "Peak demand (MW)": st.column_config.NumberColumn(format="%.1f"),
    "Capacity (MW)": st.column_config.NumberColumn(format="%.1f"),
    "Dispatched (MW)": st.column_config.NumberColumn(format="%.1f"),
    "In service": st.column_config.CheckboxColumn(width="small"),
    "Length (km)": st.column_config.NumberColumn(width="small", format="%.1f"),
    "Absorb (MVAr)": st.column_config.NumberColumn(format="%.0f"),
    "Generate (MVAr)": st.column_config.NumberColumn(format="%.0f"),
    "Before (%)": st.column_config.NumberColumn(width="small", format="%.1f"),
    "After (%)": st.column_config.NumberColumn(width="small", format="%.1f"),
    "Change (points)": st.column_config.NumberColumn(width="small", format="%+.2f"),
    "Flow change (MW)": st.column_config.NumberColumn(width="small", format="%+.1f"),
    "Flow change (% of step)": st.column_config.NumberColumn(width="small", format="%+.1f"),
    "V (pu)": st.column_config.NumberColumn(width="small", format="%.4f"),
    "Condition": st.column_config.TextColumn(width="small"),
    "kV": st.column_config.NumberColumn(width="small", format="%.0f"),
    "Rating (MVA)": st.column_config.NumberColumn(width="small", format="%.0f"),
    "Flow (MW)": st.column_config.NumberColumn(width="small", format="%.0f"),
    "TO": st.column_config.TextColumn(width="small"),
    "Zone": st.column_config.TextColumn(width="small"),
}


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------

@st.cache_resource
def trimmed_logo(path):
    image = Image.open(path).convert("RGBA")
    on_white = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    content = ImageChops.difference(on_white, Image.new("RGB", image.size, "white")).getbbox()
    return image.crop(content) if content else image


@st.cache_resource
def coastline_rings():
    """The land outline as one None-separated (lon, lat) polyline. coastline.geojson is
    Natural Earth's 1:10m countries clipped to GB; Plotly's built-in outlines stop at 1:50m."""
    with open(COASTLINE_FILE, encoding="utf-8") as handle:
        countries = json.load(handle)["features"]
    lon, lat = [], []
    for country in countries:
        geometry = country["geometry"]
        polygons = (geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                    else [geometry["coordinates"]])
        for rings in polygons:
            for ring in rings:
                lon.extend([x for x, _y in ring] + [None])
                lat.extend([y for _x, y in ring] + [None])
    return lon, lat


@st.cache_resource
def geojson_lines(path):
    """Every line and polygon ring in a geojson file (the ETYS boundaries, the FLOP zones)
    as one None-separated (lon, lat) polyline, with the feature's name at every vertex."""
    with open(path, encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    lon, lat, names = [], [], []
    for feature in features:
        geometry = feature["geometry"]
        if geometry["type"] == "LineString":
            lines = [geometry["coordinates"]]
        elif geometry["type"] == "Polygon":
            lines = geometry["coordinates"]
        else:  # MultiPolygon
            lines = [ring for polygon in geometry["coordinates"] for ring in polygon]
        for points in lines:
            lon.extend([x for x, _y in points] + [None])
            lat.extend([y for _x, y in points] + [None])
            names.extend([feature["properties"]["name"]] * len(points) + [None])
    return lon, lat, names


def available_models():
    paths = [p for ext in model_adapters.MODEL_EXTENSIONS
             for p in glob.glob(os.path.join(MODEL_DIR, f"*{ext}"))]
    return sorted(paths, key=os.path.getmtime, reverse=True)


@st.cache_resource(show_spinner="Loading network and running load flow...")
def load_model(path, _mtime, mode):
    # _mtime only invalidates the cache when the file changes.
    model = model_adapters.load_model(path)
    if mode != "None":
        model.solve(mode)
    return model


@st.cache_resource
def result_store():
    # Shared across sessions and reruns: N-1 screens and demand steps run once per key.
    return {}


# --------------------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Limits:
    loading: float
    vm: tuple

    @property
    def low(self):
        return self.vm[0]

    @property
    def high(self):
        return self.vm[1]

    @property
    def watch(self):
        """The 'Elements to look at' threshold, so branches close to the limit are listed."""
        return min(WATCH_LOADING, self.loading)

    def overloaded(self, loading):
        return loading > self.loading

    def watched(self, loading):
        return loading > self.watch

    def outside_band(self, vm):
        return (vm < self.low) | (vm > self.high)


LIMITS = Limits(LOADING_LIMIT, VM_LIMITS)


@dataclass(frozen=True)
class Controls:
    page: str
    path: str
    mode: str
    limits: Limits
    run_n1: bool
    full_resolve: bool
    resolve_above: float | None  # re-solve only outages the screen loads above this (%)


@dataclass(frozen=True)
class Filters:
    kv: list
    owners: list
    show_arrows: bool


@dataclass(frozen=True)
class N1Selection:
    """The outages an N-1 run covers, from the sidebar filters: ``from_buses`` are the buses
    of the chosen owners, or None when no owner is chosen."""
    kv: tuple
    owners: tuple
    from_buses: tuple | None

    @property
    def description(self):
        levels = " and ".join(f"{kv:.0f}kV" for kv in self.kv) or "no voltage level"
        owned = f" owned by {' or '.join(self.owners)}" if self.owners else ""
        return f"{levels} circuits and transformers{owned}"


def n1_selection(case, filters):
    kv = tuple(sorted((round(float(v), 1) for v in filters.kv), reverse=True))
    owners = tuple(filters.owners)
    from_buses = None
    if owners:
        from_buses = tuple(case.buses.index[owner_mask(case.buses["owner"], owners)])
    return N1Selection(kv, owners, from_buses)


@dataclass(frozen=True)
class Scenario:
    year: int
    fes: str
    dc: str


@dataclass(frozen=True)
class ListSpec:
    """A filterable, sortable, searchable list: ``columns`` maps source columns to headings,
    ``filters`` labels to the column they pick a value of, ``sorts`` labels to (column,
    ascending)."""
    title: str
    columns: dict
    filters: dict
    sorts: dict
    search: tuple
    total: str = None  # column summed in MW under the table


@dataclass(frozen=True)
class ScreenHighlight:
    affected: pd.DataFrame
    centre: pd.Series
    step_mw: float
    threshold: float
    criterion_text: str


def owner_names(buses):
    return sorted({part.strip() for value in buses["owner"].dropna()
                   for part in str(value).split(",") if part.strip()})


def owner_mask(owners, chosen):
    if not chosen:
        return pd.Series(True, index=owners.index)
    selected = set(chosen)
    # astype(bool): on an empty table the mapped mask would be object-typed, which pandas
    # reads as a column selection rather than a row filter.
    return owners.fillna("").map(
        lambda value: bool(selected & {part.strip() for part in str(value).split(",")})
    ).astype(bool)


def _plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _model_year(path):
    """The first year in a Step 2 file name (FULL_GRID_2030_...), else the default."""
    years = [int(y) for y in re.findall("(?<![0-9])(20[0-9][0-9])(?![0-9])", os.path.basename(path))]
    years = [y for y in years if YEAR_RANGE[0] <= y <= YEAR_RANGE[1]]
    return years[0] if years else DEFAULT_YEAR


def _option_label(value):
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _loading_band(loading):
    if pd.isna(loading):
        return len(LOADING_BANDS)
    for band, (limit, _label, _colour) in enumerate(LOADING_BANDS):
        if loading <= limit:
            return band
    return len(LOADING_BANDS) - 1


# --------------------------------------------------------------------------------------
# The network as tables
# --------------------------------------------------------------------------------------

class NetworkCase:
    """One model in one load-flow mode: the tables the page draws, extended in place when
    an N-1 screen or a demand step has run."""

    def __init__(self, model, path, mtime, mode):
        self.model = model
        self.mode = mode
        self.key = (path, mtime, mode)
        self.n1 = None
        self.buses = model.buses()
        self.branches = self._branch_table()
        self.subs = self._substation_table()
        self.loads = self._with_bus_attributes(model.loads())
        self.generators = self._with_bus_attributes(model.generators())
        self.compensation = self._with_bus_attributes(model.compensation())
        self.centres = self._data_centre_table()

    @property
    def solved(self):
        return self.model.solved

    @property
    def summary(self):
        return self.model.summary

    @property
    def has_n1_voltages(self):
        return (self.n1 is not None and self.n1["buses"] is not None
                and bool(self.n1["buses"]["min_vm_pu"].notna().any()))

    @property
    def has_screen(self):
        return "delta" in self.branches.columns

    def _branch_table(self):
        buses, branches = self.buses, self.model.branches()
        for end in ("from", "to"):
            for column in ("name", "site", "lon", "lat"):
                branches[f"{end}_{column}"] = buses[column].reindex(branches[f"{end}_bus"]).values
        branches["kv"] = buses["vn_kv"].reindex(branches["from_bus"]).values
        for column in ("zone", "owner"):
            branches[column] = buses[column].reindex(branches["from_bus"]).values
        branches["mappable"] = (
            branches[["from_lon", "from_lat", "to_lon", "to_lat"]].notna().all(axis=1)
            & (branches["from_site"] != branches["to_site"])
        )
        return branches

    def _substation_table(self):
        buses, branches = self.buses, self.branches
        located = buses[buses["lon"].notna()]
        subs = located.groupby("site").agg(
            name=("site_name", "first"),
            zone=("zone", "first"),
            owner=("owner", "first"),
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            max_kv=("vn_kv", "max"),
            buses=("name", "count"),
            min_vm_pu=("vm_pu", "min"),
            max_vm_pu=("vm_pu", "max"),
        )
        injections = self.model.bus_injections().groupby(buses["site"]).sum()
        for column in ("demand_mw", "generation_mw"):
            subs[column] = injections[column].reindex(subs.index).fillna(0.0)

        for kind, column in [("Line", "circuits"), ("Transformer", "transformers")]:
            subset = branches[branches["kind"] == kind]
            ends = pd.concat([
                subset[["from_site", "loading"]].rename(columns={"from_site": "site"}),
                subset[["to_site", "loading"]].rename(columns={"to_site": "site"}),
            ], keys=["from", "to"]).reset_index(level=1, names=["", "branch"])
            ends = ends.drop_duplicates(subset=["branch", "site"])
            subs[column] = ends["site"].value_counts().reindex(subs.index).fillna(0).astype(int)
            subs[f"max_{column}_loading"] = ends.groupby("site")["loading"].max().reindex(subs.index)

        subs["label"] = subs.index + " - " + subs["name"].fillna("")
        return subs

    def _with_bus_attributes(self, frame):
        """Joins the node name, voltage, site, zone, owner and location of each row's bus;
        rows at an unlocated bus take their substation's mean location."""
        buses, subs = self.buses, self.subs
        for column in ("name", "vn_kv", "site", "site_name", "zone", "owner", "lon", "lat"):
            target = "node_name" if column == "name" else column
            frame[target] = buses[column].reindex(frame["bus"]).values
        missing = frame["lon"].isna() & frame["site"].isin(subs.index)
        for column in ("lon", "lat"):
            frame.loc[missing, column] = subs[column].reindex(frame.loc[missing, "site"]).values
        return frame

    def _data_centre_table(self):
        loads = self.loads
        dc = loads[loads["demand_type"] == model_adapters.DATA_CENTRE_TYPE].copy()
        dc["label"] = (dc["name"].astype(str) + " - " + dc["site"].fillna("").astype(str)
                       + " (" + dc["p_mw"].map(lambda v: f"{v:,.0f}") + "MW)")
        return dc.sort_values("p_mw", ascending=False)

    def _join_branch_results(self, frames, columns):
        # columns: {branch column: result column}; frames: {model table: result frame}
        for target in columns:
            self.branches[target] = np.nan if target != "n1_outage" else None
        for table, frame in frames.items():
            rows = self.branches["table"] == table
            elements = self.branches.loc[rows, "element"]
            for target, source in columns.items():
                self.branches.loc[rows, target] = frame[source].reindex(elements).values

    def add_n1(self, n1):
        self.n1 = n1
        self._join_branch_results(n1["branches"], {
            "n1_loading": "max_loading_percent", "n1_p_mw": "p_mw", "n1_outage": "cause"})
        if n1["buses"] is not None:
            for column in ("min_vm_pu", "min_cause", "max_vm_pu", "max_cause"):
                self.buses[f"n1_{column}"] = n1["buses"][column].reindex(self.buses.index).values

    def add_screen(self, result):
        branches = self.branches
        self._join_branch_results(result["branches"], {
            "step_loading": "loading_percent", "step_p_mw": "p_mw"})
        branches["delta"] = branches["step_loading"] - branches["loading"]
        branches["delta_mw"] = branches["step_p_mw"].abs() - branches["p_mw"].abs()
        branches["delta_share"] = branches["delta_mw"] / result["step_mw"] * 100.0

    def summary_metrics(self, limits):
        injections = self.model.bus_injections()
        metrics = {"demand_mw": float(injections["demand_mw"].sum()),
                   "generation_mw": float(injections["generation_mw"].sum()),
                   "overloaded_lines": 0, "overloaded_trafos": 0,
                   "undervoltage": 0, "overvoltage": 0}
        if self.solved:
            overloaded = limits.overloaded(self.branches["loading"])
            voltage = self.buses["vm_pu"].dropna()
            metrics.update(
                overloaded_lines=int((overloaded & (self.branches["kind"] == "Line")).sum()),
                overloaded_trafos=int((overloaded & (self.branches["kind"] == "Transformer")).sum()),
                undervoltage=int((voltage < limits.low).sum()),
                overvoltage=int((voltage > limits.high).sum()),
            )
        return metrics

    def n1_metrics(self, limits):
        voltage = None
        if self.has_n1_voltages:
            voltage = int(((self.buses["n1_min_vm_pu"] < limits.low)
                           | (self.buses["n1_max_vm_pu"] > limits.high)).sum())
        return {
            "cases": self.n1["cases"],
            "unsolved": len(self.n1["unsolved"]),
            "overloaded": int(limits.overloaded(self.branches["n1_loading"]).sum()),
            "voltage": voltage,
        }

    def thermal_issues(self, limits, owners):
        branches = self.branches
        allowed = owner_mask(branches["owner"], owners)
        frames = [branches[limits.watched(branches["loading"]) & allowed]
                  .assign(condition="Intact", outage="")]
        if self.n1 is not None:
            post = branches[limits.watched(branches["n1_loading"]) & allowed]
            frames.append(post.assign(loading=post["n1_loading"], p_mw=post["n1_p_mw"],
                                      condition="N-1", outage=post["n1_outage"]))
        # An element can appear intact and post-contingency, so the index is reset for the row styling.
        table = pd.concat(frames).sort_values("loading", ascending=False).head(MAX_ISSUE_ROWS)
        return table[list(THERMAL_COLUMNS)].rename(columns=THERMAL_COLUMNS).reset_index(drop=True)

    def voltage_issues(self, limits, owners):
        located = self.buses[self.buses["vm_pu"].notna()]
        allowed = owner_mask(located["owner"], owners)
        frames = [located[limits.outside_band(located["vm_pu"]) & allowed]
                  .assign(condition="Intact", outage="")]
        if self.has_n1_voltages:
            under = located[(located["n1_min_vm_pu"] < limits.low) & allowed]
            frames.append(under.assign(vm_pu=under["n1_min_vm_pu"], condition="N-1",
                                       outage=under["n1_min_cause"]))
            over = located[(located["n1_max_vm_pu"] > limits.high) & allowed]
            frames.append(over.assign(vm_pu=over["n1_max_vm_pu"], condition="N-1",
                                      outage=over["n1_max_cause"]))
        table = pd.concat(frames)
        table = (table.assign(deviation=table["vm_pu"].sub(1.0).abs())
                 .sort_values("deviation", ascending=False).head(MAX_ISSUE_ROWS))
        return table[list(VOLTAGE_COLUMNS)].rename(columns=VOLTAGE_COLUMNS).reset_index(drop=True)

    def moved_branches(self, measure=None):
        moved = self.branches[self.branches["delta"].notna()].copy()
        moved["connects"] = moved["from_name"] + " - " + moved["to_name"]
        return moved if measure is None else moved.sort_values(measure, ascending=False)

    def screen_table(self, measure, threshold, count=MAX_ISSUE_ROWS):
        """Branches ordered by the criterion's column after the demand step; rows at or above
        the threshold are the affected set and the rest give context. Also marks the
        affected branches in ``branches`` for the charts."""
        self.branches["affected"] = (self.branches[measure] >= threshold).fillna(False)
        moved = self.moved_branches(measure)
        affected = moved[moved["affected"]]
        shown = moved.head(max(count, len(affected)))
        table = shown[list(SCREEN_COLUMNS)].rename(columns=SCREEN_COLUMNS).reset_index(drop=True)
        return affected, table

    def substation_detail(self, site):
        site_buses = self.buses[self.buses["site"] == site]
        ids = set(site_buses.index)

        node_columns = {"name": "Node", "vn_kv": "kV", "vm_pu": "V (pu)", "zone": "Zone", "owner": "TO"}
        if self.has_n1_voltages:
            node_columns.update({"n1_min_vm_pu": "N-1 V min (pu)", "n1_max_vm_pu": "N-1 V max (pu)"})
        nodes = site_buses[list(node_columns)].rename(columns=node_columns)

        branches = self.branches
        connected = branches[branches["from_bus"].isin(ids) | branches["to_bus"].isin(ids)].copy()
        connected["Connects"] = connected["from_name"] + " - " + connected["to_name"]
        branch_columns = {
            "kind": "Type", "name": "Name", "Connects": "Connects", "detail": "Detail",
            "rating_mva": "Rating (MVA)", "loading": "Loading (%)", "p_mw": "Flow (MW)"}
        if self.n1 is not None:
            branch_columns.update({"n1_loading": "N-1 loading (%)", "n1_outage": "N-1 outage"})
        if self.has_screen:
            branch_columns.update({"step_loading": "After step (%)", "delta": "Change (points)"})
        branch_columns["in_service"] = "In service"
        branch_view = connected[list(branch_columns)].rename(columns=branch_columns)

        views = []
        for frame, columns in [(self.loads, LOAD_COLUMNS), (self.generators, GENERATOR_COLUMNS),
                               (self.compensation, COMPENSATION_COLUMNS)]:
            columns = {source: heading for source, heading in columns.items()
                       if source not in SITE_COLUMNS and source in frame.columns}
            if not self.solved:
                columns = {k: v for k, v in columns.items() if k not in RESULT_COLUMNS}
            rows = frame[frame["bus"].isin(ids)]
            views.append(rows[list(columns)].rename(columns=columns))
        return nodes, branch_view, *views


class N1Job:
    """An N-1 run on a worker thread, polled by the page to draw the progress bar. A click
    on Stop interrupts the page script but not the run: the rerun finds the same job, asks
    it to stop and keeps the outages solved so far."""

    def __init__(self, key, model, mode, full_resolve, resolve_above, selection):
        self.key, self.mode, self.selection = key, mode, selection
        self.full_resolve, self.resolve_above = full_resolve, resolve_above
        self.done, self.label = 0, "starting"
        self.count = model.n1_case_count(selection.kv, selection.from_buses)
        self.result, self.error = None, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(model,), daemon=True)
        self._thread.start()

    def _run(self, model):
        try:
            self.result = model.run_n1(self.mode, self._progress, self.full_resolve,
                                       self.resolve_above, stop=self._stop.is_set,
                                       voltage_levels_kv=self.selection.kv,
                                       from_buses=self.selection.from_buses)
        except BaseException as exc:  # reported on the page thread
            self.error = exc

    def _progress(self, done, count, label):
        self.done, self.count, self.label = done, count, label

    def stop(self):
        self._stop.set()

    @property
    def running(self):
        return self._thread.is_alive()

    @property
    def stopping(self):
        return self._stop.is_set()

    @property
    def fraction(self):
        return min(self.done / max(self.count, 1), 1.0)

    @property
    def text(self):
        if self.stopping:
            return f"Stopping after the outage being solved ({self.done:,} of {self.count:,})..."
        if self.full_resolve and self.resolve_above is not None and self.label.startswith(
                ("building", "screening", "re-solving multi")):
            return f"N-1 screen ({self.mode} flows) to pick the outages to re-solve: {self.label}"
        if self.full_resolve:
            return (f"N-1 contingency {self.done:,} of {self.count:,} ({self.mode}): "
                    f"{self.label}")
        return f"N-1 screen ({self.mode} flows): {self.label}"


def n1_key(case, controls, selection):
    return ("n1", *case.key, selection.kv, selection.owners, controls.full_resolve,
            controls.resolve_above)


def n1_results(case, controls, selection):
    """The N-1 result for the selected outages, or None when the filters select none."""
    store = result_store()
    key = n1_key(case, controls, selection)
    if key in store:
        return store[key]
    if case.model.n1_case_count(selection.kv, selection.from_buses) == 0:
        return None
    job = st.session_state.get("n1_job")
    if job is not None and job.key != key:
        job.stop()  # the model, filters or options changed under a running job
        job = None
    if job is None:
        job = st.session_state["n1_job"] = N1Job(
            key, case.model, case.mode, controls.full_resolve, controls.resolve_above, selection)
    holder = st.empty()
    with holder.container():
        bar = st.progress(job.fraction, text=job.text)
        if controls.full_resolve and st.button(
                "Stop the N-1 re-solve", key="n1_stop", disabled=job.stopping,
                help="Ends the run after the outage being solved and keeps the results of "
                     "the outages solved so far."):
            job.stop()
    while job.running:
        time.sleep(0.25)
        bar.progress(job.fraction, text=job.text)
    holder.empty()
    st.session_state.pop("n1_job", None)
    if job.error is not None:
        raise job.error
    store[key] = job.result
    return job.result


def forget_n1_results(case, controls, selection):
    result_store().pop(n1_key(case, controls, selection), None)


# --------------------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------------------

def _branch_hover(row):
    loading = "no result" if pd.isna(row.loading) else f"{row.loading:.0f}%"
    flow = "" if pd.isna(row.p_mw) else f"<br>Flow: {row.p_mw:,.0f}MW"
    n1_loading = getattr(row, "n1_loading", np.nan)
    n1 = ("" if pd.isna(n1_loading)
          else f"<br>Worst N-1 loading: {n1_loading:.0f}% ({row.n1_outage} out)")
    delta = getattr(row, "delta", np.nan)
    step = ("" if pd.isna(delta)
            else f"<br>After demand step: {row.step_loading:.0f}% ({delta:+.1f} points)")
    return (f"<b>{row.name_}</b><br>{row.kind} - {row.detail}"
            f"<br>{row.from_name} to {row.to_name} ({row.kv:.0f}kV)"
            f"<br>Rating: {row.rating_mva:,.0f}MVA<br>Loading: {loading}{flow}{n1}{step}")


def _branch_hovers(branches):
    return branches.rename(columns={"name": "name_"}).apply(_branch_hover, axis=1)


def _substation_hover(row, site):
    def fmt(value, suffix="", places=2):
        return "no result" if pd.isna(value) else f"{value:,.{places}f}{suffix}"
    return (f"<b>{site} - {row['name']}</b><br>Zone: {row.zone} | TO: {row.owner}"
            f"<br>Highest voltage level: {row.max_kv:.0f}kV ({_plural(row.buses, 'node')})"
            f"<br>Voltage: {fmt(row.min_vm_pu)} to {fmt(row.max_vm_pu)}pu"
            f"<br>Demand: {row.demand_mw:,.0f}MW | Generation: {row.generation_mw:,.0f}MW"
            f"<br>{_plural(row.circuits, 'circuit')}, "
            f"{_plural(row.transformers, 'transformer')}"
            f"<br>Highest transformer loading: {fmt(row.max_transformers_loading, '%', 0)}")


def _data_centre_hover(row):
    state = "" if row.in_service else "<br><i>Out of service in this model</i>"
    return (f"<b>{row['name']}</b><br>Data centre, {row.p_mw:,.0f}MW peak demand"
            f"<br>Node {row.node_name} ({row.vn_kv:.0f}kV) at {row.site} - {row.site_name}"
            f"<br>Zone: {row.zone} | TO: {row.owner}{state}")


def _flow_bearing(branches):
    # Compass bearing of the active power flow. Positive p_mw runs from the from/HV end to
    # the to/LV end; Mercator is conformal so the bearing is also the direction drawn.
    mid_lat = np.radians((branches["from_lat"] + branches["to_lat"]) / 2)
    dx = (branches["to_lon"] - branches["from_lon"]) * np.cos(mid_lat)
    dy = branches["to_lat"] - branches["from_lat"]
    bearing = np.degrees(np.arctan2(dx, dy))
    return np.where(branches["p_mw"] < 0, bearing + 180.0, bearing) % 360.0


def _mercator_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _segments(rows):
    lon, lat = [], []
    for row in rows.itertuples():
        lon += [row.from_lon, row.to_lon, None]
        lat += [row.from_lat, row.to_lat, None]
    return lon, lat


def _midpoints(rows):
    return (rows["from_lon"] + rows["to_lon"]) / 2, (rows["from_lat"] + rows["to_lat"]) / 2


class NetworkMap:
    def __init__(self, branches, subs, selected, limits, show_arrows=True, centres=None,
                 highlight=None):
        self.drawn = branches[branches["mappable"]]
        self.subs = subs
        self.selected = selected
        self.limits = limits
        self.show_arrows = show_arrows
        self.centres = centres
        self.highlight = highlight
        self.fig = go.Figure()

    def figure(self):
        self._add_land()
        self._add_boundaries()
        self._add_loading_bands()
        if self.centres is not None:
            self._add_data_centres()
        if self.highlight is not None:
            self._add_highlight()
        self._add_branch_markers()
        self._add_substations()
        self._add_selection()
        self._layout()
        return self.fig

    def _add_land(self):
        lon, lat = coastline_rings()
        self.fig.add_trace(go.Scattergeo(
            lon=lon, lat=lat, mode="lines", fill="toself", fillcolor=LAND_COLOUR,
            line=dict(width=0.8, color=COAST_COLOUR), hoverinfo="skip", showlegend=False))

    def _add_boundaries(self):
        # Both layers start hidden (legend entry only) and are switched on from the legend.
        for path, label, colour, dash in [
            (ZONE_FILE, "DCConnect flop zones", ZONE_COLOUR, "dot"),
            (BOUNDARY_FILE, "ETYS boundary", BOUNDARY_COLOUR, "dash"),
        ]:
            lon, lat, names = geojson_lines(path)
            self.fig.add_trace(go.Scattergeo(
                lon=lon, lat=lat, mode="lines", name=label, legend="legend2",
                visible="legendonly",
                line=dict(width=1.5, color=colour, dash=dash),
                hovertext=[f"{label} {name}" if name else None for name in names],
                hoverinfo="text",
            ))

    def _add_lines(self, rows, name, width, colour, **kwargs):
        if rows.empty:
            return
        lon, lat = _segments(rows)
        self.fig.add_trace(go.Scattergeo(
            lon=lon, lat=lat, mode="lines", name=name, line=dict(width=width, color=colour),
            hoverinfo="skip", **kwargs))

    def _add_loading_bands(self):
        bands = self.drawn["loading"].map(_loading_band)
        for band, (_limit, label, colour) in enumerate(LOADING_BANDS):
            self._add_lines(self.drawn[bands == band], f"Loading {label}", 1.8, colour)
        self._add_lines(self.drawn[bands == len(LOADING_BANDS)], "No load-flow result", 1.2,
                        UNSOLVED_COLOUR)

    def _add_data_centres(self):
        located = self.centres[self.centres["lon"].notna()]
        if located.empty:
            return
        top = max(float(located["p_mw"].max()), 1.0)
        core = 12 + 26 * np.sqrt(located["p_mw"].clip(lower=0) / top)
        hover = [_data_centre_hover(row) for _, row in located.iterrows()]
        for multiple, opacity in DC_HALOS:
            is_core = multiple == 1.0
            self.fig.add_trace(go.Scattergeo(
                lon=located["lon"], lat=located["lat"], mode="markers",
                name="Data centre (size and colour: peak MW)",
                legendgroup="data centres", legend="legend3", showlegend=False,
                marker=dict(
                    size=core * multiple, opacity=opacity, color=located["p_mw"],
                    colorscale=DC_HEAT_SCALE, cmin=0, cmax=top,
                    line=dict(width=1 if is_core else 0, color="white"),
                    showscale=is_core,
                    colorbar=dict(title=dict(text="Data centre<br>peak MW", font=dict(size=11)),
                                  thickness=10, len=0.28, x=0.99, xanchor="right",
                                  y=0.99, yanchor="top", tickfont=dict(size=10),
                                  outlinewidth=0, bgcolor="rgba(255,255,255,0.7)"),
                ),
                hovertext=hover if is_core else None,
                hoverinfo="text" if is_core else "skip",
            ))
        # Plotly sizes every legend row by the largest marker of any trace in that legend, so
        # the MW-scaled markers sit in a hidden legend and this fixed-size trace stands in.
        self.fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None], mode="markers", name="Data centre (size and colour: peak MW)",
            legendgroup="data centres", legend="legend2", hoverinfo="skip",
            marker=dict(size=10, color=DC_HEAT_SCALE[1][1]),
        ))

    def _add_highlight(self):
        highlight = self.highlight
        affected = highlight.affected[highlight.affected["mappable"]]
        centre = highlight.centre
        if not affected.empty:
            self._add_lines(
                affected, f"{highlight.criterion_text} with +{highlight.step_mw:,.0f}MW at {centre['site']}",
                4.5, AFFECTED_COLOUR, legendgroup="affected", opacity=0.85)
            lon, lat = _midpoints(affected)
            self.fig.add_trace(go.Scattergeo(
                lon=lon, lat=lat, mode="markers", legendgroup="affected", showlegend=False,
                marker=dict(size=14, color=AFFECTED_COLOUR, opacity=0.001),
                hovertext=_branch_hovers(affected), hoverinfo="text",
            ))
        if pd.notna(centre["lon"]):
            self.fig.add_trace(go.Scattergeo(
                lon=[centre["lon"]], lat=[centre["lat"]], mode="markers",
                name=f"Demand step at {centre['name']}", legend="legend2",
                marker=dict(symbol="star", size=18, color=AFFECTED_COLOUR,
                            line=dict(width=1.5, color="white")),
                hoverinfo="skip",
            ))

    def _add_branch_markers(self):
        # Plotly only fires hover on vertices, and the ends sit under the substation
        # markers, so midpoint markers carry the branch hover text. Where a flow result
        # exists the marker is an arrow head in the direction of flow; otherwise invisible.
        drawn = self.drawn
        if drawn.empty:
            return
        arrow = (drawn["p_mw"].notna() & self.show_arrows).to_numpy()
        lon, lat = _midpoints(drawn)
        self.fig.add_trace(go.Scattergeo(
            lon=lon, lat=lat, mode="markers", name="Active power direction",
            marker=dict(
                symbol=np.where(arrow, "triangle-up", "circle"),
                angle=np.where(arrow, _flow_bearing(drawn), 0.0), angleref="up",
                size=np.where(arrow, 8, 12), opacity=np.where(arrow, 1.0, 0.001),
                color=ARROW_COLOUR, line=dict(width=1, color="white"),
            ),
            hovertext=_branch_hovers(drawn), hoverinfo="text", showlegend=False,
        ))

    def _add_substations(self):
        subs, limits = self.subs, self.limits
        has_problem = (
            limits.overloaded(subs["max_circuits_loading"])
            | limits.overloaded(subs["max_transformers_loading"])
            | (subs["min_vm_pu"] < limits.low)
            | (subs["max_vm_pu"] > limits.high)
        ).fillna(False)
        for problem, colour, label in [(False, SUBSTATION_COLOUR, "Substation"),
                                       (True, PROBLEM_COLOUR, "Substation with an issue")]:
            subset = subs[has_problem == problem]
            if subset.empty:
                continue
            self.fig.add_trace(go.Scattergeo(
                lon=subset["lon"], lat=subset["lat"], mode="markers", name=label,
                legend="legend2",
                marker=dict(size=(subset["max_kv"] / 100 + 4).clip(upper=9), color=colour),
                customdata=subset.index,
                hovertext=[_substation_hover(row, site) for site, row in subset.iterrows()],
                hoverinfo="text",
            ))

    def _add_selection(self):
        if self.selected not in self.subs.index:
            return
        chosen = self.subs.loc[self.selected]
        self.fig.add_trace(go.Scattergeo(
            lon=[chosen["lon"]], lat=[chosen["lat"]], mode="markers", name="Selected",
            legend="legend2",
            marker=dict(size=20, color=SELECTED_COLOUR, opacity=0.55), hoverinfo="skip",
        ))

    def _view_ranges(self):
        # Plotly letterboxes the lon/lat box in the plot area; a square box keeps the strips
        # small whatever the column's aspect, and the legend close under the map.
        if self.selected in self.subs.index:
            centre = self.subs.loc[self.selected]
            lon_centre = centre["lon"]
            lat_range = [centre["lat"] - SITE_HALF_SPAN_DEG, centre["lat"] + SITE_HALF_SPAN_DEG]
        else:
            lon_centre, lat_range = sum(GB_LON_RANGE) / 2, GB_LAT_RANGE
        lon_span = math.degrees((_mercator_y(lat_range[1]) - _mercator_y(lat_range[0])) * MAP_ASPECT)
        return [lon_centre - lon_span / 2, lon_centre + lon_span / 2], lat_range

    def _layout(self):
        # Plain geo axes rather than a tiled map: renders without GPU support or tile servers.
        lon_range, lat_range = self._view_ranges()
        self.fig.update_layout(
            geo=dict(
                projection=dict(type="mercator"),
                lonaxis=dict(range=lon_range),
                lataxis=dict(range=lat_range),
                # The land is drawn from coastline.geojson; the sea is the background colour.
                showland=False, showcoastlines=False, showcountries=False,
                showocean=False, showlakes=False, showframe=False,
                bgcolor=SEA_COLOUR,
            ),
            margin=dict(l=0, r=0, t=0, b=MAP_LEGEND_MARGIN),
            height=MAP_HEIGHT,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=charts.FONT, size=12, color=charts.INK),
            # Two legend columns under the map; each trace names its column. The group gap
            # is zero so a column with legendgroup traces keeps the same row height.
            **{key: dict(orientation="v", yanchor="top", y=-0.01, xanchor="left",
                         x=column / len(MAP_LEGEND_COLUMNS), bgcolor="rgba(0,0,0,0)",
                         font=dict(size=11), tracegroupgap=0,
                         title=dict(text=f"<b>{title}</b>", font=dict(size=11, color=charts.INK)),
                         itemsizing="constant")
               for column, (key, title) in enumerate(MAP_LEGEND_COLUMNS.items())},
            legend3=dict(visible=False),  # holds the data centre markers, see _add_data_centres
            uirevision="network-map",
        )
        # An info marker in the corner: the map's how-to appears on hover.
        self.fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text="<b>i</b>", showarrow=False, hovertext=MAP_HELP, align="left",
            font=dict(size=13, color="white"), bgcolor=BRAND["mid_blue"],
            bordercolor=BRAND["mid_blue"], borderwidth=1, borderpad=4, width=14, height=14,
            hoverlabel=dict(bgcolor="white", bordercolor=BRAND["mid_blue"],
                            font=dict(size=12, color=charts.INK)),
        )


# --------------------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------------------

def condition_styled(table):
    def shade(row):
        return [N1_ROW_STYLE if row["Condition"] == "N-1" else ""] * len(row)
    return table.style.apply(shade, axis=1)


def affected_styled(table, column, threshold):
    def shade(row):
        return [AFFECTED_ROW_STYLE if row[column] >= threshold else ""] * len(row)
    return table.style.apply(shade, axis=1)


def _load_flow_status(mode, summary):
    """'AC load flow: Successful at 100%', or Unsuccessful where the AC ramp stopped short."""
    share = summary.get("highest converged share of peak demand", 1.0)
    if share >= 1.0:
        return f"{mode} load flow: Successful at 100%"
    return f"{mode} load flow: Unsuccessful at 100% (converged at {share:.0%})"


def render_header():
    title_column, *logo_columns = st.columns([5.4, 1.8, 1.0], vertical_alignment="center")
    title_column.markdown(
        f'<div class="app-header"><p class="app-title">{APP_TITLE}</p>'
        f'<p class="app-subtitle">{APP_SUBTITLE}</p></div>', unsafe_allow_html=True)
    for column, file_name in zip(logo_columns, LOGOS):
        logo = trimmed_logo(os.path.join(LOGO_DIR, file_name))
        column.image(logo, width=round(LOGO_HEIGHT * logo.width / logo.height))
    st.markdown('<hr class="app-rule">', unsafe_allow_html=True)


def sidebar_group(title):
    group = st.container(border=True)
    group.header(title)
    return group


def render_sidebar_controls(models):
    with st.sidebar:
        with st.container(border=True):
            page = st.radio("Page", [NETWORK_PAGE, ABOUT_PAGE], horizontal=True, key="page",
                            label_visibility="collapsed")
        with sidebar_group("Model"):
            path = st.selectbox("Network", models, format_func=os.path.basename,
                                help="The network file to show.")
            mode = st.radio("Load flow", ["AC", "DC", "None"], horizontal=True,
                            help="AC solves voltages and flows, DC solves active power flows "
                                 "only, None shows the network as built.")

        with sidebar_group("N-1 contingencies"):
            run_n1 = st.checkbox(
                "Run N-1 for the filtered outages", disabled=mode == "None",
                help="Outages in turn each in-service circuit and transformer at the voltage "
                     "levels and transmission owners chosen under Filters, and reports the "
                     "worst loading of every element.")
            can_resolve = mode == "AC" and run_n1
            resolve = st.radio(
                "Re-solve with the AC load flow", [RESOLVE_NONE, RESOLVE_ABOVE, RESOLVE_ALL],
                disabled=not can_resolve,
                help="The LODF screen gives worst loadings only. Re-solving outages with "
                     "the AC load flow also gives post-outage voltages: every outage, or "
                     "only the outages the screen finds raising some element's loading "
                     "above the threshold (by at least one point over its intact loading). "
                     "A Stop button beside the progress bar ends a re-solve early and keeps "
                     "the outages solved so far.")
            threshold = st.number_input(
                "Re-solve threshold (%)", 0, 200, 90, step=5, disabled=not can_resolve,
                help=f"Used by the '{RESOLVE_ABOVE}' option.")
            full_resolve = can_resolve and resolve != RESOLVE_NONE
            resolve_above = float(threshold) if full_resolve and resolve == RESOLVE_ABOVE else None
    return Controls(page, path, mode, LIMITS, run_n1, full_resolve, resolve_above)


def render_sidebar_footer():
    with st.sidebar:
        if st.button("Refresh network", type="primary", width="stretch",
                     help="Reloads the network file and re-runs the load flow."):
            load_model.clear()
            result_store.clear()
            st.session_state.pop("screen_key", None)
            st.rerun()


def render_sidebar_filters(case):
    with st.sidebar, sidebar_group("Filters"):
        voltages = sorted(case.buses["vn_kv"].unique(), reverse=True)
        chosen_kv = st.multiselect(
            "Voltage levels", voltages, default=[v for v in voltages if v >= 132],
            format_func=lambda kv: f"{kv:g}kV",
            help="Limits the map, and the N-1 outages, to circuits and transformers at "
                 "these voltage levels.")
        chosen_tos = st.multiselect(
            "Transmission owner", owner_names(case.buses),
            help="Limits the charts, the elements to look at and the N-1 outages to these "
                 "owners.")
        show_arrows = st.checkbox(
            "Show power-flow arrows", value=True, disabled=case.mode == "None",
            help="Shows the direction of active power flow on each circuit.")
    return Filters(chosen_kv, chosen_tos, show_arrows)


def render_summary(case, limits):
    metrics = case.summary_metrics(limits)
    columns = st.columns(4)
    columns[0].metric("Demand", f"{metrics['demand_mw']:,.0f}MW", border=True,
                      help="Total in-service demand.")
    # Dashes mark results the analysis has not produced; they are unavailable, not zero.
    solved = case.solved
    needs_flow = "" if solved else " Needs a load flow."
    columns[1].metric("Generation",
                      f"{metrics['generation_mw']:,.0f}MW" if solved else UNAVAILABLE, border=True,
                      help="Total generation dispatched, including losses." + needs_flow)
    columns[2].metric("Data centres", f"{len(case.centres):,}", border=True,
                      help=f"Data centre loads, {case.centres['p_mw'].sum():,.0f}MW peak in total.")
    violations = (metrics["overloaded_lines"] + metrics["overloaded_trafos"]
                  + metrics["undervoltage"] + metrics["overvoltage"])
    needs_ac = "" if case.mode == "AC" else " Voltage violations need the AC load flow."
    columns[3].metric("Violations", f"{violations:,}" if solved else UNAVAILABLE,
                      border=True,
                      help=f"Circuits and transformers above {limits.loading}% and nodes "
                           f"outside {limits.low:.2f} - {limits.high:.2f}pu: "
                           f"{metrics['overloaded_lines']:,} circuits, "
                           f"{metrics['overloaded_trafos']:,} transformers, "
                           f"{metrics['undervoltage']:,} nodes low, "
                           f"{metrics['overvoltage']:,} nodes high." + needs_flow + needs_ac)

    summary = case.summary
    if summary is None:
        st.warning("No load flow selected: generation and violations are unavailable, not zero.")
    elif not case.solved:
        st.error(f"{case.mode} load flow: Unsuccessful "
                 f"({summary.get('error', 'unknown reason')})")
    else:
        st.caption(_load_flow_status(case.mode, summary))
        share = summary.get("highest converged share of peak demand")
        if share is not None and share < 1.0:
            st.warning(f"The AC load flow converged at {share:.0%} of peak demand; results "
                       "are for that operating point.")


def render_n1_summary(case, controls, selection):
    n1, mode, limits = case.n1, case.mode, controls.limits
    selective = n1["method"] == "LODF + load flow"
    post = case.n1_metrics(limits)
    columns = st.columns(4)
    columns[0].metric("N-1 outages run", f"{post['cases']:,}", border=True,
                      help=f"{selection.description} outaged one at a time ({mode} flows), "
                           "as chosen under Filters.")
    columns[1].metric("N-1 outages unsolved", f"{post['unsolved']:,}", border=True,
                      help="Outages that could not be assessed.")
    columns[2].metric("N-1 overloads", f"{post['overloaded']:,}", border=True,
                      help=f"Elements above {limits.loading}% after at least one outage.")
    columns[3].metric("N-1 voltage violations",
                      "n/a" if post["voltage"] is None else f"{post['voltage']:,}",
                      border=True,
                      help="Nodes outside the band after an outage; needs the AC re-solve.")
    if selective:
        st.caption(f"{post['cases']:,} outages screened with LODFs; {n1['re-solved outages']:,} "
                   f"of the {n1['re-solve candidates']:,} that raise an element's loading "
                   f"above {n1['re-solve threshold (%)']:.0f}% were re-solved with the {mode} "
                   "load flow. Loadings are the higher of the two results; voltages come "
                   "from the re-solved outages only.")
    if n1.get("stopped"):
        requested = n1["re-solve candidates"] if selective else n1["requested"]
        solved = n1["re-solved outages"] if selective else n1["cases"]
        cells = st.columns([5, 1])
        cells[0].warning(f"The re-solve was stopped after {solved:,} of {requested:,} "
                         "outages: the N-1 loadings and voltages cover those outages only.")
        if cells[1].button("Run again", key="n1_run_again"):
            forget_n1_results(case, controls, selection)
            st.rerun()
    if n1["unsolved"]:
        with st.expander(f"{post['unsolved']:,} N-1 outages did not solve"):
            st.dataframe(pd.DataFrame(n1["unsolved"]).rename(
                columns={"outage": "Outage", "reason": "Reason"}), hide_index=True)


class DemandStepScreen:
    """The DCConnect study panel: pick a data centre, add demand, re-solve and list the
    branches whose loading rises most."""

    def __init__(self, case, limits):
        self.case = case
        self.limits = limits
        self.store = result_store()

    def render(self):
        centres = self.case.centres
        if centres.empty:
            st.info("This model has no data centre loads.")
            return None
        load_id = self._pick_centre()
        step_mw, criterion, threshold = self._controls()
        self._run_if_asked(load_id, step_mw)
        result = self._active_result()
        if result is None:
            return None
        if not result["converged"]:
            st.error(f"The load flow with the demand step did not solve: {result['error']}")
            return None
        return self._results(result, criterion, threshold)

    def _pick_centre(self):
        centres = self.case.centres
        chosen = st.selectbox("Data centre", centres["label"].tolist(), key="dc_choice",
                              help="Data centres, largest first.")
        load_id = centres.index[centres["label"] == chosen][0]
        centre = centres.loc[load_id]
        st.markdown(
            f'<p class="panel-note">Connected at node <b>{centre.node_name}</b> '
            f'({centre.vn_kv:.0f}kV), {centre.site} - {centre.site_name}, {centre.zone}, '
            f'{centre.owner}.</p>', unsafe_allow_html=True)
        return load_id

    @staticmethod
    def _controls():
        step_column, criterion_column = st.columns(2)
        step_mw = step_column.number_input(
            "Additional demand (MW)", min_value=10, max_value=5000, value=DEFAULT_STEP_MW, step=10,
            help="Extra demand added at the data centre and met by scaling up generation.")
        criterion_name = criterion_column.radio(
            "Affected by", list(CRITERIA), help="How a circuit counts as affected.")
        criterion = CRITERIA[criterion_name]
        threshold = st.slider(criterion["label"], *criterion["limits"], criterion["default"],
                              key=f"threshold_{criterion['column']}", help=criterion["help"])
        return float(step_mw), criterion, threshold

    def _run_if_asked(self, load_id, step_mw):
        case = self.case
        solved = case.mode != "None" and case.solved
        run = st.button("Screen affected circuits", type="primary", width="stretch",
                        disabled=not solved,
                        help=None if solved else "Select the AC or DC load flow first.")
        if not (run and solved):
            return
        key = ("screen", *case.key, str(load_id), step_mw)
        if key not in self.store:
            centre = case.centres.loc[load_id]
            with st.spinner(f"Re-solving the {case.mode} load flow with +{step_mw:,.0f}MW at "
                            f"{centre['name']}..."):
                self.store[key] = case.model.run_demand_step(load_id, step_mw, case.mode)
        st.session_state["screen_key"] = key

    def _active_result(self):
        # The screen stays on screen across reruns until it is cleared or the case changes.
        key = st.session_state.get("screen_key")
        if key is None or key[1:4] != self.case.key or key not in self.store:
            return None
        return self.store[key]

    def _active_centre(self):
        wanted = st.session_state["screen_key"][4]
        centres = self.case.centres
        return centres.loc[next(i for i in centres.index if str(i) == wanted)]

    def _results(self, result, criterion, threshold):
        case, limits, mode = self.case, self.limits, self.case.mode
        centre = self._active_centre()
        measure = criterion["column"]
        criterion_text = (f"Loading up {threshold:g}+ points" if measure == "delta"
                          else f"Flow up {threshold:g}%+ of the added MW")
        qualifies = criterion_text[0].lower() + criterion_text[1:]
        measure_label = SCREEN_COLUMNS[measure]

        case.add_screen(result)
        affected, table = case.screen_table(measure, threshold)
        new_overloads = affected[limits.overloaded(affected["step_loading"])
                                 & ~limits.overloaded(affected["loading"])]

        st.markdown(f"**+{result['step_mw']:,.0f}MW at {centre['name']}** ({mode} load flow)")
        columns = st.columns(3)
        columns[0].metric("Circuits", f"{int((affected['kind'] == 'Line').sum()):,}",
                          border=True, help=f"Lines with {qualifies}.")
        columns[1].metric("Transformers",
                          f"{int((affected['kind'] == 'Transformer').sum()):,}", border=True,
                          help=f"Transformers with {qualifies}.")
        largest = case.branches[measure].max()
        if pd.isna(largest):
            largest_text = "n/a"
        elif measure == "delta":
            largest_text = f"{largest:+.1f} pts"
        else:
            largest_text = f"{largest:+.0f}%"
        columns[2].metric("Largest rise", largest_text, border=True,
                          help=f"Largest {measure_label.lower()} on any branch; "
                               f"{len(new_overloads)} newly above {limits.loading}%.")
        st.dataframe(affected_styled(table, measure_label, threshold), hide_index=True,
                     column_config=ISSUE_COLUMN_CONFIG, height=300)
        st.caption("Shaded rows are affected and drawn in teal on the map.")

        export = case.moved_branches(measure)[list(SCREEN_COLUMNS)].rename(columns=SCREEN_COLUMNS)
        action_left, action_right = st.columns(2)
        action_left.download_button(
            "Download all branches (CSV)", export.to_csv(index=False).encode("utf-8"),
            file_name=f"dcconnect_screen_{centre['site']}_{result['step_mw']:.0f}MW_{mode}.csv",
            mime="text/csv", width="stretch")
        if action_right.button("Clear screen", width="stretch"):
            st.session_state.pop("screen_key", None)
            st.rerun()
        return ScreenHighlight(affected, centre, result["step_mw"], threshold, criterion_text)


def render_scenario_cards(scenario):
    cards = [
        ("Scenario - year", str(scenario.year), BRAND["dark_blue"]),
        ("Scenario - FES", scenario.fes, BRAND["teal"]),
        ("Scenario - DC Connect", scenario.dc, charts.ALERT_COLOUR),
    ]
    for column, (label, value, colour) in zip(st.columns(3), cards):
        if value is None:
            value, colour = NOT_SET, BRAND["grey"]
        with column.container(border=True):
            st.markdown(f'<p class="scenario-label">{label}</p>'
                        f'<p class="scenario-value" style="color: {colour};">{value}</p>',
                        unsafe_allow_html=True)


def render_list(frame, spec, key):
    """Filter, sort and search controls above a table of ``frame``. Returns the rows shown."""
    total = len(frame)
    with st.container(border=True):
        st.markdown(f'<p class="list-title">{spec.title}</p>', unsafe_allow_html=True)
        filter_column, value_column, sort_column = st.columns(3)
        field = filter_column.selectbox("Filter by", ["All"] + list(spec.filters),
                                        key=f"{key}_filter")
        if field != "All":
            column = spec.filters[field]
            if column == "owner":
                options = owner_names(frame)
            else:
                options = sorted(frame[column].dropna().unique(), key=_option_label)
            chosen = value_column.selectbox(field, options, key=f"{key}_{column}",
                                            format_func=_option_label)
            if column == "owner":
                frame = frame[owner_mask(frame["owner"], [chosen])]
            else:
                frame = frame[frame[column] == chosen]
        sort = sort_column.selectbox("Sort", list(spec.sorts), key=f"{key}_sort")
        column, ascending = spec.sorts[sort]
        frame = frame.sort_values(column, ascending=ascending, kind="stable")
        query = st.text_input("Search", key=f"{key}_search", label_visibility="collapsed",
                              placeholder="Search")
        terms = query.lower().split()
        if terms:
            text = frame[list(spec.search)].astype(str).agg(" ".join, axis=1).str.lower()
            keep = np.logical_and.reduce([text.str.contains(t, regex=False) for t in terms])
            frame = frame[keep]
    table = frame[list(spec.columns)].rename(columns=spec.columns)
    height = min(LIST_HEIGHT, 35 * (len(table) + 1) + 3)  # no blank rows under a short list
    st.dataframe(table, hide_index=True, column_config=ISSUE_COLUMN_CONFIG, height=height)
    footer = f"{len(frame):,} of {total:,}"
    if spec.total:
        footer += f", {frame[spec.total].sum():,.0f}MW"
    st.caption(footer)
    return frame


def render_demand_tab(case):
    spec = ListSpec(
        title="Demand", columns=DEMAND_LIST_COLUMNS,
        filters=LIST_FILTERS,
        sorts={"Demand, largest first": ("p_mw", False), "Name A-Z": ("name", True),
               "Site A-Z": ("site", True)},
        search=("name", "node_name", "site", "site_name"), total="p_mw")
    render_list(case.loads, spec, "demand")


def render_generation_tab(case):
    columns = dict(GENERATION_LIST_COLUMNS)
    sorts = {"Capacity, largest first": ("capacity_mw", False),
             "Dispatched, largest first": ("dispatch_mw", False),
             "Name A-Z": ("name", True), "Site A-Z": ("site", True)}
    if not case.solved:
        columns.pop("dispatch_mw")
        sorts.pop("Dispatched, largest first")
    spec = ListSpec(
        title="Generation", columns=columns,
        filters=LIST_FILTERS,
        sorts=sorts, search=("name", "node_name", "site", "site_name", "plant_type"),
        total="capacity_mw")
    render_list(case.generators, spec, "generation")


def render_data_centres_tab(case, limits):
    """The data centre list and the connection screen; returns the screen's map highlight."""
    spec = ListSpec(
        title="Data centres", columns=DATA_CENTRE_COLUMNS,
        filters=LIST_FILTERS,
        sorts={"Peak demand, largest first": ("p_mw", False), "Name A-Z": ("name", True),
               "Site A-Z": ("site", True)},
        search=("name", "node_name", "site", "site_name"), total="p_mw")
    render_list(case.centres, spec, "centres")
    st.markdown('<p class="list-title">Screen a connection</p>', unsafe_allow_html=True)
    return DemandStepScreen(case, limits).render()


NETWORK_ELEMENTS = {
    "Nodes": ListSpec(
        title="Nodes", columns=NODE_COLUMNS,
        filters=LIST_FILTERS,
        sorts={"Name A-Z": ("name", True), "Voltage, highest first": ("vn_kv", False),
               "V (pu), lowest first": ("vm_pu", True), "V (pu), highest first": ("vm_pu", False)},
        search=("name", "site", "site_name")),
    "Circuits": ListSpec(
        title="Circuits", columns=CIRCUIT_COLUMNS,
        filters=LIST_FILTERS,
        sorts={"Loading, highest first": ("loading", False),
               "Rating, largest first": ("rating_mva", False),
               "Length, longest first": ("length_km", False), "Name A-Z": ("name", True)},
        search=("name", "from_name", "to_name")),
    "Transformers": ListSpec(
        title="Transformers", columns=TRANSFORMER_COLUMNS,
        filters=LIST_FILTERS,
        sorts={"Loading, highest first": ("loading", False),
               "Rating, largest first": ("rating_mva", False), "Name A-Z": ("name", True)},
        search=("name", "from_name", "to_name")),
    "Reactive compensation": ListSpec(
        title="Reactive compensation", columns=COMPENSATION_COLUMNS,
        filters=LIST_FILTERS,
        sorts={"Absorb, largest first": ("absorb_mvar", False),
               "Generate, largest first": ("generate_mvar", False), "Name A-Z": ("name", True)},
        search=("name", "node_name", "site_name")),
}


def render_network_tab(case):
    """One list per element kind; result columns are left out until a load flow has run."""
    element = st.radio("Element", list(NETWORK_ELEMENTS), horizontal=True,
                       label_visibility="collapsed", key="network_element")
    spec = NETWORK_ELEMENTS[element]
    frames = {"Nodes": case.buses, "Circuits": case.branches[case.branches["kind"] == "Line"],
              "Transformers": case.branches[case.branches["kind"] == "Transformer"],
              "Reactive compensation": case.compensation}
    frame = frames[element]
    hidden = [c for c in RESULT_COLUMNS if c in spec.columns
              and (not case.solved or frame[c].isna().all())]
    if hidden:
        spec = ListSpec(
            spec.title, {k: v for k, v in spec.columns.items() if k not in hidden}, spec.filters,
            {label: pair for label, pair in spec.sorts.items() if pair[0] not in hidden},
            spec.search)
    render_list(frame, spec, f"network_{element.split()[0].lower()}")


def render_config_tab(path):
    """Year and scenario labels for the view; they do not recalculate anything."""
    with st.container(border=True):
        st.markdown('<p class="list-title">Configuration</p>', unsafe_allow_html=True)
    year = st.slider("Year", *YEAR_RANGE, _model_year(path), key="config_year")
    dc = st.selectbox("Data centre scenario", DC_SCENARIOS, index=None, key="config_dc",
                      placeholder="Choose scenario")
    fes = st.selectbox("FES scenario", FES_SCENARIOS, index=None, key="config_fes",
                       placeholder="Choose scenario")
    return Scenario(year, fes, dc)


def render_map(case, controls, filters, highlight):
    """The substation search box and the map. A map click stashes the site's label in a
    pending key and reruns; the next run copies it into the search box before the selectbox
    is instantiated (Streamlit forbids writing a widget's key afterwards). Returns the
    selected site code, or None."""
    subs = case.subs
    branches = case.branches[case.branches["kv"].isin(filters.kv)] if filters.kv else case.branches

    labels = subs["label"].sort_values()
    options = [NO_SELECTION] + labels.tolist()
    pending = st.session_state.pop("pending_substation_choice", None)
    if pending in options:
        st.session_state["substation_choice"] = pending
    if st.session_state.get("substation_choice") not in options:
        st.session_state["substation_choice"] = NO_SELECTION
    chosen_label = st.selectbox("Find a substation", options, key="substation_choice",
                                help="Search by site code or name.")
    selected = None if chosen_label == NO_SELECTION else labels.index[labels == chosen_label][0]

    figure = NetworkMap(branches, subs, selected, controls.limits, filters.show_arrows,
                        case.centres, highlight).figure()
    event = st.plotly_chart(figure, key="map", on_select="rerun", selection_mode="points")
    points = (event or {}).get("selection", {}).get("points", [])
    clicked = [p.get("customdata") for p in points]
    clicked = [c[0] if isinstance(c, list) else c for c in clicked if c]
    # The chart's selection persists across reruns, so only act on a *new* click;
    # otherwise a stale click would keep overriding later choices made in the search box.
    last_click = st.session_state.get("last_map_click")
    if clicked and clicked[0] != last_click and clicked[0] in subs.index:
        st.session_state["last_map_click"] = clicked[0]
        if clicked[0] != selected:
            st.session_state["pending_substation_choice"] = subs.at[clicked[0], "label"]
            st.rerun()
    return selected


def render_substation_detail(case, selected):
    if selected is None:
        st.info("No substation selected. Click a marker on the map or use the search box.")
        return
    row = case.subs.loc[selected]
    st.markdown(f"**{selected} - {row['name']}** &nbsp; | &nbsp; {row.zone} "
                f"&nbsp; | &nbsp; {row.owner} &nbsp; | &nbsp; up to {row.max_kv:.0f}kV")
    frames = case.substation_detail(selected)
    titles = ["Nodes", "Branches", "Demand", "Generation", "Compensation"]
    tabs = st.tabs([f"{title} ({len(frame)})" for title, frame in zip(titles, frames)])
    for tab, frame in zip(tabs, frames):
        with tab:
            st.dataframe(frame, hide_index=True)


def render_charts(case, limits, owners, highlight):
    branches, buses, subs = case.branches, case.buses, case.subs
    if owners:
        branches = branches[owner_mask(branches["owner"], owners)]
        buses = buses[owner_mask(buses["owner"], owners)]
        subs = subs[owner_mask(subs["owner"], owners)]
        st.caption("Charts show " + ", ".join(owners) + " elements only.")
    # After an N-1 run both cases can be ticked: the cumulative curve then draws the worst
    # N-1 loading over the intact curve and the most-loaded chart ranks by N-1 loading.
    base, overlay = "loading", None
    if case.n1 is not None:
        cells = st.columns([1, 1, 4])
        show_intact = cells[0].checkbox("Intact network", value=True, key="charts_intact")
        show_n1 = cells[1].checkbox("Worst N-1 outage", value=True, key="charts_n1")
        if not show_intact and not show_n1:
            st.info("Tick Intact network or Worst N-1 outage to see the loading charts.")
            return
        if show_n1:
            overlay = "n1_loading" if show_intact else None
            base = "loading" if show_intact else "n1_loading"
            if case.mode == "AC" and not case.has_n1_voltages:
                cells[2].caption("N-1 voltages need the AC re-solve option.")
    circuits = charts.transmission_circuits(branches, base)
    if circuits.empty:
        st.plotly_chart(charts.empty_chart("No transmission circuit loading results to chart."),
                        width="stretch", theme=None)
        return

    stats = charts.describe_loading(circuits, base)
    # With both cases ticked the delta is the change under the worst N-1 outage.
    change = charts.describe_loading(circuits, overlay) - stats if overlay else None
    columns = st.columns(3)
    columns[0].metric("Transmission circuits", f"{int(stats['circuits']):,}", border=True,
                      help="In-service circuits at 132kV and above with a result.")
    for cell, label, key in zip(columns[1:], ["95th percentile loading", "Highest loading"],
                                ["p95", "max"]):
        delta = None if change is None else f"{change[key]:+.0f} points N-1"
        cell.metric(label, f"{stats[key]:.0f}%", delta=delta, delta_color="inverse",
                    border=True)

    if case.mode == "AC" and buses["vm_pu"].notna().any():
        # The same cases as the loading charts; post-outage voltages need the AC re-solve.
        series = ["intact"] if base == "loading" else []
        if (overlay or base == "n1_loading") and case.has_n1_voltages:
            series += ["n1_low", "n1_high"]
        voltage = charts.voltage_cumulative(buses, limits.vm, series)
    else:
        voltage = charts.empty_chart("Voltages need the AC load flow.")
    ranked = overlay or base
    figures = [charts.loading_cumulative(circuits, limits.loading, base, overlay), voltage,
               charts.most_loaded(circuits, LOADING_BANDS, limits.loading, ranked,
                                  intact_column=base if overlay else None)]
    if highlight is not None:
        figures.append(charts.screen_waterfall(branches[branches["delta"].notna()],
                                               highlight.step_mw))
    figures.append(charts.zone_balance(subs))
    for first in range(0, len(figures), 2):
        for cell, figure in zip(st.columns(2, gap="medium"), figures[first:first + 2]):
            cell.plotly_chart(figure, width="stretch", theme=None)


def render_issues(case, limits, owners):
    notes = []
    if owners:
        notes.append("Showing " + ", ".join(owners) + " only.")
    if case.n1 is not None:
        notes.append("Shaded N-1 rows are the worst post-contingency result, with the "
                     "outage that caused it.")
    if notes:
        st.caption(" ".join(notes))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"**Branches loaded above {limits.watch:.0f}%**")
        st.dataframe(condition_styled(case.thermal_issues(limits, owners)), hide_index=True,
                     column_config=ISSUE_COLUMN_CONFIG)
    with right:
        st.markdown(f"**Nodes outside {limits.low:.2f} - {limits.high:.2f}pu**")
        st.dataframe(condition_styled(case.voltage_issues(limits, owners)), hide_index=True,
                     column_config=ISSUE_COLUMN_CONFIG)
        if case.n1 is not None and not case.has_n1_voltages:
            st.caption("N-1 voltages need the AC re-solve option.")


ABOUT_INTRO = ("A load flow model of the GB transmission network, built from published NESO data, "
               "for assessing where new data centre demand can connect.")

ABOUT_SECTIONS = {
    "Data sources": """
* **Network**: NESO Electricity Ten Year Statement (ETYS) Appendix B, covering NGET,
  SPT and SHET.
* **Demand**: NESO Future Energy Scenarios (FES) by Grid Supply Point, for the chosen
  scenario and year.
* **Generation**: NESO TEC and Interconnector Registers, with the Clean Power 2030
  connections queue deciding which projects are in service.
* **Data centres**: added by hand as extra demand at the connecting substation.
""",
    "Model capabilities": """
* **AC and DC load flow** of the intact network, in pandapower.
* **N-1 contingency analysis**: each circuit and transformer outaged in turn, with an
  optional AC re-solve for post-outage voltages.
* **Data centre connections**: add demand at a data centre and see which circuits
  are affected.
* **Filters** by voltage level and transmission owner, applied to the map, tables
  and N-1 run alike.
""",
    "Outputs and visualisation": """
* **Network map** coloured by loading and voltage, with data centre markers and
  substation detail.
* **Loading and voltage results**: summary metrics, loading charts and a list of
  elements to look at.
* **Contingency impacts**: the worst N-1 loading of each element and the outage
  causing it.
* **Data tables** of circuits, transformers, demand, generation and data centres,
  with search, sort and download.
""",
}

ABOUT_LIMITS = """
| Limit | Value |
|---|---|
| Thermal | {loading:.0f}% of the continuous winter rating, intact and after any single outage. Short-term ratings are not considered. |
| Voltage | {low:.2f}pu to {high:.2f}pu at 400kV, 275kV and 132kV, intact and after any single outage ¹ |
| Contingencies | Single outages of circuits and transformers only, with no post-fault switching, tap or compensation actions. |
| Demand and dispatch | Winter evening peak at unity power factor, met by scaling generation in proportion to rated capacity. |

¹ {source}.
"""


def render_about_page(limits):
    st.markdown(ABOUT_INTRO)
    for column, (title, body) in zip(st.columns(3, gap="large"), ABOUT_SECTIONS.items()):
        with column:
            st.markdown(f"### {title}")
            st.markdown(body)
    with st.expander("Network limits and assumptions", expanded=False):
        st.markdown(ABOUT_LIMITS.format(loading=limits.loading, low=limits.low,
                                        high=limits.high, source=VOLTAGE_LIMIT_SOURCE))


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":zap:", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    render_header()

    models = available_models()
    if not models:
        st.error(f"No network files found in {MODEL_DIR}. Put a pandapower network "
                 "JSON there first.")
        return

    controls = render_sidebar_controls(models)
    limits = controls.limits
    mtime = os.path.getmtime(controls.path)
    case = NetworkCase(load_model(controls.path, mtime, controls.mode), controls.path, mtime,
                       controls.mode)
    filters = render_sidebar_filters(case)
    render_sidebar_footer()
    if controls.page == ABOUT_PAGE:
        render_about_page(limits)
        return

    scenario_cards = st.container()  # filled once the Config tab has been read
    render_summary(case, limits)
    if controls.run_n1 and case.solved:
        selection = n1_selection(case, filters)
        n1 = n1_results(case, controls, selection)
        if n1 is None:
            st.info("No in-service circuits or transformers match the voltage level and "
                    "transmission owner filters, so N-1 was not run.")
        else:
            case.add_n1(n1)
            render_n1_summary(case, controls, selection)

    left, right = st.columns(MAIN_COLUMNS, gap="large")
    # The right-hand tabs render first: the connection screen's result is drawn on the map,
    # and the step columns it adds to the branch table are read by every later section.
    with right:
        detail_tab, network_tab, demand_tab, generation_tab, centres_tab, config_tab = st.tabs(
            ["Substation detail", "Network", "Demand", "Generation", "Data centres", "Config"])
        with centres_tab:
            highlight = render_data_centres_tab(case, limits)
        with network_tab:
            render_network_tab(case)
        with demand_tab:
            render_demand_tab(case)
        with generation_tab:
            render_generation_tab(case)
        with config_tab:
            scenario = render_config_tab(controls.path)
    with scenario_cards:
        render_scenario_cards(scenario)
    with left:
        selected = render_map(case, controls, filters, highlight)
    with detail_tab:
        render_substation_detail(case, selected)

    charts_tab, issues_tab = st.tabs(["Loading charts", "Elements to look at"])
    with charts_tab:
        if case.solved:
            render_charts(case, limits, filters.owners, highlight)
        else:
            st.info("Run a load flow to see the charts.")
    with issues_tab:
        if case.solved:
            render_issues(case, limits, filters.owners)
        else:
            st.info("Run a load flow to see the elements to look at.")


if __name__ == "__main__":
    main()
