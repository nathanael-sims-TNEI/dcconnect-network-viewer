"""
Plotly charts for the detailed network viewer
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRANSMISSION_KV = (400.0, 275.0, 132.0)
BRAND = {
    "dark_blue": "#15143C",
    "mid_blue": "#035F95",
    "teal": "#008080",
    "light_blue": "#53B4DF",
    "mustard": "#C9A227",
    "grey": "#7A7F87",
}
ALERT_COLOUR = "#C1121F"
LIGHT_GREY = "#BFC3C8"
TINTS = {"n1": "#FBF3D6", "affected": "#DCEFEF"}
KV_COLOURS = {400.0: BRAND["dark_blue"], 275.0: BRAND["mustard"], 132.0: BRAND["teal"]}
OTHER_KV_COLOUR = BRAND["light_blue"]
DEMAND_COLOUR = BRAND["mid_blue"]
GENERATION_COLOUR = BRAND["mustard"]
AFFECTED_COLOUR = BRAND["teal"]
LIMIT_COLOUR = BRAND["grey"]
GRID_COLOUR = "#E6E9EE"
INK = BRAND["dark_blue"]
INK_MUTED = BRAND["grey"]
FONT = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
ROW_HEIGHT = 24
PAIRED_HEIGHT = 360  # the most-loaded and zone balance charts sit side by side
LOADING_AXIS_TITLE = "Loading (% of winter rating)"

CHART_TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=12, color=INK),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=16, t=56, b=10),
    title=dict(x=0, xanchor="left", font=dict(size=14, color=BRAND["dark_blue"])),
    xaxis=dict(gridcolor=GRID_COLOUR, zeroline=False, showline=False, ticks="",
               automargin=True, title=dict(font=dict(size=12, color=INK_MUTED))),
    yaxis=dict(gridcolor=GRID_COLOUR, zeroline=False, showline=False, ticks="",
               automargin=True, title=dict(font=dict(size=12, color=INK_MUTED))),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID_COLOUR, font=dict(family=FONT, size=12)),
    bargap=0.08,
))
# Streamlit fills in any background and font the layout leaves unset with its own theme,
# even with theme=None, so these are set on the layout itself rather than the template.
BASE_LAYOUT = dict(template=CHART_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=12, color=INK))


# The cumulative curves carry up to six legend entries, which wrap to two rows above the plot.
CURVE_LAYOUT = dict(height=340, margin=dict(l=10, r=16, t=84, b=10))
CURVE_TITLE = dict(y=0.985, yanchor="top")


def _kv_colour(kv):
    return KV_COLOURS.get(float(kv), OTHER_KV_COLOUR)


def _kv_levels(frame, column):
    return sorted(frame[column].unique(), reverse=True)


def _row_labels(frame):
    """Plotly adds up bars that share a category, so parallel circuits are numbered."""
    names = frame["name"].astype(str)
    repeats = names.groupby(names).cumcount() + 1
    duplicated = names.duplicated(keep=False)
    return names.where(~duplicated, names + " (" + repeats.astype(str) + ")") + "  "


def _rows_height(count, floor=320):
    return max(floor, ROW_HEIGHT * count + 90)


def _limit_line(fig, loading_limit):
    fig.add_vline(x=loading_limit, line=dict(color=LIMIT_COLOUR, width=1.5, dash="dash"))


def _acceptable_band(fig, x0, x1):
    fig.add_vrect(x0=x0, x1=x1, fillcolor=BRAND["teal"], opacity=0.08, line_width=0)


def transmission_circuits(branches, column="loading", kv_levels=TRANSMISSION_KV):
    rows = (branches["kind"] == "Line") & branches["kv"].isin(kv_levels)
    rows &= branches["in_service"].astype(bool) & branches[column].notna()
    return branches[rows]


def loading_cumulative(circuits, loading_limit, column="loading", overlay_column=None):
    """One curve per voltage level, and with an overlay column (the worst N-1 loading) a
    second, dashed curve per level drawn over the first."""
    series = [(column, "", "solid")]
    if overlay_column is not None:
        series.append((overlay_column, " worst N-1", "dash"))
    fig = go.Figure()
    for kv in _kv_levels(circuits, "kv"):
        at_kv = circuits[circuits["kv"] == kv]
        for source, suffix, dash in series:
            values = np.sort(at_kv[source].dropna().to_numpy(dtype=float))
            if not len(values):
                continue
            share = np.arange(1, len(values) + 1) / len(values)
            name = f"{kv:.0f}kV N-1" if suffix else f"{kv:.0f}kV ({len(values):,})"
            fig.add_trace(go.Scatter(
                x=np.concatenate([[0.0], values]), y=np.concatenate([[0.0], share]),
                mode="lines", name=name, legendgroup=f"{kv:.0f}",
                line=dict(color=_kv_colour(kv), width=2, shape="hv", dash=dash),
                hovertemplate=(f"{kv:.0f}kV{suffix}<br>%{{y:.0%}} of circuits loaded at or "
                               "below %{x:.0f}%<extra></extra>"),
            ))
    _acceptable_band(fig, 0.0, loading_limit)
    _limit_line(fig, loading_limit)
    fig.update_layout(
        **BASE_LAYOUT, **CURVE_LAYOUT,
        title=dict(text="Cumulative loading curve", **CURVE_TITLE),
        xaxis=dict(title=LOADING_AXIS_TITLE, rangemode="tozero"),
        yaxis=dict(title="Share of circuits", tickformat=".0%", range=[0, 1.02]),
        hovermode="x unified",
    )
    return fig


def most_loaded(circuits, bands, loading_limit, column="loading", intact_column=None,
                count=15):
    """The circuits with the highest loading in `column`. With an intact column (the
    chart of worst N-1 loadings) each circuit's intact loading is a grey bar beside it."""
    top = circuits.sort_values(column, ascending=False).head(count).iloc[::-1]
    labels = _row_labels(top)
    fig = go.Figure()
    if intact_column is not None:
        fig.add_trace(go.Bar(
            x=top[intact_column], y=labels, orientation="h", name="Intact network",
            marker=dict(color=LIGHT_GREY, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Intact loading %{x:.0f}%<extra></extra>",
        ))
    colours = [next(colour for limit, _label, colour in bands if value <= limit)
               for value in top[column]]
    outage = (top["n1_outage"].fillna("").map(lambda name: f" ({name} out)" if name else "")
              if column == "n1_loading" and "n1_outage" in top else [""] * len(top))
    fig.add_trace(go.Bar(
        x=top[column], y=labels, orientation="h",
        name="Worst N-1 outage" if column == "n1_loading" else "Intact network",
        marker=dict(color=colours, line=dict(width=0)),
        customdata=np.column_stack([top["kv"], top["rating_mva"], top["p_mw"].fillna(0.0),
                                    top["from_name"], top["to_name"], list(outage)]),
        hovertemplate=("<b>%{y}</b><br>%{customdata[3]} to %{customdata[4]}"
                       " (%{customdata[0]:.0f}kV)<br>Rating %{customdata[1]:,.0f}MVA, "
                       "flow %{customdata[2]:,.0f}MW<br>Loading %{x:.0f}%%{customdata[5]}"
                       "<extra></extra>"),
    ))
    _limit_line(fig, loading_limit)
    fig.update_layout(
        **BASE_LAYOUT, barmode="group", bargap=0.15, bargroupgap=0.0, height=PAIRED_HEIGHT,
        title=f"{len(top)} most loaded transmission circuits",
        xaxis=dict(title=LOADING_AXIS_TITLE, rangemode="tozero"),
        yaxis=dict(title=None, tickfont=dict(size=10)),
        showlegend=intact_column is not None,
    )
    return fig


VOLTAGE_SERIES = {
    "intact": ("vm_pu", "", "solid"),
    "n1_low": ("n1_min_vm_pu", " N-1 lowest", "dash"),
    "n1_high": ("n1_max_vm_pu", " N-1 highest", "dot"),
}


def voltage_cumulative(buses, vm_limits, series=("intact",), kv_levels=TRANSMISSION_KV):
    """One curve per voltage level for each series in `series`: the intact voltages solid,
    and the lowest and highest post-outage voltages of each node dashed and dotted."""
    located = buses[buses["vn_kv"].isin(kv_levels)]
    fig = go.Figure()
    drawn = set()
    for kv in _kv_levels(located, "vn_kv"):
        at_kv = located[located["vn_kv"] == kv]
        for key in series:
            column, suffix, dash = VOLTAGE_SERIES[key]
            if column not in at_kv:
                continue
            values = np.sort(at_kv[column].dropna().to_numpy(dtype=float))
            if not len(values):
                continue
            drawn.add(key)
            share = np.arange(1, len(values) + 1) / len(values)
            # The post-outage curves share one grey legend entry per series, added below.
            fig.add_trace(go.Scatter(
                x=values, y=share, mode="lines", legendgroup=key if suffix else f"{kv:.0f}",
                name=f"{kv:.0f}kV{suffix}" if suffix else f"{kv:.0f}kV ({len(values):,})",
                showlegend=not suffix,
                line=dict(color=_kv_colour(kv), width=2, shape="hv", dash=dash),
                hovertemplate=(f"{kv:.0f}kV{suffix}<br>%{{y:.0%}} of nodes at or below "
                               "%{x:.3f}pu<extra></extra>"),
            ))
    for key in series:
        column, suffix, dash = VOLTAGE_SERIES[key]
        if suffix and key in drawn:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines", name=suffix.strip(), legendgroup=key,
                line=dict(color=LIGHT_GREY, width=2, dash=dash), hoverinfo="skip"))
    _acceptable_band(fig, vm_limits[0], vm_limits[1])
    fig.update_layout(
        **BASE_LAYOUT, **CURVE_LAYOUT,
        title=dict(text="Cumulative voltage curve", **CURVE_TITLE),
        xaxis=dict(title="Voltage (pu)", tickformat=".2f"),
        yaxis=dict(title="Share of nodes", tickformat=".0%", range=[0, 1.02]),
        hovermode="x unified",
    )
    return fig


def zone_balance(subs, count=20):
    by_zone = (subs.groupby("zone")[["demand_mw", "generation_mw"]].sum()
               .sort_values("demand_mw", ascending=False).head(count))
    fig = go.Figure()
    for column, name, colour in [("demand_mw", "Demand", DEMAND_COLOUR),
                                 ("generation_mw", "Generation", GENERATION_COLOUR)]:
        fig.add_trace(go.Bar(
            x=by_zone.index, y=by_zone[column], name=name,
            marker=dict(color=colour, line=dict(width=0)),
            hovertemplate=f"%{{x}}<br>{name} %{{y:,.0f}}MW<extra></extra>",
        ))
    fig.update_layout(
        **BASE_LAYOUT, barmode="group", bargroupgap=0.08, height=PAIRED_HEIGHT,
        title=f"Demand and generation in the {len(by_zone)} FLOP zones with the most demand",
        xaxis=dict(title=None, tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(title="MW", tickformat=",.0f"),
    )
    return fig


def screen_waterfall(moved, step_mw, count=20):
    top = moved.sort_values("delta", ascending=False).head(count).iloc[::-1]
    labels = _row_labels(top)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["loading"], y=labels, orientation="h", name="Before",
        marker=dict(color=LIGHT_GREY, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Before %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=top["delta"].clip(lower=0), base=top["loading"], y=labels, orientation="h",
        name=f"Rise with +{step_mw:,.0f}MW",
        marker=dict(color=np.where(top["affected"], AFFECTED_COLOUR, BRAND["light_blue"]),
                    line=dict(width=0)),
        customdata=np.column_stack([top["step_loading"], top["delta"]]),
        hovertemplate=("<b>%{y}</b><br>After %{customdata[0]:.1f}% "
                       "(+%{customdata[1]:.1f} points)<extra></extra>"),
    ))
    fig.update_layout(
        **BASE_LAYOUT, barmode="overlay", height=_rows_height(len(top), 300),
        title="Branches with the largest loading rise",
        xaxis=dict(title="Loading (% of rating)", rangemode="tozero"),
        yaxis=dict(title=None, tickfont=dict(size=11)),
    )
    return fig


def empty_chart(message, height=320):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=13, color=INK_MUTED),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(**BASE_LAYOUT, height=height,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def describe_loading(circuits, column="loading"):
    values = circuits[column].astype(float)
    if values.empty:
        return pd.Series(dtype=float)
    return pd.Series({
        "circuits": float(len(values)), "mean": values.mean(), "median": values.median(),
        "p95": values.quantile(0.95), "max": values.max(),
    })
