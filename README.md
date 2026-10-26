DCConnect Network Visualiser
============================

Streamlit viewer for a pandapower model of the GB transmission network: load flow,
N-1 screening and data centre connection screening on an interactive map.

This repository holds only what the viewer needs to run. The model itself is built
elsewhere and committed here as a pandapower JSON in `models/`.

# Run locally

Python 3.11:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run network_viewer.py
```

Smoke tests (they run the app headlessly, including an AC load flow and an N-1 screen,
so allow several minutes):

```
python -m pytest tests -v
```

# Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. At https://share.streamlit.io choose **Create app**, pick the repository and branch,
   and set the main file path to `network_viewer.py`.
3. Under **Advanced settings** choose Python 3.11. No secrets are needed.

# Layout

```
network_viewer.py        the Streamlit app
model_adapters.py        presents the pandapower network and its results to the app
viewer_charts.py         Plotly charts and the brand palette
network_checks.py        dispatch, AC/DC load flow, N-1 contingency runs, demand steps
n1_screening.py          N-1 screen with line outage distribution factors (LODFs)
linearised_network.py    PTDF/LODF linearisation of a pandapower network
*.geojson                coastline, ETYS boundaries and DC Connect FLOP zones drawn on the map
*.png                    logos in the page header
models/                  pandapower network JSON files; the newest is selected by default
.streamlit/config.toml   theme
tests/                   AppTest smoke test
```

# Updating the model

Copy a new `*_pandapower.json` into `models/` and push. The app lists every JSON in that
folder, newest first.
