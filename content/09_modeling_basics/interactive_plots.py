# %% [code]
# imports
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
import psutil
from hnn_core import (
    MPIBackend,
    neymotin_2020_model,
    read_dipole,
    simulate_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model

# %% [code]
# helper functions


def resample(
    original_times,
    data,
    updated_times,
):
    """
    Resample an array onto a new set of time points
    """
    original_times = np.asarray(
        original_times,
        float,
    )
    data = np.asarray(
        data,
        float,
    )
    updated_times = np.asarray(
        updated_times,
        float,
    )

    # a single array vs. a stack of arrays, one per slider position
    if data.ndim == 1:
        resampled = np.interp(
            updated_times,
            original_times,
            data,
        )
    else:
        resampled = np.stack(
            [np.interp(updated_times, original_times, row) for row in data],
        )

    return resampled


def build_slider_payload(
    x,
    series,
    *,
    slider_values,
    slider_name,
    slider_label,
    xlabel,
    ylabel,
    ylabel_template="{label} x {value}",
    title=None,
    initial=0,
    tick_labels=None,
    symmetric_y=True,
    zero_line=True,
    y_domain=None,
    width=720,
    height=360,
):
    """Collect everything the figure needs into one package."""
    x_values = np.asarray(
        x,
        float,
    ).tolist()
    n_points = len(x_values)
    n_frames = len(slider_values)

    # make sure every line matches the time points and the slider
    prepared_series = []
    for s in series:
        data = np.asarray(
            s["data"],
            float,
        )
        if data.ndim == 2 and data.shape[0] != n_frames:
            raise ValueError(
                f"{s['label']}: {data.shape[0]} frames != {n_frames} slider values",
            )
        if data.shape[-1] != n_points:
            raise ValueError(
                f"{s['label']}: {data.shape[-1]} points != {n_points} x values",
            )
        prepared_series.append(
            {
                **s,
                "data": data.tolist(),
            },
        )

    # make sure the slider settings make sense
    if not 0 <= initial < n_frames:
        raise ValueError(f"initial={initial} out of range")
    if tick_labels is not None and len(tick_labels) != n_frames:
        raise ValueError("tick_labels length must match slider_values")

    payload = {
        "title": title,
        "width": width,
        "height": height,
        "x": {
            "label": xlabel,
            "values": x_values,
        },
        "y": {
            "label": ylabel,
            "label_template": ylabel_template,
            "symmetric": symmetric_y,
            "zero_line": zero_line,
            "domain": y_domain,
        },
        "slider": {
            "name": slider_name,
            "label": slider_label,
            "values": list(slider_values),
            "tick_labels": tick_labels,
            "initial": initial,
        },
        "series": prepared_series,
    }

    return payload


def round_floats(
    obj,
    sig,
):
    """
    Go through floats, lists, and dicts and round any floats
    """
    if isinstance(obj, float):
        rounded = float(f"{obj:.{sig}g}")
    elif isinstance(obj, list):
        rounded = [round_floats(item, sig) for item in obj]
    elif isinstance(obj, dict):
        rounded = {key: round_floats(value, sig) for key, value in obj.items()}
    else:
        rounded = obj

    return rounded


def write_widget_js(
    payload,
    out_path,
    sig=None,
):
    """
    Write the payload to a ".js" file that Textbook markdown pages can load via a
    <div data-src=""> element
    """
    out_path = Path(out_path)

    # optionally round values to keep the filesize small
    if (not isinstance(sig, bool)) and (isinstance(sig, int)) and (sig >= 0):
        payload = round_floats(
            payload,
            sig,
        )
    json_payload = json.dumps(
        payload,
        separators=(",", ":"),
    )

    # save the payload as a javascript file that registers the data under the filename
    # - this allows us to import the script with html using (for example):
    #   `<div class="hnn-slider" data-src="scaling_widget.js">`
    out_path.write_text(
        "window.HNN_WIDGET_DATA = window.HNN_WIDGET_DATA || {};\n"
        f'window.HNN_WIDGET_DATA["{out_path.name}"] = {json_payload};\n',
        encoding="utf-8",
    )

# %% [code]
# fetch MEG data
data_url = (
    "https://raw.githubusercontent.com/jonescompneurolab/hnn-data/"
    "main/MEG_detection_data/yes_trial_S1_ERP_all_avg.txt"
)

with TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir) / "threshold_data.txt"
    urlretrieve(
        data_url,
        temp_path,
    )
    threshold_experimental_dpl = read_dipole(temp_path)

# %% [code]
# set up simulation
net = neymotin_2020_model()
add_erp_drives_to_jones_model(net)

tstop = 170.0
dt = 0.025
n_trials = 1
ncores = psutil.cpu_count(logical=False) - 1


# %% [code]
# run simulation
with MPIBackend(ncores):
    dipoles_list = simulate_dipole(
        net=net,
        tstop=tstop,
        n_trials=n_trials,
        dt=dt,
    )

# %% [code]
# --------------------------------------------------
# Interactive plot for adjusting dipole scaling
# --------------------------------------------------

# meta params
save_js = True

# get threshold times, data
experimental_times = threshold_experimental_dpl.times.tolist()
experimental_data = threshold_experimental_dpl.data["agg"].tolist()

# get the simulation times, dipole
dipole_01 = deepcopy(dipoles_list[0])
simulation_times = dipole_01.times

# select the time array to use for downsampling
if len(simulation_times) > len(experimental_times):
    time_array = experimental_times
else:
    time_array = simulation_times

# define param values for interactive slider
scale_values = [
    1,
    10,
    100,
    500,
    1000,
    1500,
    3000,
    4500,
    6000,
]

# generate the interactive data series
# - in this instance, generating the data does not require simulating the
#   network, as scaling is an ad hoc transformation
dipole_data = {}
for value in scale_values:
    scaled_dipole = dipole_01.copy().scale(value)
    resampled = resample(
        simulation_times,
        scaled_dipole.data["agg"],
        time_array,
    )
    dipole_data[str(value)] = resampled

experimental_data = resample(
    experimental_times,
    experimental_data,
    time_array,
)

# generate payload and save to ".js"
payload = build_slider_payload(
    time_array,
    series=[
        {
            "label": "Simulation Dipole",
            "data": list(dipole_data.values()),
            "color": "#1f77b4",
        },
        {
            "label": "Experimental Dipole",
            "data": experimental_data,
            "color": "#ff7f0e",
        },
    ],
    slider_values=list(dipole_data.keys()),
    slider_name="scaling",
    slider_label="Dipole Scaling Factor",
    xlabel="Time (ms)",
    ylabel="Dipole (nAm)",
    initial=2,
)

if save_js:
    if "__file__" in globals():
        out_dir = Path(__file__).parent
    # fall back to the cwd (e.g., this may be triggered in a jupyter environment)
    else:
        out_dir = Path.cwd()

    write_widget_js(
        payload,
        out_dir / "scaling_widget.js",
        sig=5,
    )

# %% [code]
