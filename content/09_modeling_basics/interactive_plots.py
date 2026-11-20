# %% [code]
# imports
import json
from copy import deepcopy
from itertools import product
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
from hnn_core.dipole import _rmse
from hnn_core.network_models import add_erp_drives_to_jones_model
from matplotlib.colors import to_hex

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
    series_arrays = []
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
        series_arrays.append(data)
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

    # fill in the slider labels if none were given
    if tick_labels is None:
        tick_labels = [str(v) for v in slider_values]

    # determine the y axis label at each slider position
    y_labels = [
        ylabel_template.format(
            label=ylabel,
            value=tick,
        )
        for tick in tick_labels
    ]

    # determine the y axis range at each slider position
    if y_domain is not None:
        y_domains = [list(y_domain) for _ in range(n_frames)]
    else:
        y_domains = []
        for idx in range(n_frames):
            combined = np.concatenate(
                [arr[idx] if arr.ndim == 2 else arr for arr in series_arrays],
            )
            if symmetric_y:
                max_abs = (float(np.max(np.abs(combined))) or 1.0) * 1.05
                y_domains.append([-max_abs, max_abs])
            else:
                y_domains.append([float(np.min(combined)), float(np.max(combined))])

    payload = {
        "title": title,
        "width": width,
        "height": height,
        "x": {
            "label": xlabel,
            "values": x_values,
        },
        "y": {
            "labels": y_labels,
            "zero_line": zero_line,
            "domains": y_domains,
        },
        "slider": {
            "name": slider_name,
            "label": slider_label,
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


def count_drive_spikes(
    cell_response,
    drive_groups,
):
    """Count the spikes from each group of drives in each time bin."""
    spike_times = np.concatenate(cell_response.spike_times)
    spike_types = np.concatenate(cell_response.spike_types).astype(str)
    bin_edges = np.linspace(
        0,
        cell_response.times[-1],
        50,
    )

    # count the spikes in each time bin for each group of drives
    counts = {}
    for label, prefixes in drive_groups.items():
        in_group = np.zeros(
            len(spike_types),
            dtype=bool,
        )
        for prefix in prefixes:
            in_group |= np.char.startswith(
                spike_types,
                prefix,
            )
        group_counts, _ = np.histogram(
            spike_times[in_group],
            bin_edges,
        )
        counts[label] = group_counts

    histogram = {
        "bin_edges": bin_edges,
        "counts": counts,
    }

    return histogram


def build_drive_dipole_payload(
    *,
    hist_bin_edges,
    drive_counts,
    drive_colors,
    inverted_groups,
    dipole_times,
    simulated,
    experimental,
    scaling,
    rmse_labels,
    sliders,
    xdomain,
    xlabel="Time (ms)",
    hist_title="External Drive Spike Arrival Times",
    hist_ylabel="Counts",
    dipole_ylabel="Dipole (nAm)",
    simulated_label="Simulation",
    simulated_color="#1f77b4",
    experimental_label="Experimental data",
    experimental_color="#ff7f0e",
    width=720,
    height=640,
):
    # set up
    # ------------------------------
    # cast histogram bin_edges to an array
    hist_bin_edges = np.asarray(hist_bin_edges, float)

    # get histogram bin counts
    hist_n_bins = len(hist_bin_edges) - 1

    # cast dipoles to arrays
    simulated = np.asarray(simulated, float)
    experimental = np.asarray(experimental, float)

    # get the two slider lengths
    slider_1, slider_2 = sliders
    len_slider_1 = len(slider_1["tick_labels"])
    len_slider_2 = len(slider_2["tick_labels"])

    # `drive_counts` maps each *grouped* drive label ("Proximal", "Distal") to an array
    # whose dimensions are (len_slider_1, len_slider_2, n_hist_bins)
    #
    # for example: if slider 1 (i) has 5 positions and slider 2 (j) has 4 positions,
    # `counts` must have shape (i=5, j=4, n_hist_bins). `counts[i, j]` returns the bins
    # (spike counts) for slider 1 at index `i` and slider 2 at index `j`
    #
    # this 3D grid is *always* necessary because the "front end" (javascript) does not
    # know whether one, both, or neither slider parameters will affect the histogram
    # output of the simulation, and so the histogram data are needed for every
    # slider permutation
    groups = []
    for label, counts in drive_counts.items():
        counts = np.asarray(counts)
        if counts.shape != (
            len_slider_1,
            len_slider_2,
            hist_n_bins,
        ):
            raise ValueError(f"{label}: counts shape {counts.shape} does not fit")
        groups.append(
            {
                "label": label,
                "color": to_hex(drive_colors[label]),
                "inverted": label in inverted_groups,
                "counts": counts.tolist(),
            },
        )

    # sanity checks
    # ------------------------------
    if simulated.shape != (
        len_slider_1,
        len_slider_2,
        len(dipole_times),
    ):
        raise ValueError(f"simulated shape {simulated.shape} does not fit")
    if len(experimental) != len(dipole_times):
        raise ValueError("experimental data must match the time points")
    if np.asarray(rmse_labels).shape != (len_slider_1, len_slider_2):
        raise ValueError("rmse_labels must have one entry per slider pair")
    for slider in sliders:
        if not 0 <= slider.get("initial", 0) < len(slider["tick_labels"]):
            raise ValueError(f"{slider['label']}: initial is out of range")

    # prepare payload
    # ------------------------------
    # use one y-axis range for every slider position so the axes don't jump
    hist_max = max(np.max(group["counts"]) for group in groups)
    hist_top = (float(hist_max) or 1.0) * 1.05
    dipole_max = max(
        float(np.max(np.abs(simulated))),
        float(np.max(np.abs(experimental))),
    )
    dipole_top = (dipole_max or 1.0) * 1.05

    dipole_ylabel += f" x {scaling}"

    payload = {
        "width": width,
        "height": height,
        "x": {
            "label": xlabel,
            "domain": list(xdomain),
        },
        "sliders": sliders,
        "hist": {
            "title": hist_title,
            "ylabel": hist_ylabel,
            "bin_edges": hist_bin_edges.tolist(),
            "domain": [0, hist_top],
            "groups": groups,
        },
        "dipole": {
            "ylabel": dipole_ylabel,
            "times": np.asarray(dipole_times, float).tolist(),
            "domain": [-dipole_top, dipole_top],
            "simulated": {
                "label": simulated_label,
                "color": simulated_color,
                "frames": simulated.tolist(),
            },
            "experimental": {
                "label": experimental_label,
                "color": experimental_color,
                "data": experimental.tolist(),
            },
            "rmse": np.asarray(rmse_labels).tolist(),
        },
    }

    return payload


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
# --------------------------------------------------
# Interactive plot for adjusting drive mu and sigma
# --------------------------------------------------

save_js = True

drive_name = "evdist1"
dipole_scaling = 3000
dipole_smoothing = 30

mean_values = [
    49,
    56,
    63,
    70,
    77,
]
std_values = [
    1,
    2,
    3,
    4,
    5,
]

drive_groups = {
    "Proximal": ["evprox"],
    "Distal": ["evdist"],
}

sim_net = neymotin_2020_model()
add_erp_drives_to_jones_model(sim_net)

# run one simulation for every pair of mean and standard deviation
flat_counts = {label: [] for label in drive_groups}
flat_dipoles = []
rmse_labels = []

with MPIBackend(ncores):
    for mean, std in product(
        mean_values,
        std_values,
    ):
        drive_net = deepcopy(sim_net)
        drive_net.external_drives[drive_name]["dynamics"]["mu"] = mean
        drive_net.external_drives[drive_name]["dynamics"]["sigma"] = std

        dipoles = simulate_dipole(
            net=drive_net,
            tstop=tstop,
            n_trials=n_trials,
            dt=dt,
        )
        dipole = dipoles[0].scale(dipole_scaling).smooth(dipole_smoothing)

        # count the drive spikes in each time bin
        histogram = count_drive_spikes(
            drive_net.cell_response,
            drive_groups,
        )
        for label in drive_groups:
            flat_counts[label].append(histogram["counts"][label])

        # compare to the experimental data, then use fewer time points
        rmse = _rmse(
            dipole,
            threshold_experimental_dpl,
            0.0,
            tstop,
        )
        rmse_labels.append(f"RMSE: {rmse:.4f}")
        flat_dipoles.append(
            resample(
                dipole.times,
                dipole.data["agg"],
                experimental_times,
            ),
        )

# arrange the results as mean x standard deviation
n_means = len(mean_values)
n_stds = len(std_values)

drive_counts = {
    label: np.asarray(counts).reshape(
        n_means,
        n_stds,
        -1,
    )
    for label, counts in flat_counts.items()
}
simulated = np.asarray(flat_dipoles).reshape(
    n_means,
    n_stds,
    -1,
)
rmse_labels = (
    np.asarray(rmse_labels)
    .reshape(
        n_means,
        n_stds,
    )
    .tolist()
)

payload = build_drive_dipole_payload(
    hist_bin_edges=histogram["bin_edges"],
    drive_counts=drive_counts,
    drive_colors={
        "Proximal": "r",
        "Distal": "g",
    },
    inverted_groups=["Distal"],
    dipole_times=experimental_times,
    simulated=simulated,
    experimental=experimental_data,
    scaling=dipole_scaling,
    rmse_labels=rmse_labels,
    sliders=[
        {
            "label": "Mean arrival time",
            "tick_labels": [str(v) for v in mean_values],
            "initial": 2,
        },
        {
            "label": "Std. deviation",
            "tick_labels": [str(v) for v in std_values],
            "initial": 2,
        },
    ],
    xdomain=[0, tstop],
)

if save_js:
    # fall back to the current folder when run as a notebook
    out_dir = Path.cwd()
    if "__file__" in globals():
        out_dir = Path(__file__).parent

    write_widget_js(
        payload,
        out_dir / "drive_dipole_widget.js",
    )

# %% [code]
