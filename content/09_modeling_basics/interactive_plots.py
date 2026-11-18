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
# prepare MEG data
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

threshold_times = threshold_experimental_dpl.times.tolist()
threshold_data = threshold_experimental_dpl.data["agg"].tolist()


# %% [code]
# simulate set up
net = neymotin_2020_model()
add_erp_drives_to_jones_model(net)

tstop = 170.0
dt = 0.025
n_trials = 1
ncores = psutil.cpu_count(logical=False) - 1


# %% [code]
# run the simulation
with MPIBackend(ncores):
    dipoles_list = simulate_dipole(
        net=net,
        tstop=tstop,
        n_trials=n_trials,
        dt=dt,
    )


# %% [code]
# interactive plot for adjusting dipole scaling
save_json = False
save_js = True

dipole_01 = deepcopy(dipoles_list[0])
sim_times = dipole_01.times

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

dipole_data = {}
for value in scale_values:
    scaled_dipole = deepcopy(dipole_01)
    scaled_dipole = scaled_dipole.scale(value)
    dipole_data[str(value)] = scaled_dipole.data["agg"].tolist()

# downsample to the smaller data
if len(sim_times) > len(threshold_times):
    time_array = threshold_times
    for key in dipole_data:
        dipole_data[key] = np.interp(
            time_array,
            sim_times,
            dipole_data[key],
        ).tolist()
else:
    time_array = sim_times
    for key in dipole_data:
        dipole_data[key] = dipole_data[key].tolist()
    threshold_data = np.interp(
        time_array,
        threshold_times,
        threshold_data,
    ).tolist()

payload = {
    "experimental_data": threshold_data,
    "xlab": "Time (ms)",
    "ylab": "Dipole (nAm)",
    "x": time_array,
    "y": dipole_data,
    "slider": {
        "name": "scaling",
        "label": "Dipole Scaling Factor",
        "values": scale_values,
    },
}

if save_json:
    with open("scaling_widget.json", "w") as f:
        json.dump(
            payload,
            f,
            separators=(",", ":"),
        )

if save_js:
    out_path = Path(__file__).parent / "scaling_widget.js"
    out_path.write_text(
        f"window.HNN_WIDGET_DATA = window.HNN_WIDGET_DATA || {{}};\n"
        f'window.HNN_WIDGET_DATA["scaling_widget.js"] = {json.dumps(payload)};\n',
        encoding="utf-8",
    )

# %% [code]
