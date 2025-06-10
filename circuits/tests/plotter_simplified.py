import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tools.plotter import Plotter

# Simulated data
time = np.linspace(0, 100e-6, 1000)
frequencies = [50e3, 75e3, 100e3]
data = [2.5 * np.sin(2 * np.pi * f * time) for f in frequencies]
labels = [f"CH{i+1} - {int(f/1e3)}kHz" for i, f in enumerate(frequencies)]

table_data = [
    {"Label": labels[i], "Freq": f"{f/1e3:.2f}k", "Amplitude": "2.50 V"}
    for i, f in enumerate(frequencies)
]

# Plotting
plot = Plotter()
plot.set(data, time, labels=labels)
plot.add_table_data(table_data)
plot.create(
    view_params=(0.0, 1.0, 50e-6, 10e-6)
)

