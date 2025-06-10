import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.plotter import Plotter

import numpy as np
import matplotlib.pyplot as plt

plot = Plotter()

# Signal setup
x = np.linspace(0, 100e-6, 1000)
frequencies = [50e3, 75e3, 100e3, 125e3, 150e3, 175e3]
labels = [f"CH{i+1} - {int(f/1e3)}kHz" for i, f in enumerate(frequencies)]

# Table data
table_data = [
    {"Label": labels[i], "Freq": f"{f/1e3:.2f}k", "Amplitude": "2.50 V"}
    for i, f in enumerate(frequencies)
]

# Table layout setup
table_width = 0.3
fig, ax = plt.subplots(figsize=(14 + table_width, 8))


# Color cycle from style
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot lines
for i, (f, label) in enumerate(zip(frequencies, labels)):
    y = 2.5 * np.sin(2 * np.pi * f * x)
    ax.plot(x, y, label=label, color=style_colors[i % len(style_colors)])

# Grid, legend, table
plot.lock_grid(ax, x_range=87e-6, y_range=8, x_divs=10, y_divs=8)
plot.create_legend(ax, location='lower left', font_size=10)
plot.add_table(fig, ax, table_data, table_width=table_width, font_size=10, row_height=0.05)

plt.show()