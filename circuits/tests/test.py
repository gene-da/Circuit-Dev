import os
import numpy as np
import matplotlib.pyplot as plt
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.editor.spice_editor import SpiceEditor
from spicelib.raw.raw_read import RawRead
from spicelib.utils.sweep_iterators import sweep_log
from utils import metric_notation as mn


def plot_ac_sweep(data, labels, title):
    plt.style.use(r'utils/theme/tokyonight.mplstyle')
    plt.figure(figsize=(10, 6))
    for (freq, trace), label in zip(data, labels):
        plt.plot(freq, trace, label=label)

    plt.xscale('log')
    plt.xlim(10, mn.fm('10G'))
    plt.ylim(-60, 5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|V(out)| [dB]")
    plt.title(title)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Output directory
outdir = "./rc_sweep_output"
os.makedirs(outdir, exist_ok=True)

# Netlist path
netlist_path = os.path.join(outdir, "rc_filter.cir")

# Create base netlist file (will be edited in-place)
with open(netlist_path, 'w') as f:
    f.write("* RC Low-Pass Filter\n")
    f.write(".option rawfmt=ps\n")
    f.write("Vin in 0 AC 1\n")
    f.write("R1 in out 200\n")
    f.write("C1 out 0 1u\n")  # Initial value will be overridden in loop
    f.write(".ac dec 100 10 100G\n")
    f.write(".save V(out)\n")
    f.write(".end\n")

# Load and prepare netlist
netlist = SpiceEditor(netlist_path)

# Set up simulator
simulator = NGspiceSimulator.create_from("ngspice.exe")  # Adjust path if needed
runner = SimRunner(output_folder=outdir, simulator=simulator)

# Sweep capacitor values (logarithmic scale)
cap_values = sweep_log(mn.fm('1n'), mn.fm('1000n'), 2)
all_data = []  # list of (freq, vout_db) tuples
labels = []

for cval in cap_values:
    print(f"Running simulation for C1 = {mn.tm(cval)} F")
    netlist["C1"].value = cval

    runner.run(netlist)
    runner.wait_completion()

    raw_path = os.path.join(outdir, f"rc_filter_{runner.runno}.raw")
    raw = RawRead(raw_path)

    freq = np.real(np.array(raw.get_trace("frequency")))
    vout = np.array(raw.get_trace("v(out)"))

    valid = freq > 0
    freq = freq[valid]
    vout = vout[valid]

    vout_mag = np.abs(vout)
    vout_mag = np.clip(vout_mag, 1e-20, None)
    vout_db = 20 * np.log10(vout_mag)

    all_data.append((freq, vout_db))
    labels.append(f"C={mn.tm(cval)} F")

# Plot all frequency responses
plot_ac_sweep(all_data, labels, "RC Sweep Test")
