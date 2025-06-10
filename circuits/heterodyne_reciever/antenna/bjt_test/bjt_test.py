import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.editor.spice_editor import SpiceEditor
from spicelib.raw.raw_read import RawRead
from tools import plotter as plot
from tools import metric_notation as mn

# Get the absolute path to the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Set up relative paths ---
# Path to the output directory
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Netlist
netlist = os.path.join(script_dir, 'bjt_test.cir')

# Sim instance
simulator = NGspiceSimulator.create_from('ngspice.exe')
runner = SimRunner(output_folder=output_dir, simulator=simulator)

# Run Simulation
runner.run(netlist)
runner.wait_completion()

print("Files in output directory:")
for file in os.listdir(output_dir):
    print("-", file)
    
raw_path = os.path.join(output_dir, "bjt_test_1.raw")
raw = RawRead(raw_path)

print("Available traces:", raw.get_trace_names())

frequency = raw.get_trace('frequency')

v_in = raw.get_trace('v(vin)').data
v_out = raw.get_trace('v(out)').data
v_out_db = 20 * np.log10(np.abs(v_out))

v_base = raw.get_trace('v(base)')
v_base_db = 20 * np.log10(np.abs(v_base))

v_collector = raw.get_trace('v(collector)')
v_collector_db = 20 * np.log10(np.abs(v_collector))

v_emitter = raw.get_trace('v(emitter)')
v_emitter_db = 20 * np.log10(np.abs(v_emitter))

if v_in is None:
    raise ValueError("v_in trace is missing!")
if np.any(np.abs(v_in) == 0):
    raise ValueError("v_in contains zero values — cannot divide!")

gain = 20 * np.log10(np.abs(v_out / v_in))



# --- Call your AC sweep plotter ---
plot.plot_ac_sweep(
    data=[(frequency, v_out_db), (frequency, v_collector_db), (frequency, v_emitter_db), (frequency, v_base_db)],
    # data=[(frequency, gain)],
    labels=["out", 'collector', 'emitter', 'base'],
    # labels=['Gain'],
    title="BJT Test: BC547b - Gain Test",
    x_limit_high='1.1G',
    y_limit_high='20',
    y_limit_low= '-80',
    x_limit_low= '1'
)

v_base = raw.get_trace('v(base)')
v_collector = raw.get_trace('v(collector)')
v_emitter = raw.get_trace('v(emitter)')

# plot.plot_transient(
#     data=[(time, v_out)],
#     labels=["V(out)"],
#     title="Transient Response of V(out)",
#     x_center="1.5m",
#     x_divs=10,
#     y_center=0,
#     y_divs=8
# )