import os
import numpy as np
import matplotlib.pyplot as plt
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.editor.spice_editor import SpiceEditor
from spicelib.raw.raw_read import RawRead
from spicelib.utils.sweep_iterators import sweep_log
from utils import metric_notation as mn

# Output Directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'preamp_test')
os.makedirs(output_dir, exist_ok=True)

# Netlist
netlist = os.path.join(script_dir, 'reciever.cir')

# Sim Instance
simulator = NGspiceSimulator.create_from('ngspice.exe')
runner = SimRunner(output_folder=output_dir, simulator=simulator)