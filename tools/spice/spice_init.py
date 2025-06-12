import os
import shutil
import numpy as np
from dataclasses import dataclass
from tools.spice.spicelib.sim.sim_runner import SimRunner
from tools.spice.spicelib.simulators.ngspice_simulator import NGspiceSimulator
from tools.spice.spicelib.editor.spice_editor import SpiceEditor
from tools.spice.spicelib.raw.raw_read import RawRead
from tools import metric_notation as mn
from tools import plotter as plot

@dataclass
class SpiceData:
    name: any
    data: any

class SpiceProject:        
    def __init__(self, output=None, netlist=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.simulator = NGspiceSimulator.create_from(r'C:\Users\eugene.dann\Documents\dev\sims\ngspice.exe')
        self.netlist = None
        self.output = None
        self.runner = None
        self.reader = None
        self.raw = []
        self._output_files = []
        
        if output:
            self.set_output(output)
            self.__clear_output__()
            self.set_runner()
        if netlist:
            self.set_netlist(netlist=netlist)
        
    def set_output(self, output_path: str):
        self.output = os.path.join(self.script_dir, output_path)
        os.makedirs(self.output, exist_ok=True)
        
        if os.path.exists(self.output):
            print(f"Directory Created:\t✅\t{output_path}")
        else:
            print(f"Failed to create directory '{self.output}'.")
        
    def set_netlist(self, netlist: str):
        if self.output is None:
            raise ValueError('Did not set output directory')

        full_path = os.path.join(self.script_dir, netlist)
        
        if os.path.isfile(full_path):
            print(f'Netlist Found:\t\t✅\t{netlist}')
            self.netlist = SpiceEditor(full_path)
        else:
            raise FileNotFoundError(f'Netlist file not found at: {full_path}')
            
    def set_runner(self):
        if self.output is None:
            raise ValueError(f'Did not set output directory')
        else:
            self.runner = SimRunner(output_folder=self.output, simulator=self.simulator)
            
    def run(self):
        self.runner.run(self.netlist)
        self.runner.wait_completion()
        self.__check_output_files__()
        
    def __clear_output__(self):
        for file in os.listdir(self.output):
            path = os.path.join(self.output, file)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f'⚠️ Failed to delete {path}. Reason: {e}')
            
    def __check_output_files__(self):        
        for file in os.listdir(self.output):
            if file not in self._output_files:
                print(f' - Created:\t\t✅\t{file}')
                self._output_files.append(file)
            
    def read_raw(self):
        count = 0
        for file in os.listdir(self.output):
            if file.endswith('.raw'):
                count += 1
                path = os.path.join(self.output, file)
                raw = RawRead(path)
                self.raw.append(SpiceData(name=f'Sim{count}', data=raw))
        
        # print(self.raw)
                
    def component(self, component: str, value: str=None):
        if value:
            print(f' - Changed:\t\t❗️\t {component}: {mn.tm(self.netlist[component].value)} ➡️  {mn.tm(mn.fm(value))}')
            self.netlist[component].value = mn.fm(value)
        else:
            return mn.tm(self.netlist[component].value)
        
    def get_data(self, trace: str, sim_type: str):
        data_output = []
        count = 0
        for sim in self.raw:
            count += 1
            read = any
            x = any
            if sim_type == 'tran':
                x = np.real(np.array(sim.data.get_trace('time')))
            if sim_type == 'ac':
                x = np.real(np.array(sim.data.get_trace('frequency')))
            
            y = np.array(sim.data.get_trace(trace))
                
            data_output.append(SpiceData(name=f'Sim {count}', data=(x, y)))
        
        return data_output

# Test Scripts
project = SpiceProject('output', 'reciever.cir')
project.netlist.set_parameter('fc', '450k')
project.run()
project.netlist.set_parameter('fc', '775k')
project.run()
project.netlist.set_parameter('fc', '1000k')
project.run()
project.netlist.set_parameter('fc', '1350k')
project.run()
project.netlist.set_parameter('fc', '1700k')
project.run()
project.read_raw()
am = project.get_data('v(N001)', 'tran')
signals = []
labels = ['450k', '775k', '1000k', '1350k', '1700k']


for signal in am:
    x, y = signal.data  # each .data is a (time, voltage) tuple
    signals.append((x, y))  # this is what plot.transient expects

plot.generic(signals, labels)