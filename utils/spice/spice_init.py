import os
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.editor.spice_editor import SpiceEditor

class SpiceProject:
    def __init__(self, output=None, netlist=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.simulator = NGspiceSimulator.create_from('ngspice.exe')
        self.netlist = None
        self.output = None
        self.runner = None
        if output:
            self.set_output(output)
            self.set_runner()
        if netlist:
            self.set_netlist()
        
    def set_output(self, output_path: str):
        self.output = output_path
        os.makedirs(self.output, exist_ok=True)
        
    def set_netlist(self, netlist: str):
        if self.output is None:
            raise ValueError(f'Did not set output directory')
        else:
            self.netlist = SpiceEditor(os.path.join(self.output, netlist))
            
    def set_runner(self):
        if self.output is None:
            raise ValueError(f'Did not set output directory')
        else:
            self.runner = SimRunner(output_folder=self.output, simulator=self.simulator)
            
    def run(self):
        self.runner.run(self.netlist)

        
    
if __name__ == "main":
    project = SpiceProject('./output', 'reciever.cir')
    project.set_output('./output')
        