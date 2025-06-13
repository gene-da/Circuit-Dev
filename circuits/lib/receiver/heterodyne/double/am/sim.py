from tools.spice.spice_init import SpiceProject
from tools.plotter import generic

project = SpiceProject('output', 'reciever.cir')
project.run()
project.read_raw()

signals = []