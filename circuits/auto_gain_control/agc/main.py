import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from calcs.metric_notation import to_metric, from_metric
from calcs.inductance import Inductance
from calcs.antenna import Antenna

ant = Antenna()
ant.impedance = '1000'
ant.resistance = '100'

frequency = '1M'

ant.inductance = Inductance.From.series_zrf(ant.impedance, ant.resistance, frequency)

print(f'Antenna Charactristics:')
print(f'\tTarget Inductance:\t{ant.inductance}')
print(f'\tFrom:')
print(f'\t- Target Freq:\t\t{frequency}Hz')
print(f'\t- Impedance:\t\t{ant.impedance}Ω')
print(f'\t- Ser Res:\t\t{ant.resistance}Ω')

