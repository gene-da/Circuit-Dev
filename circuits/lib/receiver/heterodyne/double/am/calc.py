from tools.resonant import *
from tools import metric_notation as mn
from tabulate import tabulate
from tools.calcs_output import save_markdown_table as save
import numpy as np

coils = []
min_l = int(mn.fm('1u') * 1e6)
max_l = int(mn.fm('300u') * 1e6)
step = 1

coil_range = range(min_l, max_l+1, step)

for coil_val in coil_range:
    coil = coil_val * 1e-6
    coils.append(mn.tm(coil, 0))


freq = mn.fm('1000k')

data = [
    ['Inductance (H)', 'Resonant Freq (Hz)', 'Capacitance (F)', 'Impedance (Ω)']
]

def parallel_z(res, cap, ind, frq):
    r = mn.fm(res)
    c = mn.fm(cap)
    l = mn.fm(ind)
    f = mn.fm(freq)
    
    w = 2*np.pi*f
    
    ri = 1 / r if r != 0 else 0
        
    li = 1/(w*l)
    ci = -1 / (w * l) if l != 0 else 0
    
    react = ci**2
    
    if np.abs(react) < 1e-12:
        return float('inf')
    
    z = 1/np.sqrt(np.abs(ri+react))
    return z

for coil in coils:
    cap = res_c(coil, freq)
    imp = parallel_z('50', cap, coil, freq)
    data.append([f'{coil}', f'{mn.tm(freq, 0)}', f'{mn.tm(cap, 1)}', f'{mn.tm(imp)}'])

save('resonance_calcs', data)