import sys
import os
import numpy as np
from tools import metric_notation as mn

def xL(ind, freq):
    l = mn.fm(ind)
    f = mn.fm(freq)
    
    return 2*np.pi*f*l

def res_c(ind, freq):
    l = mn.fm(ind)
    f = mn.fm(freq)
    return 1/(4*(np.pi**2)*l*(f**2))

def ind_react(react, freq):
    x = mn.fm(react)
    f = mn.fm(freq)
    
    return x/(2*np.pi*f)

def pind(inductance, name='0'):
    print(f'L{name}: {mn.tm(inductance)}H')
    
def pcap(capacitance, name='0'):
    print(f'C{name}: {mn.tm(capacitance)}F')

lower_freq = mn.fm('530k')
upper_freq = mn.fm('1700k')

class Inductance:
    @staticmethod
    def mean_reactance(inductance, lower_freq, upper_freq, step):
        l = mn.fm(inductance)
        frequency_range = range(int(mn.fm(lower_freq)), int(mn.fm(upper_freq)), int(mn.fm(step)))
        
        reactance = []
        
        for frequency in frequency_range:
            xl = xL(l, frequency)
            reactance.append(xl)
            
        mean = np.mean(reactance)
        
        print(f'Inductance: {inductance}H | Range {lower_freq}Hz - {upper_freq}Hz')
        print(f'Mean XL: {mn.tm(mean)}Ω | Lower: {mn.tm(xL(l, mn.fm(lower_freq)))}Ω | Upper: {mn.tm(xL(l, mn.fm(upper_freq)))}Ω')
        
        return mean

l = '400u'
lower_freq = '530k'
upper_freq = '1700k'
print()
Inductance.mean_reactance(l, lower_freq, upper_freq, '10k')

def mean_capicitance(inductance, lower_freq, upper_freq, step):
    l = mn.fm(inductance)
    frequency_range = range(int(mn.fm(lower_freq)), int(mn.fm(upper_freq)), int(mn.fm(step)))
    
    reactance = []
    
    for frequency in frequency_range:
        c = res_c(l, frequency)
        reactance.append(c)
        
    mean = np.mean(reactance)
    
    print(f'Inductance: {inductance}H | Range {lower_freq}Hz - {upper_freq}Hz')
    print(f'Mean Cap: {mn.tm(mean)}F | Lower: {mn.tm(res_c(l, mn.fm(lower_freq)))}F | Upper: {mn.tm(res_c(l, mn.fm(upper_freq)))}F')
    
    return mean

print()
mean_capicitance(l, lower_freq, upper_freq, '10k')
print()

c_100k = res_c(l, '100k')

print(f'Cap: {mn.tm(c_100k)}')
print