import numpy as np
from calcs.metric_notation import from_metric, to_metric

class Inductance:
    @staticmethod
    def reatance(inductacne, frequency):
        l = from_metric(inductacne)
        f = from_metric(frequency)
        
        return to_metric(2 * np.pi * f * l)
    
    class From:
        @staticmethod
        def fxl(frequency, reactance):
            f = from_metric(frequency)
            x = from_metric(reactance)
            return to_metric(x / (2 * np.pi * f))
    
        @staticmethod
        def series_zrf(impedance, resistance, frequency):
            z = from_metric(impedance)**2
            r = from_metric(resistance)**2
            f = from_metric(frequency)
            w = 2 * np.pi * f
            l = np.sqrt(np.abs(z - r)) / w
            
            return to_metric(l)