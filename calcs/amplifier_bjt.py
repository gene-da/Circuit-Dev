import numpy as np
from calcs.metric_notation import from_metric, to_metric

class CommonEmitter:
    def __init__(self, vcc: str, dcBeta: str, r1: str, r2: str, rc: str, re: str):
        self.r1 = from_metric(r1)
        self.r2 = from_metric(r2)
        self.rc = from_metric(rc)
        self.re = from_metric(re)
        self.vcc = from_metric(vcc)
        self.vbe = 0.7
        self.dcBeta = from_metric(dcBeta)

        # Initialize computed values
        self.ce = None
        self.rth = None
        self.vth = None
        self.ie = None
        self.ic = None
        self.ve = None
        self.vb = None
        self.vc = None
        self.vce = None
        self.av = None
        self.tre = None

    def calc_rth(self):
        self.rth = (self.r1 * self.r2) / (self.r1 + self.r2)
        return to_metric(self.rth)

    def calc_vth(self):
        self.vth = (self.r2 / (self.r1 + self.r2)) * self.vcc
        return to_metric(self.vth)

    def calc_ie(self):
        if self.rth is None:
            self.calc_rth()
        if self.vth is None:
            self.calc_vth()
        self.ie = (self.vth - self.vbe) / (self.re + (self.rth / self.dcBeta))
        return to_metric(self.ie)

    def calc_ic(self):
        if self.ie is None:
            self.calc_ie()
        self.ic = self.ie  # assuming IE ≈ IC
        return to_metric(self.ic)

    def calc_ve(self):
        if self.ie is None:
            self.calc_ie()
        self.ve = self.ie * self.re
        return to_metric(self.ve)

    def calc_vb(self):
        if self.ve is None:
            self.calc_ve()
        self.vb = self.ve + self.vbe
        return to_metric(self.vb)

    def calc_vc(self):
        if self.ic is None:
            self.calc_ic()
        self.vc = self.vcc - (self.ic * self.rc)
        return to_metric(self.vc)

    def calc_vce(self):
        if self.vc is None:
            self.calc_vc()
        if self.ve is None:
            self.calc_ve()
        self.vce = self.vc - self.ve
        return to_metric(self.vce)
    
    def av(self, loadR: str=None, baseV: str=None):
        pass
            
        
    def ovg(self, sourceV: str):
        if self.vc is None:
            self.calc_vc
            
        vs = from_metric(sourceV)
        return to_metric(self.vc / vs)
    
    def calc_ce(self, frequency):
        f = from_metric(frequency)
        xce = self.re/10
        self.ce = 1/(2*np.pi*f*xce)
        return to_metric(self.ce)
    
    def calc_tre(self, baseV: str=None):
        if baseV is None:
            if self.vb is None:
                self.calc_vb
            if self.ie is None:
                self.calc_ie
            
            self.tre = self.vb/self.ie
            return to_metric(self.tre)
        else:
            vb = from_metric(baseV)
            tre = (vb + self.vb)/self.ie
            return to_metric(tre)
    
    def calc_complete_rc(self, out_c: str, loadR: str):
        c = from_metric(out_c)
        rl = from_metric(loadR)
        
        rc = (self.rc*rl)/(self.rc+rl)
        return to_metric(rc)

    def summary(self):
        return {
            "Rth": to_metric(self.rth) if self.rth else self.calc_rth(),
            "Vth": to_metric(self.vth) if self.vth else self.calc_vth(),
            "IE": to_metric(self.ie) if self.ie else self.calc_ie(),
            "IC": to_metric(self.ic) if self.ic else self.calc_ic(),
            "VE": to_metric(self.ve) if self.ve else self.calc_ve(),
            "VB": to_metric(self.vb) if self.vb else self.calc_vb(),
            "VC": to_metric(self.vc) if self.vc else self.calc_vc(),
            "VCE": to_metric(self.vce) if self.vce else self.calc_vce(),
        }

    
        
    