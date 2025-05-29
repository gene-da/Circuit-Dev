from modules.oscope import Oscope
import numpy as np

am_modulator = r'circuit_sims/AM Circuits/Base Injected/Test AM curcuit/BaseInjectedAM.raw'

scope = Oscope(am_modulator)

scope.set_channel(1, 'V(n003)')
scope.set_channel(2, "V(af)")
scope.set_math(True, False, 'mul')
plots = [('ch1', 'dc'), ('ch2', 'ac')]
scope.plot(plots, '8m', '500m', '1m')

scope.set_fft("ch2")
freq, db = scope.get_channel_freq("ch2")
print(f"Dominant freq: {freq:.2f} Hz @ {db:.2f} dB")


carrier_freq, _ = scope.detect_fft_peak()
scope.plot_fft(center=f"{carrier_freq:.0f}", freq_per_div="500", mag_per_div="50", use_log_freq=True)

