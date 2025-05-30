from dataclasses import dataclass

@dataclass
class Antenna:
    current: float = None
    voltage: float = None
    resistance: float = None
    inductance: float = None
    capacitance: float = None
    impedance: complex = None

    frequency: float = None
    wavelength: float = None
    length: float = None
    gain: float = None
    efficiency: float = None

    power_input: float = None
    power_radiated: float = None
    swr: float = None
    bandwidth: float = None

    name: str = None
    type: str = None
        