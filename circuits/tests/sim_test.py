import os
import tempfile
from spicelib.sim.sim_runner import SimRunner
from spicelib.editor.spice_editor import SpiceEditor
from spicelib.simulators.ngspice_simulator import NGspiceSimulator
from spicelib.utils.sweep_iterators import sweep_log

# Paths
project_root = os.path.dirname(os.path.abspath(__file__))
ngspice_path = os.path.join(project_root, 'ngspice.exe')
output_path = os.path.join(project_root, 'temp')

# Check NGSpice presence
if not os.path.isfile(ngspice_path):
    raise FileNotFoundError(f"NGSpice not found at: {ngspice_path}")

# --- SAFE BASELINE NETLIST (DC source only) ---
netlist_str = """* RC Low-Pass Filter Test
V1 in 0 DC 5
R1 in out 1k
C1 out 0 1u
.tran 1n 3m
.save all
.end
"""

# --- Write to temp file ---
with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.net') as tmp_netlist:
    tmp_netlist.write(netlist_str)
    tmp_netlist_path = tmp_netlist.name

# Load into SpiceEditor
netlist = SpiceEditor(tmp_netlist_path)

# Optional: dump the final netlist to console
print("\n[DEBUG] Final Netlist Sent to NGSpice:\n")
final_netlist = ''.join(netlist.netlist)
print(final_netlist)

# Also write the netlist to file for manual testing
with open("debug_test.net", "w") as f:
    f.write(final_netlist)
print("\n[INFO] Wrote netlist to debug_test.net — test it manually with:")
print(f"       {ngspice_path} debug_test.net")

# --- Setup simulation ---
runner = SimRunner(output_folder=output_path, simulator=NGspiceSimulator.create_from(ngspice_path))

def processing_data(raw_file, log_file):
    print(f"✔️ Simulation completed → RAW: {raw_file}, LOG: {log_file}")

# Sweep capacitor from 1pF to 10uF
for cap in sweep_log(1e-12, 10e-6, 8):  # 8 steps to match your last run
    netlist['C1'].value = cap
    print(f"\n[INFO] Running simulation with C1 = {cap}")
    runner.run(netlist, callback=processing_data)

# Wait for completion
runner.wait_completion()

# Final result
print(f"\n✅ Successful simulations: {runner.okSim}/{runner.runno}")
