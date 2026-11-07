"""Targeted independent numerical checks; not a full convergence certificate.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/refine.py
"""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from pyrpqd import ScatteringSolver
from pyrpqd.benchmark import model_for, metadata, save_json
from pyrpqd.rpmd import thermal_rate
from pyrpqd.wavepacket import PacketConfig, propagate

out = Path("results/refinement")
out.mkdir(parents=True,exist_ok=True)
report = {"metadata":metadata(),"stationary":{},"resonance_peaks":[],"rpmd":{},"packets":{}}
for kind,energies in {"tunneling":[.5,1.25,2.,3.5,4.5],"zpe":[.8,.99,1.01,1.1,1.5,2.]}.items():
    records = []
    for basis,dx,extent in ([(6,.15,10.),(8,.075,12.)] if kind == "tunneling" else [(8,.15,18.),(10,.075,24.)]):
        solver = ScatteringSolver(model_for(kind),(basis,basis),dx,extent)
        spectrum = solver.spectrum(energies)
        records.append({"basis":basis,"dx":solver.dx,"extent":extent,
                        "spectrum":{k:v.tolist() for k,v in spectrum.items()},
                        "threshold":float(solver.right_e[0]-solver.model.ground_energy)})
    report["stationary"][kind] = records
    save_json(out/"checks.json",report)
    print("stationary checks",kind,flush=True)
for basis,dx in [(6,.15),(6,.075),(6,.0375),(8,.075),(6,.01875),(6,.009375)]:
    solver = ScatteringSolver(model_for("resonance"),(basis,basis),dx,12.)
    peaks = []
    brackets = [(.479,.482),(1.643,1.646)] if dx < .03 else [(0.45,.51),(1.60,1.69)]
    for bracket in brackets:
        result = minimize_scalar(lambda e:-solver.scatter(e)["T0"],bounds=bracket,
                                 method="bounded",options={"xatol":1e-10})
        peaks.append({"energy":float(result.x),"height":float(-result.fun)})
    report["resonance_peaks"].append({"basis":basis,"dx":solver.dx,"peaks":peaks})
    save_json(out/"checks.json",report)
    print("resonance peaks",basis,dx,peaks,flush=True)
solver = ScatteringSolver(model_for("tunneling"),(6,6),.075,12.)
spectrum = solver.spectrum(np.arange(.01,9.,.02))
report["tunneling_refined_thermal_qd"] = {"basis":[6,6],"dx":solver.dx,"extent":12.,
    "energy_step":.02,"rates":{str(beta):solver.thermal_rate(spectrum,beta) for beta in [2.,4.]},
    "max_unitarity_error":float(spectrum["unitarity_error"].max())}
save_json(out/"checks.json",report)
print("refined thermal QD",report["tunneling_refined_thermal_qd"],flush=True)
report["rpmd"]["tunneling_beta2"] = thermal_rate(model_for("tunneling"),beta=2.,beads=32,
    samples=1024,dt=.01,time=35.,seed=9917)
save_json(out/"checks.json",report)
print("refined RPMD",report["rpmd"]["tunneling_beta2"]["rate"],flush=True)
cfg = PacketConfig(energy=2.,x0=-24.,sigma=4.,x_extent=112.,nx=1536,nt=32,
    dt=.015,time=35.,beta=12.,beads=48,samples=32,thermal_levels=1,seed=20261107)
report["packets"]["tunneling_QD3D"] = propagate(model_for("tunneling"),cfg,(0,1,2))
save_json(out/"checks.json",report)
print("refined packet",report["packets"]["tunneling_QD3D"]["product_probability"],flush=True)
