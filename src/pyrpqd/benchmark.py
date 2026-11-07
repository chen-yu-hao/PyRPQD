"""Reproducible CLI. Results are screening calculations, not automatic proof of convergence."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.optimize import minimize_scalar, brentq
from scipy.signal import find_peaks
from .models import Model
from .qd import ScatteringSolver
from .rpmd import thermal_rate
from .correction import corrected_rate
from .wavepacket import PacketConfig, propagate


def metadata():
    root = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "source_sha256": hashes, "units": "hbar=k_B=1; masses vary at fixed force constants",
            "status": "screening; inspect refinement and diagnostics before scientific conclusions"}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def spectrum_with_peaks(solver, step=.04, emax=9., refine=False):
    """Resolve detected resonances beyond a uniform scan.

    Detection is not a completeness proof: a second denser initial scan is
    necessary to establish that narrow isolated resonances were not missed.
    """
    cache = {}
    def evaluate(e):
        e = float(e)
        if e not in cache:
            cache[e] = solver.scatter(e)
        return cache[e]
    grid = np.arange(step/2, emax, step)
    for e in grid:
        evaluate(e)
    records = []
    if refine:
        candidates = set()
        for key in ("N", "T0"):
            vals = np.array([cache[float(e)][key] for e in grid])
            for index in find_peaks(vals, prominence=1e-7)[0]:
                candidates.add((int(index),key))
        for peak_index,(index,key) in enumerate(sorted(candidates)):
            if peak_index % 10 == 0:
                print(f"  refining detected peak {peak_index+1}/{len(candidates)}",flush=True)
            lo, hi = float(grid[index-1]),float(grid[index+1])
            optimum = minimize_scalar(lambda e: -evaluate(e)[key],bounds=(lo,hi),method="bounded",
                                       options={"xatol":1e-10})
            center, height = float(optimum.x),float(-optimum.fun)
            baseline = min(evaluate(lo)[key],evaluate(hi)[key])
            half = (height+baseline)/2
            if height-baseline < 1e-6:
                continue
            widths = []
            for endpoint in (lo,hi):
                if evaluate(endpoint)[key] < half:
                    root = brentq(lambda e:evaluate(e)[key]-half,min(endpoint,center),max(endpoint,center),xtol=1e-11)
                    widths.append(abs(root-center))
                else:
                    widths.append(abs(endpoint-center))
            width = min(widths)
            for endpoint in (lo,hi):
                direction = np.sign(endpoint-center)
                offsets = width*np.sinh(np.linspace(0,np.arcsinh(abs(endpoint-center)/width),25))
                for delta in offsets:
                    evaluate(center+direction*delta)
            records.append({"observable":key,"energy":center,"height":height,
                            "local_half_prominence_width":sum(widths)})
    energies = np.array(sorted(cache))
    spectrum = {key:np.array([cache[e][key] for e in energies]) for key in
                ("energy","N","T0","R0","unitarity_error")}
    return spectrum, records


def model_for(kind):
    return Model(kind,coupling=(.25,.125) if kind == "resonance" else (.3,.15))


def run_thermal(output, kinds, quick=False, samples=None, betas=(2.,4.)):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data = {"metadata":metadata(),"models":{}}
    for model_index,kind in enumerate(kinds):
        model = model_for(kind)
        n = 4 if quick else (8 if kind == "zpe" else 6)
        dx = .25 if quick else .15
        extent = 18. if kind == "zpe" else 10.
        step = .08 if quick else .04
        layouts = {"1D":((1,1),(0,)),"2D":((n,1),(0,1)),"3D":((n,n),(0,1,2))}
        item = {"parameters":asdict(model),"qd":{},"rpmd":{},"corrections":{}}
        for label,(basis,active) in layouts.items():
            start = time.perf_counter()
            solver = ScatteringSolver(model,basis,dx,extent)
            spectrum,peaks = spectrum_with_peaks(solver,step=step,refine=kind=="resonance")
            filename = f"{kind}_{label}_spectrum.csv"
            with (output/filename).open("w",newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(spectrum.keys())
                writer.writerows(zip(*spectrum.values()))
            item["qd"][label] = {"basis":list(basis),"dx":float(solver.dx),"extent":extent,
                                  "energy_step_initial":step,"energy_max":9.,"points":len(spectrum["energy"]),
                                  "max_unitarity_error":float(spectrum["unitarity_error"].max()),
                                  "right_ground_threshold":float(solver.right_e[0]-model.ground_energy),
                                  "rates":{str(beta):solver.thermal_rate(spectrum,beta) for beta in betas},
                                  "peaks":peaks,"spectrum_file":filename,
                                  "seconds":time.perf_counter()-start}
            print(f"QD {kind} {label}: {item['qd'][label]['rates']}",flush=True)
            data["models"][kind] = item
            save_json(output/"thermal.json",data)
        for beta_index,beta in enumerate(betas):
            key = str(beta)
            item["rpmd"][key] = {}
            for dim_index,(label,(_,active)) in enumerate(layouts.items()):
                result = thermal_rate(model,beta=beta,beads=(8 if quick else int(8*beta)),
                                      samples=samples or (128 if quick else 2048),
                                      dt=.04 if quick else .02,
                                      time=60. if kind == "resonance" else 35.,
                                      seed=41000+1000*model_index+100*beta_index+dim_index,active=active)
                item["rpmd"][key][label] = result
                data["models"][kind] = item
                save_json(output/"thermal.json",data)
                print(f"RPMD {kind} beta={beta} {label}: {result['rate']:.7g} +/- {result['stderr']:.2g}; {result['warnings']}",flush=True)
            item["corrections"][key] = {}
            for label in ("1D","2D"):
                try:
                    item["corrections"][key][label] = corrected_rate(item["rpmd"][key]["3D"],
                                                                   item["qd"][label]["rates"][key],
                                                                   item["rpmd"][key][label])
                except ValueError as exc:
                    item["corrections"][key][label] = {"status":"undefined","reason":str(exc)}
        data["models"][kind] = item
        save_json(output/"thermal.json",data)
    return data


def run_packets(output,kinds,quick=False,samples=None,resume=False):
    output = Path(output)
    checkpoint = output/"packets.json"
    data = json.loads(checkpoint.read_text()) if resume and checkpoint.exists() else {"metadata":metadata(),"models":{}}
    if resume:
        data.setdefault("resume_history",[]).append(metadata())
    for kind in kinds:
        cfg = PacketConfig(energy={"resonance":1.64,"tunneling":2.,"zpe":.65}[kind],
                           x0=-24.,sigma=4.,x_extent=200. if kind == "resonance" else 112.,
                           nx=(896 if quick else 1792) if kind == "resonance" else (512 if quick else 1024),
                           nt=16 if quick else 24,dt=.06 if quick else .03,
                           time={"resonance":75.,"tunneling":35.,"zpe":60.}[kind],beta=12.,
                           beads=16 if quick else 48,samples=samples or (8 if quick else 32),
                           thermal_levels=1,seed=20261107)
        item = {"parameters":asdict(model_for(kind)),"methods":{},
                "initial_state_note":"QD retains ground bath state; omitted thermal weight is reported and bounds thermal-truncation error for bounded observables"}
        if resume and kind in data["models"]:
            item = data["models"][kind]
        for label,active in {"QD3D":(0,1,2),"B1+2":(0,),"B2+1_xy":(0,1),"B2+1_xz":(0,2)}.items():
            if label in item["methods"]:
                if item["methods"][label]["config"] != asdict(cfg):
                    raise ValueError("Resume settings do not match checkpoint")
                print(f"reuse packet {kind} {label}",flush=True)
                continue
            start = time.perf_counter()
            result = propagate(model_for(kind),cfg,active)
            result["seconds"] = time.perf_counter()-start
            result["source_sha256"] = metadata()["source_sha256"]
            item["methods"][label] = result
            print(f"packet {kind} {label}: {result['product_probability']:.7g} +/- {result['stderr']:.2g}; {result['warnings']}",flush=True)
            data["models"][kind] = item
            save_json(output/"packets.json",data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["thermal","packet","all"],default="thermal")
    parser.add_argument("--models",nargs="+",choices=["resonance","tunneling","zpe"],default=["resonance","tunneling","zpe"])
    parser.add_argument("--output",default="results/local")
    parser.add_argument("--quick",action="store_true",help="Smoke settings; not converged scientific results")
    parser.add_argument("--samples",type=int)
    parser.add_argument("--resume",action="store_true",help="Resume matching packet checkpoints")
    parser.add_argument("--betas",type=float,nargs="+",default=[2.,4.])
    args = parser.parse_args()
    if args.resume and args.mode != "packet":
        parser.error("--resume currently applies only to --mode packet")
    if args.mode in {"thermal","all"}:
        run_thermal(args.output,args.models,args.quick,args.samples,tuple(args.betas))
    if args.mode in {"packet","all"}:
        run_packets(args.output,args.models,args.quick,args.samples,args.resume)


if __name__ == "__main__":
    main()
