"""Direction A: matched-Hamiltonian log-rate correction, never fit to 3D QD."""
import numpy as np


def corrected_rate(full_rpmd, reduced_qd, reduced_rpmd):
    values = (full_rpmd["rate"], reduced_qd, reduced_rpmd["rate"])
    if not np.all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError("A log correction requires finite, positive rates")
    log_factor = float(np.log(reduced_qd)-np.log(reduced_rpmd["rate"]))
    rate = float(np.exp(np.log(full_rpmd["rate"])+log_factor))
    relative_variance = ((full_rpmd["stderr"]/values[0])**2
                         +(reduced_rpmd["stderr"]/values[2])**2)
    return {"rate": rate, "stderr_delta_method": float(rate*np.sqrt(relative_variance)),
            "log_factor": log_factor, "factor": float(np.exp(log_factor)),
            "uncertainty_assumption": "independent full/reduced RPMD samples; QD discretization error excluded",
            "warnings": sorted(set(full_rpmd.get("warnings", [])+reduced_rpmd.get("warnings", []))),
            "uses_3d_qd_for_fitting": False}
