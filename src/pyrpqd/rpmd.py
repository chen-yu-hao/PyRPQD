"""Thermal RPMD flux-side rates using independent Gaussian importance samples.

Sampling reference: free x ring at a constrained centroid, and asymptotic
harmonic transverse rings. Importance weights give the *absolute* constrained
density ratio; there is no resampling or fitted free-energy normalization.
Paired forward/reversed flux-biased momenta estimate recrossing. Dynamics are
microcanonical, with exact free-ring normal-mode propagation between kicks.
"""
import numpy as np
from scipy.fft import fft, ifft
from .models import Model


def spring_frequencies(beads, beta):
    if int(beads) != beads or beads < 1 or beta <= 0:
        raise ValueError("Positive integer beads and positive beta required")
    return 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)


def harmonic_ring(rng, samples, beads, beta, mass, omega, centroid=None):
    w = spring_frequencies(beads, beta)
    denom = w*w+omega*omega
    sd = np.sqrt(beads/(beta*mass*np.where(denom > 0, denom, 1)))
    if centroid is not None:
        sd[0] = 0
    elif omega == 0:
        raise ValueError("A free ring needs a fixed centroid")
    white = rng.normal(size=(samples, beads))
    q = ifft(fft(white, axis=1, norm="ortho")*sd, axis=1, norm="ortho").real
    return q if centroid is None else q+centroid


def free_ring_step(q, v, dt, beta):
    w = spring_frequencies(q.shape[1], beta)
    shape = (1, -1)+(1,)*(q.ndim-2)
    co = np.cos(w*dt).reshape(shape)
    si_over_w = (dt*np.sinc(w*dt/np.pi)).reshape(shape)
    w2 = (w*w).reshape(shape)
    qk, vk = fft(q, axis=1, norm="ortho"), fft(v, axis=1, norm="ortho")
    nq = co*qk+si_over_w*vk
    nv = co*vk-w2*si_over_w*qk
    return (ifft(nq, axis=1, norm="ortho").real,
            ifft(nv, axis=1, norm="ortho").real)


def ring_energy(model, q, v, beta, active=(0,1,2)):
    m = np.asarray(model.masses)[list(active)]
    omega_p = q.shape[1]/beta
    value = model.potential_gradient(q, active)[0]
    kin = 0.5*np.sum(m*v*v, axis=-1)
    spring = 0.5*omega_p**2*np.sum(m*(q-np.roll(q, 1, axis=1))**2, axis=-1)
    return np.mean(kin+spring+value, axis=1)


def thermal_rate(model: Model, beta=2.0, beads=16, samples=1024, dt=0.02,
                 time=30.0, seed=1234, active=(0,1,2), surface=None, checkpoints=31):
    active = model.validate_active(active)
    if beta <= 0 or beads < 1 or samples < 2 or dt <= 0 or time <= 0 or checkpoints < 3:
        raise ValueError("Invalid sampling or propagation settings")
    rng = np.random.default_rng(seed)
    surface = model.dividing_surface if surface is None else float(surface)
    mass = np.asarray(model.masses)[list(active)]
    q = np.empty((samples, beads, len(active)))
    for a, j in enumerate(active):
        q[..., a] = harmonic_ring(rng, samples, beads, beta, mass[a],
                                  0 if j == 0 else model.omega[j-1],
                                  surface if j == 0 else None)
    delta_v = model.potential_gradient(q, active)[0]-model.reference_potential(q, active)
    logw = -beta*np.mean(delta_v, axis=1)
    shift = float(logw.max())
    w = np.exp(logw-shift)
    scale = np.exp(shift)/np.sqrt(2*np.pi*beta*mass[0])
    # Positive flux distribution: centroid speed is Rayleigh, not half-normal.
    v = rng.normal(size=q.shape)*np.sqrt(beads/(beta*mass))
    v[..., 0] -= v[..., 0].mean(axis=1, keepdims=True)
    v[..., 0] += rng.rayleigh(1/np.sqrt(beta*mass[0]), samples)[:, None]
    q = np.concatenate((q, q.copy()), axis=0)
    v = np.concatenate((v, -v), axis=0)
    steps = int(np.ceil(time/dt))
    dt = time/steps
    save = set(np.unique(np.linspace(1, steps, checkpoints).astype(int)))
    e0 = ring_energy(model, q, v, beta, active)
    times, kappas, rates, values = [], [], [], []
    grad = model.potential_gradient(q, active)[1]
    for step in range(1, steps+1):
        v -= 0.5*dt*grad/mass
        q, v = free_ring_step(q, v, dt, beta)
        grad = model.potential_gradient(q, active)[1]
        v -= 0.5*dt*grad/mass
        if step in save:
            side = (q[..., 0].mean(axis=1) > surface).astype(float)
            difference = side[:samples]-side[samples:]
            obs = w*difference
            times.append(step*dt)
            kappas.append(float(obs.sum()/w.sum()))
            rates.append(float(scale*obs.mean()))
            values.append(obs)
    # A plateau-window average, with uncertainty over independent paths, not times.
    tail = np.array(values[-max(3, len(values)//5):]).mean(axis=0)
    rate = float(scale*tail.mean())
    stderr = float(scale*tail.std(ddof=1)/np.sqrt(samples))
    final_energy = ring_energy(model, q, v, beta, active)
    drift = float(np.max(abs(final_energy-e0)/(1+abs(e0))))
    tail_rates = np.asarray(rates[-max(3, len(values)//5):])
    span = float(np.ptp(tail_rates))
    ess = float(w.sum()**2/np.sum(w*w))
    warnings = []
    if ess < 0.1*samples:
        warnings.append("low_importance_ESS")
    if rate <= 0 or stderr > 0.1*abs(rate):
        warnings.append("large_statistical_uncertainty")
    if span > max(3*stderr, 0.05*abs(rate)):
        warnings.append("flux_side_plateau_not_resolved")
    if drift > 0.01:
        warnings.append("large_energy_drift")
    return {"rate": rate, "stderr": stderr, "beta": float(beta), "beads": beads,
            "samples": samples, "seed": seed, "active": list(active), "dt": dt,
            "time": time, "surface": surface, "importance_ess": ess,
            "density_ratio": float(np.exp(shift)*w.mean()),
            "max_relative_energy_drift": drift, "plateau_span": span,
            "times": times, "kappa": kappas, "rate_trace": rates,
            "warnings": warnings,
            "status": "diagnostic_pass" if not warnings else "needs_convergence"}
