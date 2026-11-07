"""Unitary grid QD and a variational mean-field wavepacket/RP baseline.

The hybrid is an explicitly approximate Hamiltonian, NOT an isomorphic-
Hamiltonian, surface-hopping, or exact reduced-density-matrix algorithm:
 H = <T_q + P^-1 sum_b V(q,Q_b)> + P^-1 sum_b [M V_b^2/2 + spring_b].
Bath canonical momenta have masses M/P. Consequently the bath acceleration
is -<dV/dQ_b>/M, NOT -<dV/dQ_b>/(P*M). This scaling conserves the stated H.

Potential kicks leave quantum coordinate densities invariant. Quantum kinetic
and free-ring propagation form the middle step of a symmetric splitting.
No absorbing potential or renormalization is used: edge occupancy is reported.
"""
from dataclasses import dataclass, asdict
from itertools import product
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.special import eval_hermite, factorial
from .models import Model
from .rpmd import harmonic_ring, free_ring_step


@dataclass(frozen=True)
class PacketConfig:
    energy: float = 1.0
    x0: float = -18.0
    sigma: float = 3.0  # standard deviation of |psi|^2 along x
    x_extent: float = 80.0
    nx: int = 512
    transverse_extent: float = 6.0
    nt: int = 20
    dt: float = 0.04
    time: float = 40.0
    beta: float = 8.0
    beads: int = 16
    samples: int = 24
    seed: int = 2026
    thermal_levels: int = 2

    def __post_init__(self):
        if min(self.energy, self.sigma, self.x_extent, self.transverse_extent,
               self.dt, self.time, self.beta) <= 0:
            raise ValueError("Positive packet scales required")
        if min(self.nx, self.nt, self.beads, self.samples, self.thermal_levels) < 1:
            raise ValueError("Positive grid and ensemble sizes required")
        if abs(self.x0)+4*self.sigma >= self.x_extent:
            raise ValueError("Initial wavepacket is too close to the periodic boundary")


def _grids(config, active):
    axes = [np.linspace(-config.x_extent, config.x_extent, config.nx, endpoint=False)]
    axes += [np.linspace(-config.transverse_extent, config.transverse_extent,
                         config.nt, endpoint=False) for _ in active[1:]]
    mesh = np.meshgrid(*axes, indexing="ij", sparse=True)
    return axes, mesh


def propagate(model: Model, config: PacketConfig, active=(0,1,2)):
    """active=(0,1,2): thermal-mixture 3D QD; smaller active: hybrid baseline.

    QD sums retained thermal states deterministically. Hybrid samples thermal
    quantum states and asymptotic harmonic bath rings independently. Both have
    the same incident Gaussian, beta, PES, masses and asymptotic factorization.
    """
    active = model.validate_active(active)
    bath = tuple(j for j in (1,2) if j not in active)
    rng = np.random.default_rng(config.seed)
    axes, mesh = _grids(config, active)
    shape = tuple(len(a) for a in axes)
    ndim = len(active)
    sum_axes = tuple(range(1, ndim+1))
    if not bath:
        states = list(product(range(config.thermal_levels), repeat=2))
        weights = np.array([np.exp(-config.beta*np.dot(model.omega, n)) for n in states])
        retained = float(np.prod(1-np.exp(-config.beta*model.omega*config.thermal_levels)))
        weights /= weights.sum()
        quantum_n = np.array(states)
    else:
        weights = np.full(config.samples, 1/config.samples)
        retained = 1.0
        quantum_n = np.zeros((config.samples, 2), dtype=int)
        for j in active[1:]:
            quantum_n[:, j-1] = rng.geometric(1-np.exp(-config.beta*model.omega[j-1]), config.samples)-1
    count = len(weights)
    psi = np.ones((count,)+shape, complex)
    px = np.sqrt(2*model.masses[0]*config.energy)
    psi *= np.exp(-(mesh[0]-config.x0)**2/(4*config.sigma**2)+1j*px*mesh[0])[None, ...]
    for a, j in enumerate(active[1:], start=1):
        r = np.sqrt(model.masses[j]*model.omega[j-1])*mesh[a]
        for s in range(count):
            n = quantum_n[s, j-1]
            psi[s] *= eval_hermite(n, r)*np.exp(-r*r/2)/np.sqrt(2.**n*factorial(n))
    # Discrete L2 normalization; uniform-cell factors are absorbed into psi.
    psi /= np.sqrt(np.sum(abs(psi)**2, axis=sum_axes)).reshape((-1,)+(1,)*ndim)
    q = np.empty((count, config.beads, len(bath)))
    v = np.empty_like(q)
    for a, j in enumerate(bath):
        q[..., a] = harmonic_ring(rng, count, config.beads, config.beta,
                                  model.masses[j], model.omega[j-1])
        v[..., a] = rng.normal(size=(count,config.beads))*np.sqrt(config.beads/(config.beta*model.masses[j]))
    u, _, k, _, c, _ = model.fields(mesh[0])
    system_potential = np.broadcast_to(u, shape).copy()
    for a, j in enumerate(active[1:], start=1):
        system_potential += 0.5*k[..., j-1]*(mesh[a]-c[..., j-1])**2
    k = [k[..., j-1].reshape(len(axes[0])) for j in bath]
    c = [c[..., j-1].reshape(len(axes[0])) for j in bath]
    kinetic = np.zeros(shape)
    for a, j in enumerate(active):
        p = 2*np.pi*fftfreq(len(axes[a]), axes[a][1]-axes[a][0])
        ps = [1]*ndim
        ps[a] = len(p)
        kinetic += p.reshape(ps)**2/(2*model.masses[j])
    steps = int(np.ceil(config.time/config.dt))
    dt = config.time/steps
    kinetic_phase = np.exp(-1j*dt*kinetic)
    broadcast = (count, len(axes[0]))+(1,)*(ndim-1)
    bath_mass = np.asarray(model.masses)[list(bath)]

    def bath_potential(qb):
        # Every bath term depends on the quantum x coordinate only. Keep the
        # bath phase one-dimensional instead of exponentiating a full batch
        # of 2D potentials at every half-step.
        out = np.zeros((count, len(axes[0])))
        for a in range(len(bath)):
            mean = qb[..., a].mean(axis=1)[:, None]
            square = (qb[..., a]**2).mean(axis=1)[:, None]
            out += 0.5*k[a]*(square-2*c[a]*mean+c[a]**2)
        return out

    def effective_potential(qb):
        return system_potential + bath_potential(qb).reshape(broadcast)

    def force(density, qb):
        out = np.empty_like(qb)
        marginal = density.sum(axis=tuple(range(2, ndim+1))) if ndim > 1 else density
        for a in range(len(bath)):
            expected_k = marginal@k[a]
            expected_kc = marginal@(k[a]*c[a])
            out[..., a] = -expected_k[:, None]*qb[..., a]+expected_kc[:, None]
        return out

    def energy():
        pk = fftn(psi, axes=sum_axes, norm="ortho")
        en = np.sum(abs(pk)**2*kinetic+abs(psi)**2*effective_potential(q), axis=sum_axes)
        if bath:
            spring = 0.5*(config.beads/config.beta)**2*np.sum(bath_mass*(q-np.roll(q, 1, axis=1))**2, axis=-1)
            en += np.mean(spring+0.5*np.sum(bath_mass*v*v, axis=-1), axis=1)
        return en

    initial_energy = energy()
    static_phase = np.exp(-0.5j*dt*system_potential)
    saves = set(np.unique(np.linspace(0, steps, 41).astype(int)))
    rows, final_per_sample = [], None
    max_edge = 0.0
    max_transverse_edge = 0.0
    surface = 6.0 if model.kind != "zpe" else 10.0
    for step in range(steps+1):
        if step in saves:
            density = abs(psi)**2
            marginal = density.sum(axis=tuple(range(2,ndim+1))) if ndim > 1 else density
            left = marginal[:, axes[0] < -surface].sum(axis=1)
            right = marginal[:, axes[0] > surface].sum(axis=1)
            edge = marginal[:, abs(axes[0]) > 0.9*config.x_extent].sum(axis=1)
            max_edge = max(max_edge, float(weights@edge))
            for a in range(1, ndim):
                mask = (abs(mesh[a]) > 0.85*config.transverse_extent)
                max_transverse_edge = max(max_transverse_edge, float(weights@np.sum(density*mask,axis=sum_axes)))
            norm = np.sum(density, axis=sum_axes)
            final_per_sample = right
            rows.append({"time": step*dt, "left": float(weights@left),
                         "right": float(weights@right), "interaction": float(weights@(norm-left-right)),
                         "norm": float(weights@norm)})
        if step == steps:
            break
        if bath:
            v += 0.5*dt*force(abs(psi)**2, q)/bath_mass
            psi *= np.exp(-0.5j*dt*bath_potential(q)).reshape(broadcast)
        psi *= static_phase
        psi = ifftn(fftn(psi, axes=sum_axes, norm="ortho")*kinetic_phase,
                    axes=sum_axes, norm="ortho")
        if bath:
            q, v = free_ring_step(q, v, dt, config.beta)
        if bath:
            psi *= np.exp(-0.5j*dt*bath_potential(q)).reshape(broadcast)
        psi *= static_phase
        if bath:
            v += 0.5*dt*force(abs(psi)**2, q)/bath_mass
    drift = float(np.max(abs(energy()-initial_energy)/(1+abs(initial_energy))))
    warnings = []
    if max_edge > 1e-5 or max_transverse_edge > 1e-5:
        warnings.append("periodic_boundary_occupancy")
    if rows[-1]["interaction"] > 0.01:
        warnings.append("not_an_asymptotic_transmission")
    if drift > 0.005:
        warnings.append("large_energy_drift")
    return {"method": "grid_QD" if not bath else "mean_field_Q_RP",
            "active": list(active), "bath": list(bath), "config": asdict(config),
            "product_probability": rows[-1]["right"],
            "stderr": float(np.std(final_per_sample, ddof=1)/np.sqrt(count)) if bath and count > 1 else 0.0,
            "initial_thermal_omitted_weight": 1-retained,
            "max_relative_energy_drift": drift, "max_x_edge_probability": max_edge,
            "max_transverse_edge_probability": max_transverse_edge,
            "trace": rows, "warnings": warnings,
            "claim": "finite-time wavepacket probability; not a thermal rate or monochromatic P(E)"}
