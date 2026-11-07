"""Stationary coupled-channel QD with exact lattice lead self-energies.

Second-order finite differences along x; oscillator Galerkin bases along y,z.
The scattering problem is solved without dynamical approximations on this
finite discretization. Continuum/basis/lead-length convergence is separate.
"""
import numpy as np
from scipy.linalg import eigh, solve
from scipy.integrate import trapezoid
from .models import Model, oscillator_operators


def lead_self_energy(energy, eigenvalues, vectors, hopping):
    a = 1-(energy-eigenvalues)/(2*hopping)
    sigma = np.empty(a.shape, complex)
    opened = np.abs(a) < 1
    sigma[opened] = -hopping*(a[opened]+1j*np.sqrt(1-a[opened]**2))
    closed = ~opened
    ac = a[closed]
    sigma[closed] = -hopping/(ac+np.sign(ac)*np.sqrt(np.maximum(ac*ac-1, 0)))
    gamma = np.maximum(-2*sigma.imag, 0)
    return (vectors*sigma)@vectors.T, gamma


class ScatteringSolver:
    def __init__(self, model: Model, basis=(6, 6), dx=0.15, extent=12.0):
        if dx <= 0 or extent <= dx or len(basis) != 2 or min(basis) < 1:
            raise ValueError("Invalid scattering discretization")
        self.model, self.basis = model, tuple(int(n) for n in basis)
        self.x = np.linspace(-extent, extent, int(np.ceil(2*extent/dx))+1)
        self.dx = self.x[1]-self.x[0]
        self.hopping = 1/(2*model.masses[0]*self.dx**2)
        ny, nz = self.basis
        ey, y, y2 = oscillator_operators(ny, model.masses[1], model.omega[0])
        ez, z, z2 = oscillator_operators(nz, model.masses[2], model.omega[1])
        self.eye = np.eye(ny*nz)
        qops = (np.kron(y, np.eye(nz)), np.kron(np.eye(ny), z))
        q2ops = (np.kron(y2, np.eye(nz)), np.kron(np.eye(ny), z2))
        base = np.diag((ey[:, None]+ez).ravel())
        u, _, k, _, c, _ = model.fields(self.x)
        self.blocks = base[None, :, :]+u[:, None, None]*self.eye
        for j in range(2):
            self.blocks += 0.5*(k[:, j]-model.frequencies[j]**2)[:, None, None]*q2ops[j]
            self.blocks -= (k[:, j]*c[:, j])[:, None, None]*qops[j]
            self.blocks += (0.5*k[:, j]*c[:, j]**2)[:, None, None]*self.eye
        self.left_e, self.left_u = eigh(self.blocks[0])
        self.right_e, self.right_u = eigh(self.blocks[-1])

    def scatter(self, collision_energy):
        if not np.isfinite(collision_energy) or collision_energy <= 0:
            raise ValueError("Collision energy must be finite and positive")
        energy = collision_energy+self.model.ground_energy
        t, eye = self.hopping, self.eye
        sl, gl = lead_self_energy(energy, self.left_e, self.left_u, t)
        sr, gr = lead_self_energy(energy, self.right_e, self.right_u, t)
        # Block Thomas elimination of (E-H-Sigma) X = source on the left.
        invs, rhs = [], []
        previous = np.zeros_like(eye, complex)
        b = eye.astype(complex)
        for j, block in enumerate(self.blocks):
            a = (energy-2*t)*eye-block-t*t*previous
            if j == 0:
                a = a-sl
            else:
                b = -t*previous@b
            if j == len(self.blocks)-1:
                a = a-sr
            previous = solve(a, eye, assume_a="gen", check_finite=False)
            invs.append(previous)
            rhs.append(b)
        g_right_left = invs[-1]@rhs[-1]
        g = g_right_left
        for j in range(len(invs)-2, -1, -1):
            g = invs[j]@(rhs[j]-t*g)
        trans = -1j*(np.sqrt(gr)[:, None]*(self.right_u.T@g_right_left@self.left_u))*np.sqrt(gl)[None, :]
        refl = eye-1j*(np.sqrt(gl)[:, None]*(self.left_u.T@g@self.left_u))*np.sqrt(gl)[None, :]
        tp, rp = np.sum(abs(trans)**2, axis=0), np.sum(abs(refl)**2, axis=0)
        opened = gl > 1e-10
        error = float(np.max(abs(tp[opened]+rp[opened]-1))) if np.any(opened) else 0.0
        return {"energy": float(collision_energy), "N": float(tp.sum()),
                "T0": float(tp[0]), "R0": float(rp[0]), "unitarity_error": error,
                "transmission_by_incoming_channel": tp}

    def spectrum(self, energies):
        rows = [self.scatter(float(e)) for e in energies]
        return {key: np.array([row[key] for row in rows]) for key in
                ("energy", "N", "T0", "R0", "unitarity_error")}

    def thermal_rate(self, spectrum, beta):
        """Rate per unit asymptotic reactant line density (velocity units).

        E is measured from the full reactant transverse ground state. A basis
        size of one defines a projected/frozen mode, not a thermal oscillator.
        The partition sum uses the retained incoming channels and must converge.
        """
        if beta <= 0:
            raise ValueError("beta must be positive")
        e, n = np.asarray(spectrum["energy"]), np.asarray(spectrum["N"])
        if len(e) < 2 or np.any(np.diff(e) <= 0) or np.any(e <= 0):
            raise ValueError("Need an increasing positive energy grid")
        qrel = np.exp(-beta*(self.left_e-self.model.ground_energy)).sum()
        # Include E=0, where cumulative reaction probability vanishes.
        integral = trapezoid(np.r_[0., np.exp(-beta*e)*n], np.r_[0., e])
        value = np.sqrt(beta/(2*np.pi*self.model.masses[0]))*integral/qrel
        return float(value)
