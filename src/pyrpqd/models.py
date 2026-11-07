"""Three coupled Cartesian degrees of freedom and matched projections.

Force constants are held fixed when masses change. Reduced models project
omitted transverse coordinates onto their *reactant* oscillator ground state;
they are not frozen-coordinate or vibrationally adiabatic models.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Model:
    kind: str = "tunneling"
    coupling: tuple[float, float] = (0.3, 0.15)
    masses: tuple[float, float, float] = (1.0, 1.0, 1.0)
    frequencies: tuple[float, float] = (1.0, 1.5)
    height: float | None = None
    width: float | None = None
    separation: float = 2.0
    sigma: float = 0.5
    coupling_width: float = 1.0

    def __post_init__(self):
        if self.kind not in {"free", "resonance", "tunneling", "zpe"}:
            raise ValueError("Unknown model kind")
        if len(self.masses) != 3 or len(self.frequencies) != 2 or len(self.coupling) != 2:
            raise ValueError("Expected three masses and two transverse parameters")
        values = (*self.masses, *self.frequencies, self.separation,
                  self.sigma, self.coupling_width)
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("Masses, frequencies and length scales must be positive")
        if not np.all(np.isfinite(self.coupling)):
            raise ValueError("Couplings must be finite")
        if self.height is not None and (not np.isfinite(self.height) or self.height < 0):
            raise ValueError("Height must be finite and nonnegative")
        if self.width is not None and (not np.isfinite(self.width) or self.width <= 0):
            raise ValueError("Width must be finite and positive")

    @property
    def omega(self):
        return np.asarray(self.frequencies) / np.sqrt(self.masses[1:])

    @property
    def ground_energy(self):
        return float(self.omega.sum() / 2)

    @property
    def dividing_surface(self):
        return {"free": 0.0, "tunneling": 0.0,
                "resonance": self.separation, "zpe": 10.0}[self.kind]

    def fields(self, x):
        """Return U,U', force constants,k', centers,center' at x."""
        x = np.asarray(x, dtype=float)
        u, du = np.zeros_like(x), np.zeros_like(x)
        if self.kind == "resonance":
            h = 3.0 if self.height is None else self.height
            for a in (-self.separation, self.separation):
                g = h * np.exp(-0.5 * ((x-a)/self.sigma)**2)
                u += g
                du -= (x-a) * g / self.sigma**2
        elif self.kind in {"tunneling", "zpe"}:
            h = (4.0 if self.kind == "tunneling" else 0.2) if self.height is None else self.height
            a = (1.0 if self.kind == "tunneling" else 3.0) if self.width is None else self.width
            th = np.tanh(x/a)
            u = h * (1-th*th)
            du = -2*u*th/a
        k = np.broadcast_to(np.asarray(self.frequencies)**2, x.shape+(2,)).copy()
        dk = np.zeros_like(k)
        centers, dc = np.zeros_like(k), np.zeros_like(k)
        if self.kind == "zpe":
            a = 3.0 if self.width is None else self.width
            th = np.tanh(x/a)
            om = np.asarray(self.frequencies) + ((1+th)/2)[..., None]
            k = om**2
            dk = om * ((1-th*th)/a)[..., None]
        elif self.kind != "free":
            f = np.exp(-0.5*(x/self.coupling_width)**2)
            centers = f[..., None]*np.asarray(self.coupling)
            dc = -x[..., None]*centers/self.coupling_width**2
        return u, du, k, dk, centers, dc

    @staticmethod
    def validate_active(active):
        active = tuple(active)
        if not active or active[0] != 0 or len(set(active)) != len(active) or any(j not in (0,1,2) for j in active):
            raise ValueError("Active coordinates must be unique, start with x=0, and belong to 0,1,2")
        return active

    def potential_gradient(self, q, active=(0, 1, 2)):
        """Potential and Cartesian gradient, last axis follows active.

        Frozen reference energies cancel between numerator/denominator of rates.
        Their x-dependent ground-state expectation remains in the potential.
        """
        active = self.validate_active(active)
        q = np.asarray(q, dtype=float)
        if q.shape[-1] != len(active):
            raise ValueError("Coordinate shape does not match active dimensions")
        u, du, k, dk, c, dc = self.fields(q[..., 0])
        v, grad = u.copy(), np.zeros_like(q)
        grad[..., 0] = du
        for j in (1, 2):
            i = j-1
            if j in active:
                a = active.index(j)
                d = q[..., a]-c[..., i]
                v += 0.5*k[..., i]*d*d
                grad[..., a] = k[..., i]*d
                grad[..., 0] += 0.5*dk[..., i]*d*d-k[..., i]*d*dc[..., i]
            else:
                variance = 1/(2*self.masses[j]*self.omega[i])
                v += 0.5*(k[..., i]-self.frequencies[i]**2)*variance + 0.5*k[..., i]*c[..., i]**2
                grad[..., 0] += 0.5*dk[..., i]*(variance+c[..., i]**2)+k[..., i]*c[..., i]*dc[..., i]
        return v, grad

    def reference_potential(self, q, active=(0, 1, 2)):
        active = self.validate_active(active)
        out = np.zeros(np.asarray(q).shape[:-1])
        for a, j in enumerate(active):
            if j:
                out += 0.5*self.frequencies[j-1]**2*q[..., a]**2
        return out


def oscillator_operators(n, mass, omega):
    """Exact projections of q and q^2 (q^2 != truncated_q @ truncated_q)."""
    if n < 1:
        raise ValueError("Basis size must be positive")
    levels = np.arange(n)
    q = np.diag(np.sqrt(np.arange(1, n)/(2*mass*omega)), 1)
    q += q.T
    q2 = np.diag((2*levels+1)/(2*mass*omega))
    off = np.sqrt(np.arange(1, n-1)*np.arange(2, n))/(2*mass*omega)
    q2 += np.diag(off, 2)+np.diag(off, -2) if n > 2 else 0
    return omega*(levels+0.5), q, q2
