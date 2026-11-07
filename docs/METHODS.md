# Definitions and scientific validation

## Hamiltonians and dimensional reduction

All models use three Cartesian coordinates `(x,y,z)` and hbar = k_B = 1.
The PES is U(x) + sum_j k_j(x) [q_j-c_j(x)]^2/2. Force constants are independent
of isotope mass. The reactant oscillator frequencies are sqrt(k_j/m_j).

- Resonance: two Gaussians, height 3, centers +/-2, width 0.5;
  centers of transverse wells are (0.25,0.125) exp(-x^2/2).
- Tunneling: 4 sech^2(x); transverse displacements (0.3,0.15) exp(-x^2/2).
- ZPE: 0.2 sech^2(x/3); square roots of force constants change from (1,1.5)
  to (2,2.5) using [1+tanh(x/3)]/2. Both transverse modes are coupled to x.

For unit masses the ZPE model has reactant transverse ground energy 1.25 and
product ground energy 2.25. Thus a ground-channel collision energy below 1
cannot create outgoing product flux. A finite-width packet centered below 1
can still transmit its above-threshold momentum tail; these statements do
not contradict each other.

A one-dimensional model projects both y and z onto their *reactant* ground
states. A two-dimensional model retains x,y and projects z. It keeps <q^2>,
not <q>^2. Projection matrices use exact matrix elements of q^2, including
the top basis level. Squaring a truncated q matrix is incorrect here.
The reduced QD and RPMD Hamiltonians agree up to an irrelevant constant.

## Stationary QD

x is discretized with the second-order central finite difference. Transverse
harmonic-oscillator Galerkin bases provide the channel Hamiltonian at each x.
Semi-infinite lattice leads are attached using their analytic retarded
self-energies, including closed channels. A block Thomas solve obtains G_RL
and G_LL. Flux-normalized scattering amplitudes are

    t = -i sqrt(Gamma_R) U_R^T G_RL U_L sqrt(Gamma_L)
    r = I - i sqrt(Gamma_L) U_L^T G_LL U_L sqrt(Gamma_L).

The probabilities sum over open outgoing channels. T0 is ground-channel
transmission; N is transmission summed over all incoming channels. Numerical
unitarity tests the discrete solve, not continuum convergence.

Collision energy is measured from the full reactant transverse ground state.
For a converged channel basis the rate per unit reactant line density is

    k_v(beta) = sqrt(beta/(2*pi*m_x)) / q_perp_rel
                * integral_0^infinity exp(-beta*E) N(E) dE.

The retained lead channel energies define q_perp_rel; basis convergence must
also establish convergence of this partition sum. In the free limit the
answer is 1/sqrt(2*pi*beta*m_x). No eta broadening or absorbing potential is
used in stationary scattering. Finite lead length still requires a check.

## Absolute RPMD flux-side estimator

For P beads, omega_P = P/beta and omega_k = 2P/beta sin(pi*k/P). Independent
reference configurations are sampled exactly from a free x ring with fixed
centroid xi and harmonic reactant transverse rings. Their covariance in
orthonormal real normal modes is P/[beta*m*(omega_k^2+omega^2)].

Define w = exp[-beta/P sum_b (V(q_b)-V_reference(q_b))]. The mean of w is
the constrained density ratio to asymptotic reactants; no fitted free-energy
normalization or umbrella integration is needed for these special models.

The positive centroid flux distribution is Rayleigh with scale
1/sqrt(beta*m_x), not a half-normal Maxwell distribution. Internal velocities
and transverse velocities have the usual Maxwell distribution at beta/P.
For every path, propagate v and -v with no thermostat. Then

    k_v(t) = 1/sqrt(2*pi*beta*m_x)
             * E_reference[ w * (h(xbar_plus(t)-xi)-h(xbar_minus(t)-xi)) ].

The last fifth of saved times is averaged per independent path. Standard
errors are over paths, not over correlated time points. Importance ESS,
flux-side plateau span and relative energy drift are reported. Diagnostic
pass is not bead, time-step, statistical-tail or dividing-surface convergence.
Deep tunneling can make Gaussian importance sampling inefficient; the code
reports this rather than silently interpreting an unstable estimator.

## A: matched-dimensional correction

    log(k_A_3D) = log(k_RPMD_3D) + log(k_QD_red) - log(k_RPMD_red).

The function has no 3D-QD argument. All errors against 3D QD are evaluated
after prediction. The delta-method uncertainty assumes independent full and
reduced samples and excludes QD quadrature/discretization error. It is not
reliable for nearly zero or very uncertain denominators. Failure to improve
is a valid outcome and must not be filtered out of the report.

## B: variational mean-field Q/RP baseline

Only environment coordinates have beads. The explicit quantum wavefunction
(or sampled mixture) is propagated on one or two Cartesian quantum grids.
The Hamiltonian is

    H = <psi| T_q + (1/P) sum_b V(q,Q_b) |psi>
        + (1/P) sum_b [ M v_b^2/2 + M omega_P^2 (Q_b-Q_(b+1))^2/2 ].

Bath canonical momenta are p_b=M v_b/P, so the bath acceleration from the
interaction is -<dV/dQ_b>/M. An additional 1/P in the acceleration would be
inconsistent with this Hamiltonian. A symmetric split alternates exact
potential kicks/phases with quantum FFT kinetic propagation and exact
free-ring normal-mode propagation. Norm and the stated H are monitored.

For linearly displaced harmonic modes, bead internal fluctuations contribute
only a scalar quantum phase; the coupling to system motion is through the
centroid. This baseline does not magically recover quantum bath coherence.
For coordinate-dependent force constants (ZPE model), bead second moments
also enter the system Hamiltonian. Neither case is an exact entangled state.

An incoming Gaussian of position-probability standard deviation sigma has
momentum standard deviation 1/(2*sigma). The nominal collision energy is
p0^2/(2m), and the mean kinetic energy additionally contains 1/(8*m*sigma^2).
The exact reference can sum a truncated thermal oscillator mixture; the
omitted initial probability weight is explicitly returned. Hybrid trajectories
sample the corresponding thermal bath rings and retained quantum states.

Propagation is unitary on a periodic grid, without CAP or wavefunction
renormalization. Non-negligible boundary occupancy requires a larger box.
Finite-time product population can remain below the asymptotic transmission
because of an unescaped resonant component. Report the interaction-region
population and do not relabel these values as rates or monochromatic P(E).

## Required checks before publication-level claims

1. QD: independently refine x spacing, transverse basis, lead length, initial
   energy scan, peak refinement, energy cutoff and thermal partition sum.
2. RPMD: independently refine beads, integration step, time, surface and sample
   count; use independent seeds and ensure importance ESS is adequate.
3. Hybrid: quantum grids/time step, bath beads, ensemble count, boundary and
   finite-time completion; test decoupled limits and energy conservation.
4. Compare statistical and numerical errors before claiming improvement;
   do not tune correction parameters to the 3D validation reference.

## Primary references

- Craig & Manolopoulos, Chemical reaction rates from ring polymer molecular
  dynamics, JCP 122, 084106 (2005), https://doi.org/10.1063/1.1850093
- Craig & Manolopoulos, A refined ring polymer molecular dynamics theory of
  chemical reaction rates, JCP 123, 034102 (2005), https://doi.org/10.1063/1.1954769
- Suleimanov, Aoiz & Guo, Chemical Reaction Rate Coefficients from Ring Polymer
  Molecular Dynamics: Theory and Practical Applications, JPCA 120, 8488 (2016),
  https://doi.org/10.1021/acs.jpca.6b07140
- Tao, Shushkov & Miller, Path-integral isomorphic Hamiltonian for including
  nuclear quantum effects in non-adiabatic dynamics, JCP 148, 102327 (2018),
  https://doi.org/10.1063/1.5005544

The last reference is a related method, not the implementation of B in this
repository. The simple variational baseline must not be cited as iso-RPMD.
