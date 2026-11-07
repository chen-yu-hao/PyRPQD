import numpy as np
import pytest
from pyrpqd import Model, ScatteringSolver
from pyrpqd.models import oscillator_operators
from pyrpqd.rpmd import harmonic_ring, spring_frequencies, thermal_rate, free_ring_step
from pyrpqd.correction import corrected_rate
from pyrpqd.wavepacket import PacketConfig, propagate


@pytest.mark.parametrize("kind", ["free", "resonance", "tunneling", "zpe"])
@pytest.mark.parametrize("active", [(0,), (0,1), (0,2), (0,1,2)])
def test_analytic_gradient(kind, active):
    model = Model(kind, masses=(2.,3.,4.))
    q = np.random.default_rng(11).normal(size=(4, len(active)))
    g = model.potential_gradient(q,active)[1]
    for j in range(len(active)):
        delta = np.zeros_like(q)
        delta[:,j] = 1e-6
        fd = (model.potential_gradient(q+delta,active)[0]-model.potential_gradient(q-delta,active)[0])/2e-6
        np.testing.assert_allclose(g[:,j],fd,atol=1e-8,rtol=1e-6)


def test_mass_change_preserves_pes():
    q = np.array([[.4,.1,-.2]])
    v1 = Model(masses=(1.,1.,1.)).potential_gradient(q)[0]
    v2 = Model(masses=(2.,3.,4.)).potential_gradient(q)[0]
    np.testing.assert_array_equal(v1,v2)


def test_projected_q_squared_is_not_squared_projection():
    _, q, q2 = oscillator_operators(1,1.,1.)
    assert q[0,0] == 0
    assert q2[0,0] == .5
    _, q, q2 = oscillator_operators(3,1.,1.)
    assert q2[-1,-1] == 2.5
    assert not np.allclose(q@q,q2)


@pytest.mark.parametrize("basis", [(1,1),(3,2)])
def test_free_scattering_unitarity(basis):
    solver = ScatteringSolver(Model("free"),basis,dx=.3,extent=3)
    for energy in [.3,1.2,2.1]:
        result = solver.scatter(energy)
        assert result["T0"] == pytest.approx(1,abs=1e-12)
        assert result["R0"] < 1e-20
        assert result["unitarity_error"] < 1e-12


def test_sech_barrier_analytic_transmission():
    # Exact continuum result for V0 sech^2(x/a), 8mV0a^2 > 1.
    energy, height = 1.3, 2.
    a = np.sinh(np.pi*np.sqrt(2*energy))**2
    exact = a/(a+np.cosh(np.pi*np.sqrt(8*height-1)/2)**2)
    model = Model("tunneling",coupling=(0.,0.),height=height)
    coarse = ScatteringSolver(model,(1,1),dx=.2,extent=12).scatter(energy)["T0"]
    fine = ScatteringSolver(model,(1,1),dx=.05,extent=12).scatter(energy)["T0"]
    assert abs(fine-exact) < abs(coarse-exact)/8
    assert fine == pytest.approx(exact,rel=.003)


def test_zpe_closed_product_channel():
    solver = ScatteringSolver(Model("zpe"),(6,6),dx=.3,extent=20)
    result = solver.scatter(.8)
    assert result["T0"] == 0
    assert result["R0"] == pytest.approx(1,abs=1e-11)
    assert solver.scatter(1.5)["T0"] > .9


def test_fixed_projection_matches_analytic_effective_potential():
    model = Model("zpe")
    solver = ScatteringSolver(model,(1,1),dx=.5,extent=10)
    v = model.potential_gradient(solver.x[:,None],(0,))[0]
    np.testing.assert_allclose(solver.blocks[:,0,0]-model.ground_energy,v,atol=1e-14)


def test_ring_sampling_finite_bead_variance():
    rng = np.random.default_rng(4)
    beads,beta,mass,omega = 16,3.,2.,1.3
    q = harmonic_ring(rng,10000,beads,beta,mass,omega)
    exact = np.sum(1/(spring_frequencies(beads,beta)**2+omega**2))/(beta*mass)
    assert np.mean(q*q) == pytest.approx(exact,rel=.025)
    assert np.var(q.mean(axis=1)) == pytest.approx(1/(beta*mass*omega**2),rel=.04)


def test_free_ring_reversibility():
    rng = np.random.default_rng(3)
    q,v = rng.normal(size=(2,4,3)),rng.normal(size=(2,4,3))
    q1,v1 = free_ring_step(q,v,.17,2.)
    q2,v2 = free_ring_step(q1,-v1,.17,2.)
    np.testing.assert_allclose(q,q2,atol=1e-14)
    np.testing.assert_allclose(v,-v2,atol=1e-14)


@pytest.mark.parametrize("beads", [1,8])
def test_free_rpmd_absolute_rate(beads):
    result = thermal_rate(Model("free"),beta=2.,beads=beads,samples=64,time=.2,dt=.02)
    assert result["rate"] == pytest.approx(1/np.sqrt(4*np.pi),abs=1e-13)
    assert result["kappa"][-1] == 1
    assert result["stderr"] < 1e-14


def test_log_correction_and_invalid_rates():
    r = corrected_rate({"rate":2.,"stderr":.1},1.5,{"rate":1.,"stderr":.05})
    assert r["rate"] == pytest.approx(3)
    assert r["uses_3d_qd_for_fitting"] is False
    with pytest.raises(ValueError):
        corrected_rate({"rate":0.,"stderr":0.},1.,{"rate":1.,"stderr":0.})


@pytest.mark.parametrize("active", [(0,), (0,1), (0,2), (0,1,2)])
def test_wavepacket_norm(active):
    config = PacketConfig(nx=64,nt=12,x_extent=32,x0=-10,sigma=2,
                          time=.2,dt=.02,samples=3,thermal_levels=1)
    result = propagate(Model("tunneling"),config,active)
    assert result["trace"][-1]["norm"] == pytest.approx(1,abs=1e-12)
    assert result["max_relative_energy_drift"] < 1e-3


def test_hybrid_decoupled_limit():
    config = PacketConfig(nx=128,nt=16,x_extent=40,x0=-10,sigma=2,
                          time=8.,dt=.04,samples=3,thermal_levels=1)
    qd = propagate(Model("free"),config,(0,1,2))
    hybrid = propagate(Model("free"),config,(0,))
    assert qd["product_probability"] == pytest.approx(hybrid["product_probability"],abs=1e-11)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Model("unknown")
    with pytest.raises(ValueError):
        Model(masses=(1,0,1))
    with pytest.raises(ValueError):
        ScatteringSolver(Model(),(0,1))
    with pytest.raises(ValueError):
        ScatteringSolver(Model()).scatter(-1)


@pytest.mark.parametrize("kind", ["resonance", "tunneling", "zpe"])
def test_coupled_scattering_flux_balance(kind):
    solver = ScatteringSolver(Model(kind),(3,3),dx=.25,extent=12)
    for e in [.4,1.3,2.7]:
        r = solver.scatter(e)
        assert -1e-12 <= r["T0"] <= 1+1e-12
        assert r["unitarity_error"] < 1e-10


def test_hybrid_energy_step_refinement():
    common = dict(energy=2.,x0=-3.,sigma=.5,x_extent=16.,nx=128,nt=16,
                  time=2.,beta=3.,beads=8,samples=4,thermal_levels=1)
    coarse = propagate(Model("tunneling"),PacketConfig(dt=.08,**common),(0,))
    fine = propagate(Model("tunneling"),PacketConfig(dt=.04,**common),(0,))
    assert fine["max_relative_energy_drift"] < coarse["max_relative_energy_drift"]/2
