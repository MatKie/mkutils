from pytest import approx
from mkutils.mie import mie, mixture

CGW2_dens = 5000 / (6.68257 ** 3)  # 1/nm^3
eps = 400  # K
sigma = 0.37467  # nm


def test_LRCp_single():
    system = mie(8, 6, eps, sigma, rc=2)
    lrc_p = system.LRC_p(CGW2_dens)
    assert lrc_p == approx(-206.93, 0.01)


def test_LRCen_single():
    system = mie(8, 6, eps, sigma, rc=2.3)
    lrc_en = system.LRC_en(CGW2_dens)
    # Multiply by 5000 to account for N and divide by 1000 (kJ)
    lrc_en *= 5.0
    assert lrc_en == approx(-1237.88, 0.01)


def test_mix_single():
    sys1 = mie(8, 6, eps, sigma, rc=2)
    sys2 = mie(8, 6, eps, sigma, rc=2)
    m12 = mie.mix(sys1, sys2)
    assert m12.eps == approx(eps, 0.0001)
    assert m12.sig == approx(sigma, 0)
    assert m12.l_r == approx(8, 0)
    assert m12.l_a == approx(6, 0)
    assert m12.rc == approx(2, 0)


def test_pure_mixture():
    sys1 = mie(8, 6, eps, sigma, rc=2)
    sys2 = mie(8, 6, eps, sigma, rc=2)
    sys3 = mie(8, 6, eps, sigma, rc=2)

    m12 = mie.mix(sys1, sys2)
    m13 = mie.mix(sys1, sys3)
    m23 = mie.mix(sys2, sys3)

    mix = mixture([sys1, sys2, sys3, m12, m13, m23])
    LRC_p = mix.LRC_p(CGW2_dens, [0.6, 0.3, 0.1])
    LRC_control = sys1.LRC_p(CGW2_dens)

    assert LRC_p == approx(LRC_control, 1e-4)


def test_NaF_mixture():
    # For future reference the energy correction
    # on this is: -976.066 for 6783 water and
    # 123 sodium and chloride ions resp.
    rc = 1.4
    # Spc(o) 78.249K, 0.3166nm
    sys1 = mie.lj(78.249, 0.3166, rc=rc)
    # Na 38.487K, 0.2450nm
    sys2 = mie.lj(38.487, 0.245, rc=rc)
    # F 120.272K, 0.3700nm
    sys3 = mie.lj(120.272, 0.37, rc=rc)

    rule = "geom"
    m12 = mie.mix(sys1, sys2, k=0.2504612547, rule=rule)
    assert m12.eps == approx(41.13, 0.01)
    m13 = mie.mix(sys1, sys3, rule=rule)
    m23 = mie.mix(sys2, sys3, rule=rule)

    mix = mixture([sys1, sys2, sys3, m12, m13, m23])
    composition = [0.965002134, 0.01749893299, 0.01749893299]
    dens = 7029.0 / (5.88702 ** 3)
    LRC_p = mix.LRC_p(dens, composition)

    assert LRC_p == approx(-79.4747, 0.01)


eps_W = 305.21  # Kelvin
sigma_W = 0.29016  # nm
lambda_W = 8.0
sigma_T = 0.45012
eps_T = 358.37
lambda_T = 15.947


def test_SAFT_rules():
    mie_W = mie(lambda_W, 6.0, eps_W, sigma_W)
    mie_T = mie(lambda_T, 6.0, eps_T, sigma_T)

    mie_WT = mie.mix(mie_W, mie_T, rule="SAFT", k=0.31)
    assert mie_WT.eps == approx(212.40, abs=0.01)
    assert mie_WT.sig == approx(0.37014, abs=1e-5)
    assert mie_WT.l_r == approx(11.046, abs=0.001)


def test_epsilon():
    assert mie.cgw_ift(293.0).eps == approx(304.28, 1e-2)
    assert mie.cgw_ift(298).eps == approx(305.21, 1e-3)
    assert mie.cgw_ift(463.0).eps == approx(345.43, 1e-3)
    assert mie.cgw_ift(343).eps == approx(318.84, 1e-3)


def test_sigma():
    assert mie.cgw_ift(293).sig == approx(2.9055e-1, 1e-3)
    assert mie.cgw_ift(298).sig == approx(2.9016e-1, 1e-4)
    assert mie.cgw_ift(343).sig == approx(2.8811e-1, 1e-4)
    assert mie.cgw_ift(463).sig == approx(2.866e-1, 5e-4)


def test_prefacs():
    water = mie.cgw_ift(298.15)
    C6 = water.get_C_attr()
    C8 = water.get_C_rep()
    assert C6 == approx(1.435826405e-2)
    assert C8 == approx(1.208769438e-3)


def test_eps_mix():
    mie_W = mie(lambda_W, 6.0, eps_W, sigma_W)
    mie_T = mie(lambda_T, 6.0, eps_T, sigma_T)

    mie_WT = mie.mix(mie_W, mie_T, rule="SAFT", k=111, eps_mix=212.4)
    assert mie_WT.eps == approx(212.40, abs=0.01)
    assert mie_WT.sig == approx(0.37014, abs=1e-5)
    assert mie_WT.l_r == approx(11.046, abs=0.001)
