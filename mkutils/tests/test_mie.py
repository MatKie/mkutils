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
