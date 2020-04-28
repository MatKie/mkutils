from pytest import approx
from mkutils.mie import mie

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
