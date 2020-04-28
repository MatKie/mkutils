from pytest import approx
from mkutils.mie import mie

CGW2_dens = 5000 / (6.68257 ** 3)  # 1/nm^3
eps = 400  # K
sigma = 0.37467  # nm
rc = 2.0  # nm


def test_LRCp_single():
    system = mie(8, 6, eps, sigma, rc=2)
    lrc_p = system.LRC_p(CGW2_dens)
    assert lrc_p == approx(-206.93, 0.01)
