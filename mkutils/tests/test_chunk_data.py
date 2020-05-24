from mkutils import ChunkData
from mkutils import save_to_file
from mkutils import create_fig
import os
import pytest

os.chdir("/Users/matthias/CODE/mkutils/mkutils/tests")
solute = ChunkData("chunk_replica_testfiles/REPLICA1/solute_nvt.prof", trim_data=True)
solvent = ChunkData("chunk_replica_testfiles/REPLICA1/solvent_nvt.prof", trim_data=True)
bounds = (10000, 1000000)


def test_get_stats(sys, xbounds, bounds, prop="density/number"):
    ret = sys.get_stats(prop, xbounds=xbounds, bounds=bounds)
    print(ret)


test_get_stats(solvent, (0, 0.1), bounds)
test_get_stats(solvent, (0.1, -0.1), bounds)
test_get_stats(solvent, (0.4, 0.6), bounds)


def test_create_combined_property(sys, method, average, name):
    sys.create_combined_property(
        ["density/number", "Ncount"],
        name,
        bounds=(10000, 100000),
        method=method,
        average=average,
        factor=0.018015 * 10 ** 7 / 6.022,
    )
    print(name)


test_create_combined_property(solvent, "add", False, "1")
test_create_combined_property(solvent, "add", True, "2")
test_create_combined_property(solvent, "sub", False, "3")
test_create_combined_property(solvent, "sub", True, "4")
test_create_combined_property(solvent, "mul", False, "5")
test_create_combined_property(solvent, "mul", True, "6")
test_create_combined_property(solvent, "div", False, "7")
test_create_combined_property(solvent, "div", True, "8")


def test_molality(solu, solv, bounds):
    solv.create_combined_property(
        ["density/number"],
        "density/weight",
        bounds=bounds,
        method="multiply",
        average=False,
        factor=0.018015 * 10 ** 7 / 6.022,
    )

    rho_solv = solv.get_data("density/weight", bounds=bounds)

    solu.create_combined_property(
        ["density/number", rho_solv],
        "molality",
        bounds=bounds,
        method="div",
        factor=0.5 * 10 ** 7 / 6.022,
        average=False,
    )

    mol1 = solu.get_stats("molality", xbounds=(0.4, 0.6), bounds=bounds)

    solv.create_combined_property(
        ["density/number"],
        "density/weight2",
        bounds=bounds,
        method="multiply",
        average=False,
        factor=0.018015,
    )

    rho_solv = solv.get_data("density/weight2", bounds=bounds)

    solu.create_combined_property(
        ["density/number", rho_solv],
        "molality2",
        bounds=bounds,
        method="div",
        factor=0.5,
        average=False,
    )

    mol2 = solu.get_stats("molality2", xbounds=(0.4, 0.6), bounds=bounds)

    print(mol1)
    print(mol2)
    assert mol1[1] == pytest.approx(mol2[1], 1e-2)


test_molality(solute, solvent, bounds)


def test_get_stats2(solv, bounds, xbounds=(0.1, -0.1)):
    solv.create_combined_property(
        ["density/number"],
        "density/weight",
        bounds=bounds,
        method="multiply",
        average=False,
        factor=0.018015 * 10 ** 7 / 6.022,
    )

    rho_solv = solv.get_data("density/weight", bounds=bounds)
    print(solv.get_stats(bounds=bounds, xbounds=xbounds))
    print(solv.get_stats(props="density/weight", bounds=bounds, xbounds=xbounds))
    tupl_data = solv.get_data("density/weight", bounds=bounds)
    print(solv.get_stats(props=tupl_data, bounds=bounds, xbounds=xbounds))
    print(solv.get_stats(props=tupl_data[1], bounds=bounds, xbounds=xbounds))


test_get_stats2(solvent, bounds)
