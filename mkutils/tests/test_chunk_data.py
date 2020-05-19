from mkutils import ChunkData
from mkutils import save_to_file
from mkutils import create_fig
import os

os.chdir("/Users/matthias/CODE/mkutils/mkutils/tests")
solute = ChunkData("chunk_replica_testfiles/REPLICA1/solute_nvt.prof", trim_data=True)
sys = ChunkData("chunk_replica_testfiles/REPLICA1/solvent_nvt.prof", trim_data=True)


print(sys.get_stats("density/number", xbounds=(0.0, 0.1), bounds=(10000, 1000000)))
print(sys.get_stats("density/number", xbounds=(0.1, -0.1), bounds=(10000, 1000000)))
print(sys.get_stats("density/number", xbounds=(0.4, 0.6), bounds=(10000, 1000000)))

sys.create_combined_property(
    ["density/number"],
    "density/weight",
    bounds=(10000, 1000000),
    method="multiply",
    average=False,
    factor=0.018015 * 10 ** 7 / 6.022,
)

print(sys.get_stats("density/weight", xbounds=(0.0, 0.1), bounds=(10000, 1000000)))

print(sys.get_stats("density/weight", xbounds=(0.1, -0.1), bounds=(10000, 1000000)))

solute.create_combined_property(
    ["density/number"],
    "density/weight",
    bounds=(10000, 1000000),
    method="multiply",
    average=False,
    factor=0.05844 / 2 * 10 ** 7 / 6.022,
)

data = solute.get_data("density/weight", bounds=(10000, 1000000))

sys.create_combined_property(
    [data, "density/weight"],
    "density/ratio",
    bounds=(10000, 1000000),
    method="div",
    factor=None,
    average=False,
)

print(sys.get_stats("density/ratio", xbounds=(0.1, -0.1), bounds=(10000, 1000000)))
print(sys.get_stats("density/ratio", xbounds=(0.4, 0.6), bounds=(10000, 1000000)))
