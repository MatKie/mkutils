from mkutils import EvalReplica
import os

os.chdir("/Users/matthias/CODE/mkutils/mkutils/tests")
path = "/Users/matthias/CODE/mkutils/mkutils/tests/chunk_replica_testfiles"
basename = "REPLICA"
rho_filename = "solvent_nvt.prof"
props_filename = "props_nvt.out"

rho = EvalReplica(path, basename, rho_filename, simtype="CHUNKS", trim_data=True)
props = EvalReplica(path, basename, props_filename, simtype="LAMMPS")

rho.create_combined_property(
    ["density/number"],
    "density/weight",
    average=False,
    bounds=(10000, 100000),
    factor=0.01803 * 10 ** 7 / 6.022,
)

props.create_combined_property(["avg_ke", "thermo_pe"], "total_energy", average=False)

props.create_combined_property(
    ["avg_ke", "thermo_pe"], "total_energy", average=False, bounds=(1, 10)
)

# One output for each replica sim
print(props._replica_stats(xbounds=(0, 0.2)))
print(rho._replica_stats(bounds=(10000, 100000), xbounds=(0.1, -0.1)))


props.replica_stats(outfile="props.out")

rho.replica_stats(outfile="rho.out", xbounds=(0.1, -0.1), bounds=(10000, 1000000))

print("-------\n")
print(props.get_stats())
print("-------\n")


print(
    rho.get_stats(props="density/weight", bounds=(10000, 100000), xbounds=(0.1, -0.1))
)
print("-------\n")


print(rho.get_stats(bounds=(10000, 100000), xbounds=(0.1, -0.1)))

