from mkutils.replica import EvalReplica
from mkutils.plotting import PlotLAMMPS, ChunkData, save_to_file
from mkutils.plot_functions import create_fig
from mkutils.mie import mie, mixture
import os
import numpy as np
from eval_OPAS_functions import *

case = "mkutils/tests/chunk_replica_testfiles"
plot = True

path = os.path.join(os.getcwd(), case)
basename = "REPLICA"
outfile = "results_rep.out"
timestep = 5e-6
xy = 45.0
MW_solvent = 0.018015  # kg per mol
MW_solute = 0.05844

zo_l = 0.1
zo_h = -0.1
zi_l = 0.4
zi_h = 0.6
trim_data = True
tmin = 5000000
tmax = 10000000
T = 298.15
nu = 2

bounds = (tmin * timestep, tmax * timestep)
wallfile = "Wall_nvt.out"
solutefile = "solute_nvt.prof"
solventfile = "solvent_nvt.prof"

# For LRCs
Water = mie(8, 6, 305.21, 0.29016, rc=2)

Sodium = mie(8, 6, 8.92, 0.232, rc=2)
Chloride = mie(8, 6, 31.98, 0.334, rc=2)

Ws = mie.mix(Water, Sodium, k=-2.412976833)  # 179K
Wc = mie.mix(Water, Chloride, k=-1.053426773)  # 202K
Sc = mie.mix(Sodium, Chloride, k=-0.1634317133)  # 1 9K

Mix = mixture([Water, Sodium, Chloride, Ws, Wc, Sc])

### Start Evaluation
if replica is True and single_traj is True:
    Wall = EvalReplica(path, basename, wallfile)
    Wall0 = Wall.replica_sims[0]
    Solute = EvalReplica(
        path, basename, solutefile, simtype="CHUNKS", trim_data=trim_data
    )
    Solvent = EvalReplica(
        path, basename, solventfile, simtype="CHUNKS", trim_data=trim_data
    )
    Solute0 = Solute.replica_sims[0]  # one instance needed to evaluate data
    name = path
elif replica is True and single_traj is False:
    raise NotImplementedError
elif replica is False:
    Wall = PlotLAMMPS(os.path.join(path, basename, wallfile))
    Wall0 = Wall
    Solute = ChunkData(os.path.join(path, basename, solutefile), trim_data=trim_data)
    Solvent = ChunkData(os.path.join(path, basename, solventfile), trim_data=trim_data)
    Solute0 = Solute
    name = path
else:
    raise RuntimeError("Combination of evaluation procedure not possible")

av_osmP, err_osmP = eval_wall(Wall, Wall0, xy, bounds, plot=plot, path=name)

(
    av_mol,
    err_mol,
    av_brine,
    err_brine,
    av_solv,
    err_solv,
    av_solute,
    err_solute,
    moldens_solute,
    moldens_solute_err,
    moldens_solv_in,
    moldens_solv_in_err,
    moldens_solv_o,
    moldens_solv_o_err,
) = eval_densities(
    Solute,
    Solvent,
    Solute0,
    tmin,
    tmax,
    zi_h,
    zi_l,
    zo_h,
    zo_l,
    nu,
    MW_solute,
    MW_solvent,
    plot=plot,
    path=name,
)

av_osmPideal, err_osmPideal = osm_press_ideal(av_solv, av_mol, err_solv, err_mol)
av_osmC, err_osmC = osm_coeff(av_osmPideal, err_osmPideal, av_osmP, err_osmP)

# Calculate difference in LRCs
dp = eval_LRC(Mix, moldens_solv_in, moldens_solv_o, moldens_solute)
av_osmP_LRC = av_osmP + dp
av_osmC_LRC = av_osmP_LRC / av_osmPideal

# Write results one more average than
averages = [
    av_mol,
    av_osmP,
    av_osmC,
    av_brine,
    av_solv,
    dp,
    av_osmP_LRC,
    av_osmC_LRC,
    av_osmPideal,
]
errors = [
    err_mol,
    err_osmP,
    err_osmC,
    err_brine,
    err_solv,
    0,
    err_osmP,
    err_osmC,
    err_osmPideal,
]

write_results(averages, errors, path=os.path.join(name, outfile))
write_results(averages, errors, path=os.path.join(outfile))
