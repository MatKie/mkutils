import numpy as np
from .plot_sims import PlotSims
from .plot_gromacs import PlotGromacs
from .plot_lammps import PlotLAMMPS
from .chunk_data import ChunkData
import os
import copy


class EvalReplica(PlotSims):
    """
    Class designed to evaluate replicas of the same simulation.
    """

    def __init__(self, path, basename, filename, simtype="LAMMPS", trim_data=False):
        """
        simsuite is either LAMMPS or GROMACS or CHUNKS
        
        @ToDo 
            Implement ChunkData as well
        """
        self.path = path
        self.basename = basename
        self.filename = filename
        self.replica_dirs = self._collect_replica_dirs()
        self.replicas = self._collect_replicas()
        self.trim_data = trim_data  # For profiles...
        self.replica_stats_data = []

        # What are we evaluating?
        simsuites = {"LAMMPS": PlotLAMMPS, "GROMACS": PlotGromacs, "CHUNKS": ChunkData}
        if simtype not in simsuites.keys():
            raise ValueError("Simtype must be one of: {:s}".format(simsuites.keys()))
        else:
            self.simsuite = simsuites.get(simtype)

        # Get properties for any replica, they should all be the same
        # Checks should not be necessary...
        self.properties = self.simsuite(self.replicas[0]).get_properties()

        self.replica_sims = self._collect_replica_sims()

        self.combined_properties = []

    def _collect_replica_dirs(self):
        """
        Collect als directories in path and checks if basename is in there
        """

        # Get all directories in base dir
        base_dir_dirs = [
            os.path.join(self.path, di)
            for di in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, di))
        ]

        # Check if replica basename is in directory name to exclude
        # non simulation directories
        replica_dirs = [
            di for di in base_dir_dirs if self.basename in di.split(os.sep)[-1]
        ]

        return replica_dirs

    def _collect_replicas(self):
        """
        Goes in all the replica dirs and collects the filepaths
        """
        replicas = []
        for replica_dir in self.replica_dirs:
            filename = [
                fi for fi in os.listdir(replica_dir) if fi.startswith(self.filename)
            ]
            if len(filename) > 1.5:
                raise RuntimeError(
                    "More than one match for the given \
                                    filename ({:s}) in {:s}".format(
                        self.filename, replica_dir
                    )
                )
            elif len(filename) < 0.5:
                raise RuntimeError(
                    "Not match foundf or given filename \
                                    ({:s}) in {:s}".format(
                        self.filename, replica_dir
                    )
                )
            else:
                replicas.append(os.path.join(replica_dir, *filename))

        return replicas

    def _collect_replica_sims(self):
        """
        Inititialise objects according to the type of data
        """
        replica_sims = [self.simsuite(replica) for replica in self.replicas]
        if isinstance(replica_sims[0], ChunkData):
            for replica_sim in replica_sims:
                replica_sim.trim_data = self.trim_data
        # This would be a good place to implement keyword args like timestep

        return replica_sims

    def create_combined_property(
        self,
        props,
        name,
        average=True,
        difference=False,
        absolute_values=False,
        bounds=None,
        factor=None,
    ):
        self.combined_properties.append(name)
        for replica_sim in self.replica_sims:
            replica_sim.create_combined_property(
                props,
                name,
                average=average,
                difference=difference,
                absolute_values=absolute_values,
                bounds=bounds,
                factor=factor,
            )

    def replica_stats(
        self,
        props=True,
        blocks=10,
        bounds=None,
        xbounds=(0, 1),
        outfile="replica_stats.out",
    ):
        # list(replicas) of lists(prop, mean, std, drift)
        stats = self._replica_stats(
            props=props, blocks=blocks, bounds=bounds, xbounds=xbounds
        )
        properties = stats[0][0]
        means = []
        tmeans = [item[1] for item in stats]
        for i in range(len(properties)):
            tlist = [item[i] for item in tmeans]
            means.append(tlist)
        mean = [np.mean(meani) for meani in means]
        std = [np.std(meani) for meani in means]

        if outfile is not None:
            with open(outfile, "w") as f:
                for (i, prop), meani, stdi in zip(enumerate(properties), mean, std):
                    f.write(
                        "{:15s}\t{:12s}\t{:12s}\t{:12s}\n".format(
                            "Property", "Mean", "Error", "Drift"
                        )
                    )
                    for rep in stats:
                        args = [item[i] for item in rep]
                        string = "{:15s}".format(args[0])
                        for item in args[1:]:
                            string = "\t".join([string, "{:<12.4f}".format(item)])
                        f.write("{:s}\n".format(string))
                    f.write("{:s}\n".format("-" * 63))
                    f.write(
                        "{:15s}\t{:12s}\t{:12s}\n".format("Property", "Mean", "Error")
                    )
                    f.write(
                        "{:15s}\t{:<12.4f}\t{:<12.4f}\n\n".format(
                            str(prop), meani, stdi
                        )
                    )
        return properties, mean, std

    def _replica_stats(self, props=True, blocks=10, bounds=None, xbounds=(0, 1)):
        stats = []
        for replica_sim in self.replica_sims:
            tstats = replica_sim.get_stats(
                props=props, blocks=blocks, bounds=bounds, xbounds=xbounds
            )
            if isinstance(tstats[0], str):
                tstats = [[item] for item in tstats]
            stats.append(tstats)

        return stats

    def _replica_combine_data(self, prop, bounds=None):
        """
        This method returns the x,y of a combined trajcetory excluding equil.
        Specifically it makes the data more or less one trajectory
        """

        x0 = 0.0
        xs, ys = [], []
        for i, replica_sim in enumerate(self.replica_sims):
            xr, yr = copy.deepcopy(replica_sim.get_data(prop, bounds=bounds))
            xr -= xr[0]  # Value is non-zero for non-zero lower bound
            xr += x0 + xr[1] - xr[0]  # The dx is bc we miss one step
            x0 = xr[-1]  # Save end value of last trajectory
            xs.append(xr)
            ys.append(yr)

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        return x, y

    def get_stats(self, props=True, blocks=10, bounds=None, xbounds=None):
        """
        Calculate mean, error and drift on all or a selection of properties.
        
        Mean is calculated over 10 blocks (can also be specified)
        The error is the standard deviation of the these blocks -- You might
        want to multiply this by 2.5 to get the 95% confidence interval. 
        Unmultiplied only 68% confidence interval.
        Drift is calculated from a linear fit.
        """
        stats = []  # prop, avg, err, drift
        if props is True:
            props = self.properties
        elif isinstance(props, str):
            props = [props]
        elif isinstance(props, (tuple, list)):
            pass
        else:
            raise ValueError("props must be a string, list, tuple or True")

        replica_sim = self.replica_sims[0]

        if xbounds is not None:
            x0, xE = xbounds
            discretisation = 1.0 / len(replica_sim.x)
            bound_cross = False
            if xE < 0:
                xE = 1 + xE
                bound_cross = True
            index0 = int(np.round(x0 / discretisation))
            indexE = int(np.round(xE / discretisation))

        for prop in props:
            x, y = self._replica_combine_data(prop, bounds=bounds)
            if xbounds is not None:
                if bound_cross:
                    y = np.concatenate((y[:, :index0], y[:, indexE:]), axis=1)
                else:
                    y = y[:, index0:indexE]

            tstats = replica_sim._get_stats(x, y, blocks=blocks)
            stats.append([prop] + tstats)

        if len(stats) < 1.5:
            ret = tuple(item for item in stats[0])
            return ret
        else:
            ret = tuple([stat[i] for stat in stats] for i in range(len(stats[0])))
            return ret  # This could be adopted to only handle three stats
