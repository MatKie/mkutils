import numpy as np
import warnings
import matplotlib
from mk.plotting import PlotSims, PlotGromacs, PlotLAMMPS, ChunkData
import os

class EvalReplica(PlotSims):
    """
    Class designed to evaluate replicas of the same simulation.
    """
    def __init__(self, path, basename, filename, simtype="LAMMPS"):
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
        
        
        # What are we evaluating?
        simsuites = {"LAMMPS": PlotLAMMPS, "GROMACS": PlotGromacs,
                    "CHUNKS": ChunkData}
        if simtype not in simsuites.keys():
            raise ValueError("Simtype must be one of: {:s}".format(
                                                        simsuites.keys()))
        else:
            self.simsuite = simsuites.get(simtype)

        # Get properties for any replica, they should all be the same
        # Checks should not be necessary...
        self.properties = self.simsuite(self.replicas[0]).get_properties()
        
        self.replica_sims = self._collect_replica_sims()

        self.combined_properties = []

    def _collect_replica_dirs(self):
        # Get all directories in base dir
        base_dir_dirs = [os.path.join(self.path, di) for di in 
                os.listdir(self.path) if 
                os.path.isdir(os.path.join(self.path, di))]

        # Check if replica basename is in directory name to exclude
        # non simulation directories
        replica_dirs = [di for di in base_dir_dirs if self.basename in 
                             di.split(os.sep)[-1]]
        
        return replica_dirs 

    def _collect_replicas(self):
        replicas = []
        for replica_dir in self.replica_dirs:
            filename = [fi for fi in os.listdir(replica_dir) 
                        if fi.startswith(self.filename)]
            if len(filename) > 1.5:
                raise RuntimeError("More than one match for the given \
                                    filename ({:s}) in {:s}".format(
                                    self.filename, replica_dir))
            elif len(filename) < 0.5:
                raise RuntimeError("Not match foundf or given filename \
                                    ({:s}) in {:s}".format(
                                    self.filename, replica_dir))
            else:
                replicas.append(os.path.join(replica_dir, *filename))

        return replicas

    def _collect_replica_sims(self):
        replica_sims = [self.simsuite(replica) for replica in self.replicas]
        # This would be a good place to implement keyword args like timestep
        
        return replica_sims

    def create_combined_property(self, props, name, average=True, 
                difference=False, absolute_values=False):
        self.combined_properties.append(name)
        for replica_sim in self.replica_sims:
            replica_sim.create_combined_property(props, 
                                                name, 
                                                average=average, 
                                                difference=difference, 
                                            absolute_values=absolute_values)

    def create_combined_profile(self, props, tmin=None, tmax=None,
                                method="add", factor=None, average=True):
        
        data = []
        for replica_sim in self.replica_sims:
            data.append(replica_sim.combine_quantity(props, 
                                                    tmin=tmin, tmax=tmax,
                                                    method=method,
                                                    factor=factor,
                                                    average=average))

        profile = np.concatenate(data)
        
        return profile

    def replica_stats(self, props=True, blocks=10, bounds=None):
        properties, avs, errs, drifts = [], [], [], []
        for replica_sim in self.replica_sims:
            prop, av, err, drift = replica_sim.get_stats(props=props,
                                                         blocks=blocks,
                                                         bounds=bounds)
            properties.append(prop)
            avs.append(av)
            errs.append(err)
            drifts.append(drift)

        return properties, avs, errs, drifts

    def _replica_combine_data(self, pos, bounds=None):
        """
        This method returns the x,y of a combined trajcetory excluding equil.
        """

        x0 = 0.0
        xs, ys = [], []
        for i, replica_sim in enumerate(self.replica_sims):
            xr, yr = replica_sim._get_data(pos, bounds=bounds)
            xr -= xr[0] # Value is non-zero for non-zero lower bound
            xr += x0 + xr[1] - xr[0] # The dx is bc we miss one step
            x0 = xr[-1] # Save end value of last trajectory
            xs.append(xr)
            ys.append(yr)

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        
        return x , y

    def get_stats(self, props=True, blocks=10, bounds=None):
        """
        Calculate mean, error and drift on all or a selection of properties.
        
        Mean is calculated over 10 blocks (can also be specified)
        The error is the standard deviation of the these blocks -- You might
        want to multiply this by 2.5 to get the 95% confidence interval. 
        Unmultiplied only 68% confidence interval.
        Drift is calculated from a linear fit.
        """ 
        stats = [] #prop, avg, err, drift
        if props is True:
            props = self.properties
        elif isinstance(props, str):
            props = [props]
        elif isinstance(props, (tuple, list)):
            pass
        else:
            raise ValueError("props must be a string, list, tuple or True")   
        
        replica_sim = self.replica_sims[0] 
        
        for prop in props:
            pos = replica_sim._get_pos(prop)
            x, y = self._replica_combine_data(pos, bounds=bounds)
            tlist = [prop]
            tstats = replica_sim._get_stats(x,y,blocks=blocks, bounds=bounds)
            stats.append([prop] + tstats)
        
        if len(stats) < 1.5:
            return stats[0][0], stats[0][1], stats[0][2], stats[0][3]
        else:
            return ([sublist[0] for sublist in stats], 
                    [sublist[1] for sublist in stats],
                    [sublist[2] for sublist in stats], 
                    [sublist[3] for sublist in stats])
