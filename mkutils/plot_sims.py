import matplotlib.pyplot as plt
import numpy as np


class PlotSims:
    def __init__(self, infile):
        self.infile = infile
        self.properties = self.get_properties()
        self.timestep = 1e-5  # ns
        self.combined_properties = []
        self.combined_properties_data = []

    def get_properties(self):
        return self.properties + self.combined_properties

    def _get_data(self, pos, bounds=None):
        return [0], [0]

    def _get_unit(self, pos):
        return ""

    def show_properties(self):
        for prop in self.properties:
            print(prop)
        for prop in self.combined_properties:
            print(prop)

    def _get_pos(self, prop):
        if prop in self.properties:
            return self.properties.index(prop) + 1
        elif prop in self.combined_properties:
            ind = self.combined_properties.index(prop) + len(self.properties) + 1
            return ind
        else:
            raise ValueError("Property not available")

    def plot_timeseries(
        self,
        ax,
        prop,
        blocksize=10,
        timeseries=False,
        bounds=None,
        color=None,
        alpha=1,
        label=None,
        **kwargs
    ):
        """
        Plot the block averaged timeseries.
        """
        pos = self._get_pos(prop)

        label = self._get_label(prop, label)

        x, y = self._get_data(pos, bounds=bounds)

        bounds = (x[0], x[-1])

        if blocksize:
            x_av = self.block_average(x, blocksize=blocksize)
            y_av = self.block_average(y, blocksize=blocksize)
            ax.plot(x_av, y_av, color=color, label=label, alpha=alpha, **kwargs)
            alpha = 0.4
            label = ""
        if timeseries:
            ax.plot(x, y, label=label, color=color, alpha=alpha, **kwargs)

        self._set_xlims(ax, x, bounds=bounds)
        unit = self._get_unit(prop)
        ax.set_xlabel("Time / ns")
        ax.set_ylabel("{:s} / {:s}".format(prop, unit))
        plt.tight_layout()

    def plot_mean(
        self,
        ax,
        prop,
        bounds=None,
        drift=False,
        err=False,
        blocksize=15,
        color=None,
        alpha=1,
        label=None,
    ):

        label = self._get_label(prop, label)

        pos = self._get_pos(prop)

        x, y = self._get_data(pos, bounds=bounds)

        blocks = self.block_average(y, blocksize=int(len(x) / blocksize))
        mean = np.mean(blocks)
        ax.plot((x[0], x[-1]), (mean, mean), color=color, label=label)

        if err:
            std = np.std(blocks)
            ax.fill_between(
                (x[0], x[-1]),
                (mean - std, mean - std),
                (mean + std, mean + std),
                color=color,
                alpha=0.4,
            )

        if drift:
            ystart, yend = self._calc_drift(x, y)
            ax.plot((x[0], x[-1]), (ystart, yend), color=color, linestyle=":")

        self._set_xlims(ax, x, bounds=bounds)
        unit = self._get_unit(prop)
        ax.set_xlabel("Time / ns")
        ax.set_ylabel("{:s} / {:s}".format(prop, unit))
        plt.tight_layout()

    def _calc_drift(self, x, y):
        linfit = np.polyfit(x, y, 1)
        ystart = x[0] * linfit[0] + linfit[1]
        yend = x[-1] * linfit[0] + linfit[1]

        return ystart, yend

    def _get_label(self, prop, label):
        if label is not None:
            pass
        else:
            label = prop
        return label

    def block_average(self, x, blocksize=20):
        Nobs = len(x)
        blocks = int(Nobs / blocksize)
        cutout = Nobs - (blocks * blocksize)
        # As of now just discard the first few samples if they do not
        # fit in a nice array
        x = np.reshape(x[cutout:], (blocks, blocksize, 1))
        return np.mean(x, axis=1)

    def _set_xlims(self, ax, x, bounds=None):
        if bounds is not None:
            ax.set_xlim(bounds[0], bounds[1])
        else:
            ax.set_xlim(x[0], x[-1])

    def _get_stats(self, x, y, blocks=10):
        blocksize = int(x.size / blocks)
        blocks = self.block_average(y, blocksize=blocksize)
        mean = np.mean(blocks)
        err = np.std(blocks)
        start, end = self._calc_drift(x, y)
        return [mean, err, end - start]

    def get_stats(self, props=True, blocks=10, bounds=None, *args, **kwargs):
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

        for prop in props:
            pos = self._get_pos(prop)
            x, y = self._get_data(pos, bounds=bounds)
            tstats = self._get_stats(x, y, blocks=blocks)
            stats.append([prop] + tstats)

        if len(stats) < 1.5:
            return stats[0][0], stats[0][1], stats[0][2], stats[0][3]
        else:
            return (
                [sublist[0] for sublist in stats],
                [sublist[1] for sublist in stats],
                [sublist[2] for sublist in stats],
                [sublist[3] for sublist in stats],
            )

    def write_stats(self, ofile="energies.out", props=True, blocks=10, bounds=None):
        stats = self.get_stats(props=props, blocks=blocks, bounds=bounds)
        with open(ofile, "w") as f:
            f.write(
                "{:<18s}{:<12s}{:<12s}{:<12s}\n".format(
                    "Property", "Mean", "Error", "Drift"
                )
            )
            for prop, mean, err, drift in zip(*stats):
                f.write(
                    "{:<18s}{:<12.4f}{:<12.4f}{:<12.4f}\n".format(
                        prop, mean, err, drift
                    )
                )
        # Routine to write shizzle

    def create_combined_property(
        self,
        props,
        name,
        average=True,
        difference=False,
        absolute_values=False,
        *args,
        **kwargs
    ):
        """
        This method creates combined properties from input data.

        Inputs:
        -------
            props (list, tuple) : name of properties as in input file
            name (str) : Name of created property
            average (bool): wether or not to average the quantity at 
                            timestep
            difference (bool): Substracts the data of second prop from
                               second prop
            absolute_value (bool) : Takes absolute values of all involved
                                    properties

        output:
        -------
            None, append property and values to self.combined_properties and
            self.combined_properties_data
        """

        if not isinstance(props, (list, tuple)):
            raise ValueError("props must be list or tuple")

        # Read first element to get the right shapes
        pos = self._get_pos(props[0])
        x, y = self._get_data(pos, bounds=None)
        values = len(y)
        datasets = np.zeros((len(props), values))
        # datasets[0,:] = y
        for i, prop in enumerate(props):
            pos = self._get_pos(prop)
            x, y = self._get_data(pos, bounds=None)
            if absolute_values is True:
                y = np.absolute(y)
            # Each property is one row
            datasets[i, :] = y

        if average is True:
            # axis=0 sums over columns, averaging the properties at each
            # timestep
            res = np.average(datasets, axis=0)
        elif difference is True:
            res = np.subtract(datasets[0, :], datasets[1, :])
        else:
            # summing properties over columns
            res = np.sum(datasets, axis=0)

        # Here I transpose the matrix again... Sorry for that
        # Now we have x, y as column vectors again
        combined_property = np.zeros((values, 2))
        combined_property[:, 0] = x / self.timestep
        combined_property[:, 1] = res
        self.combined_properties.append(name)
        self.combined_properties_data.append(combined_property)
