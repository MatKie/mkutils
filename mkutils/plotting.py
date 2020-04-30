import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import copy


def save_to_file(filename, fig=None):
    """Save to @filename with a custom set of file formats.
    
    By default, this function takes to most recent figure,
    but a @fig can also be passed to this function as an argument.
    """
    formats = [
        "pdf",
        "png",
    ]
    if fig is None:
        for form in formats:
            plt.savefig("{0:s}.{1:s}".format(filename, form))
    else:
        for form in formats:
            fig.savefig("{0:s}.{1:s}".format(filename, form))


class PlotSims:
    def __init__(self, infile):
        self.infile = infile
        self.properties = self.get_properties()
        self.timestep = 1e-5  # ns
        self.combined_properties = []
        self.combined_properties_data = []

    def get_properties(self):
        return self.properties + self.combined_properties

    def _get_data(self):
        pass

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
            ax.plot(x_av, y_av, color=color, label=label, alpha=alpha)
            alpha = 0.4
            label = ""
        if timeseries:
            ax.plot(x, y, label=label, color=color, alpha=alpha)

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

    def _get_stats(self, x, y, blocks=10, bounds=None):
        blocksize = int(x.size / blocks)
        blocks = self.block_average(y, blocksize=blocksize)
        mean = np.mean(blocks)
        err = np.std(blocks)
        start, end = self._calc_drift(x, y)
        return [mean, err, end - start]

    def get_stats(self, props=True, blocks=10, bounds=None):
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
            tlist = [prop]
            tstats = self._get_stats(x, y, blocks=blocks, bounds=bounds)
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
        self, props, name, average=True, difference=False, absolute_values=False
    ):
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
            # axis=0 sums over columns, averaging the properties at each timestep
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


class PlotLAMMPS(PlotSims):
    def get_properties(self):
        properties = np.loadtxt(
            self.infile, comments=None, dtype=str, skiprows=1, max_rows=1
        )
        for i, prop in enumerate(properties):
            prop = prop.strip("c_")
            prop = prop.strip("v_")
            if "press[1]" in prop:
                prop = prop.split("[1]")[0] + "_xx"
            if "press[2]" in prop:
                prop = prop.split("[2]")[0] + "_yy"
            if "press[3]" in prop:
                prop = prop.split("[3]")[0] + "_zz"
            if "press[4]" in prop:
                prop = prop.split("[4]")[0] + "_xy"
            if "press[5]" in prop:
                prop = prop.split("[5]")[0] + "_xz"
            if "press[6]" in prop:
                prop = prop.split("[6]")[0] + "_yz"
            properties[i] = prop

        # Do not return the "#" and timestep at the beginning of line
        return list(properties[2:])

    def _get_unit(self, prop):
        return ""

    def _get_data(self, pos, bounds=None):

        len_props = len(self.properties) + 1
        if pos < len_props:
            data = np.loadtxt(self.infile, usecols=(0, pos))
        else:
            data = self.combined_properties_data[pos - len_props]

        if bounds is not None:
            # Calculate steps from nanoseconds
            bounds = [b / self.timestep for b in bounds]
            t0 = data[0, 0]
            dt = data[1, 0] - t0
            n0 = int((bounds[0] - t0) / dt)
            ne = int((bounds[1] - t0) / dt)
            length = len(data[:, 0])
            # Implicitly catch out of bounds error
            # Put out warnings in future.
            if ne >= length:
                ne = length - 1
                warnings.warn("Upper limit out of bounds", RuntimeWarning)
            if n0 < 0:
                n0 = 0
                warnings.warn("Lower limit out of bounds", RuntimeWarning)
            x = data[n0:ne, 0]
            y = data[n0:ne, 1]

        else:
            x = data[:, 0]
            y = data[:, 1]

        return x * self.timestep, y

    def _get_t0tE(self):
        time, _ = self._get_data(0, bounds=None)
        return time[0], time[-1]


class PlotGromacs(PlotSims):
    def __init__(self, infile="energy.xvg", statistics=None):
        self.infile = infile
        self.properties = self.get_properties()
        self.data = np.loadtxt(self.infile, comments=["@", "#"])

    def _get_unit(self, prop):
        if "Temperature" in prop:
            return "K"
        elif "Pressure" in prop:
            return "bar"
        else:
            return r"kJ$\,$mol$^{-1}$"

    def get_properties(self):
        properties = []
        with open(self.infile, "r") as f:
            for line in f:
                if not line[0] == "@" and not line[0] == "#":
                    break
                elif not line[0:3] == "@ s":
                    continue
                else:
                    prop = line.split("legend")[-1].strip()
                    prop = prop.strip('"')
                    properties.append(prop)
        return properties

    def _get_data(self, pos, bounds=None):
        if bounds is not None:
            bounds = [b * 1000 for b in bounds]

            t0 = self.data[0, 0]
            dt = self.data[1, 0] - t0
            n0 = int(((bounds[0]) - t0) / dt)
            ne = int(((bounds[1]) - t0) / dt)
            length = len(self.data[:, 0])
            # Implicitly catch out of bounds error
            if ne >= length:
                ne = length - 1
            x = self.data[n0:ne, 0]
            y = self.data[n0:ne, pos]

        else:
            x = self.data[:, 0]
            y = self.data[:, pos]

        return x / 1000, y


class ChunkData:
    def __init__(self, infile, trim_data=False):
        self.infile = infile
        self.trim_data = trim_data
        self.binsize_varies = False
        self.x, self.t, self.chunks, self.props = self._get_xtp()
        self.data = None
        self.tmin = None
        self.tmax = None
        self.xmin = None
        self.xmax = None
        self.dx = None

    def show_props(self):
        for item in self.props:
            print(item)

    def get_properties(self):
        # Compatablity
        return self.props

    def _get_xtp(self):
        # Looks like you need to write a parser
        t, x, chunks = [], [], []
        with open(self.infile, "r") as f:
            for i in range(3):
                line = f.readline()
            properties = [item.strip("\n") for item in line.split(" ")[3:]]
            line = f.readline()
            t0, chunk0, N0 = [float(item) for item in line.split(" ")]
            t.append(t0)
            chunk0 = int(chunk0)
            chunks.append(chunk0)

            for _ in range(chunk0):
                x.append(float(f.readline().split()[1]))

            while 0 < 1:
                try:
                    tline = f.readline().split()
                    t.append(float(tline[0]))
                    tchunk = int(tline[1])
                    chunks.append(tchunk)
                    for _ in range(tchunk):
                        next(f)
                except (IndexError, StopIteration):
                    break
                except:
                    raise RuntimeError
            x = np.array(x)
            t = np.array(t)
            chunks = np.array(chunks)

            self._chunksize_varies(chunks)
            if self.binsize_varies is True:
                x = np.arange(len(chunks))

            self.dx = x[1] - x[0]

        return x, t, chunks, properties

    def _chunksize_varies(self, chunks):
        # Checking if all the chunks o(ver time) are of the same size.
        # np.any returns true if any value is non-zero in array.
        self.binsize_varies = np.any(chunks - chunks[0] * np.ones(np.shape(chunks)))

    def _get_data(self, tmin, tmax):
        if (
            # Do not read again if we already read that timeframe
            tmin == self.tmin
            and tmax == self.tmax
            and isinstance(self.data, np.ndarray)
        ):
            return True

        ind_tmin = np.where(self.t == tmin)[0][0]
        ind_tmax = np.where(self.t == tmax)[0][0]
        tdiff = ind_tmax - ind_tmin
        usecols = [item for item in range(2, len(self.props) + 2)]
        if self.binsize_varies:
            data = np.zeros((tdiff, max(self.chunks), len(self.props)))
            pastchunks = 0
            for ti in range(tdiff):
                chunks = self.chunks[ti]
                data[ti, :chunks, :chunks] = np.loadtxt(
                    self.infile,
                    skiprows=pastchunks + 4,
                    usecols=usecols,
                    max_rows=chunks,
                )
                pastchunks += chunks + 1

        else:
            data = np.zeros((tdiff, len(self.x), len(self.props)))
            chunks = self.chunks[0]
            dchunks = chunks + 1
            for ti in range(tdiff):
                data[ti, :, :] = np.loadtxt(
                    self.infile,
                    skiprows=ti * dchunks + 4,
                    usecols=usecols,
                    max_rows=chunks,
                )
        if self.trim_data is True:
            if len(self.x) > len(data[0, :-1, 0]):
                self.x = self.x[:-1]
            self.dx = self.x[1] - self.x[0]
            self.xmin = self.x[0] - self.dx / 2.0
            self.xmax = self.x[-1] + self.dx / 2.0
            data = data[:, :-1, :]

        self.tmin = tmin
        self.tmax = tmax
        self.data = data

    @staticmethod
    def average_data(data, err=False):
        average = np.average(data, axis=0)
        if err is True:
            raise NotImplemented

        return average

    def combine_quantity(
        self, data, tmin=None, tmax=None, method="add", factor=None, average=True
    ):

        tmin, tmax = self._get_tminmax(tmin, tmax)
        self._get_data(tmin, tmax)

        if method == "add":
            combined_quantity = np.zeros(
                (len(self.data[:, 0, 0]), len(self.data[0, :, 0]))
            )
        if method == "mul":
            combined_quantity = np.ones(
                (len(self.data[:, 0, 0]), len(self.data[0, :, 0]))
            )
        else:
            combined_quantity = copy.deepcopy(self.get_data(data[0], tmin, tmax))

            data = data[1:]

        method_dict = {
            "add": np.add,
            "sub": np.subtract,
            "div": np.divide,
            "mul": np.multiply,
        }
        for item in data:
            ydata = self.get_data(item, tmin, tmax)
            method_dict.get(method)(combined_quantity, ydata, out=combined_quantity)

        if average is True:
            combined_quantity = self.average_data(combined_quantity)

        if factor is not None:
            combined_quantity *= float(factor)

        return combined_quantity

    def _get_pos(self, prop):
        return self.props.index(prop)

    def _get_tminmax(self, tmin, tmax):
        if tmin is not None or tmax is not None:
            if tmin is not None and tmax is None:
                tmax = self.t[-1]
            if tmin is None and tmax is not None:
                tmin = self.t[0]
        else:
            tmin, tmax = self.t[0], self.t[-1]
        return tmin, tmax

    def plot_profile(self, ax, data, tmin=None, tmax=None, color=None):

        tmin, tmax = self._get_tminmax(tmin, tmax)

        ydata = self.get_data(data, tmin, tmax)
        if np.shape(ydata) != np.shape(self.x):
            ydata = self.average_data(ydata)

        ax.plot(self.x, ydata, color=color, lw=2)

    def get_data(self, data, tmin, tmax):
        if isinstance(data, str):
            self._get_data(tmin, tmax)  # This sets self.data
            index = self._get_pos(data)
            ydata = self.data[:, :, index]
        elif isinstance(data, np.ndarray):
            ydata = data
        else:
            raise TypeError("Data must either be at sring or numpy array")

        return ydata
        # return copy.deepcopy(ydata)

    def get_stats(self, data, x0, xE, tmin=None, tmax=None):
        tmin, tmax = self._get_tminmax(tmin, tmax)
        ydata = self.get_data(data, tmin, tmax)
        discretisation = 1.0 / len(self.x)
        bound_cross = False
        if xE < 0:
            xE = 1 + xE
            bound_cross = True

        index0 = int(np.round(x0 / discretisation))
        indexE = int(np.round(xE / discretisation))
        if bound_cross:
            ydata = np.concatenate((ydata[:, :index0], ydata[:, indexE:]), axis=1)
        else:
            ydata = ydata[:, index0:indexE]
        mean, error = self._get_stats(ydata)
        return mean, error

    def block_average(self, x, blocksize=20):
        Nobs = len(x)
        blocks = int(Nobs / blocksize)
        cutout = Nobs - (blocks * blocksize)
        # As of now just discard the first few samples if they do not
        # fit in a nice array
        x = np.reshape(x[cutout:], (blocks, blocksize, 1))
        return np.mean(x, axis=1)

    def _get_stats(self, ydata, blocks=10):
        times = np.shape(ydata)[0]
        blocksize = int(np.ceil(times / blocks))
        x = np.average(ydata, axis=1)
        blocks = self.block_average(x, blocksize=blocksize)
        mean = np.mean(blocks)
        err = np.std(blocks)
        return [mean, err]
