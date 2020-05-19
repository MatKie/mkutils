from .plot_sims import PlotSims
import numpy as np
import warnings
import copy


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
            data = data[:, :-1, :]
        self.dx = self.x[1] - self.x[0]
        self.xmin = self.x[0] - self.dx / 2.0
        self.xmax = self.x[-1] + self.dx / 2.0

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
        self, data, bounds=(None, None), method="add", factor=None, average=True
    ):
        tmin, tmax = *bounds
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
            # For division and subtraction initialis with first element
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
