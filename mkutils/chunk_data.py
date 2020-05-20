import numpy as np
import copy


class ChunkData:
    def __init__(self, infile, trim_data=False):
        """
        Class designed to evaluate spatial profiles in LAMMPS data.

        Parameters
        ----------
        infile : str
            path to the file containing the LAMMPS profile data
        trim_data : bool, optional
            Sometimes the last bin in a profile file is to big for the 
            box -- this option excluded the last bin from evaluation, by
            default False.
        """
        self.infile = infile
        self.trim_data = trim_data
        self.binsize_varies = False
        self.x, self.t_full, self.chunks, self.props = self._get_xtp()
        self.data = None
        self.t = [0]
        self.xmin = None
        self.xmax = None
        self.dx = None
        self.combined_properties = []
        self.combined_properties_data = []

    def get_properties(self):
        """
        See which properties are in the profile in

        Returns
        -------
        list of str
            list of names of properties
        """
        return self.props

    def _get_xtp(self):
        """
        Parsing lammps profile files for times, x-values and number of 
        bins at each timestep. 
        """
        t, x, chunks = [], [], []
        with open(self.infile, "r") as f:
            for i in range(3):
                line = f.readline()
            properties = [item.strip("\n") for item in line.split(" ")[3:]]
            line = f.readline()
            # Timestep, Number of bins, Number of particles
            t0, chunk0, N0 = [float(item) for item in line.split(" ")]
            t.append(t0)
            chunk0 = int(chunk0)
            chunks.append(chunk0)

            # Read x-valueonly for first timestep -- no ideal.
            for _ in range(chunk0):
                x.append(float(f.readline().split()[1]))

            # Certainly now how one should do that but it works
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

            # Check if number of chunks varies, if so replace x by a vector
            # of length chunks. Doesn't really make sense here, but
            # binsize_varies is an important attribute in _get_data, so
            # there might be a reason for it..
            self._chunksize_varies(chunks)
            if self.binsize_varies is True:
                x = np.arange(len(chunks))

            self.dx = x[1] - x[0]

        return x, t, chunks, properties

    def _chunksize_varies(self, chunks):
        # Checking if all the frames have the same number of chunks.
        # np.any returns true if any value is non-zero in array.
        self.binsize_varies = np.any(chunks - chunks[0] * np.ones(np.shape(chunks)))

    def _set_data(self, bounds):
        tmin, tmax = self._get_tminmax(*bounds)
        if (
            # Do not read again if we already read that timeframe
            tmin == self.t[0]
            and tmax == self.t[-1]
            and isinstance(self.data, np.ndarray)
        ):
            return True

        ind_tmin = np.where(self.t_full == tmin)[0][0]
        ind_tmax = np.where(self.t_full == tmax)[0][0]
        self.t = self.t_full[ind_tmin:ind_tmax]  #  Excluding last one
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
            data = np.zeros((tdiff, self.chunks[0], len(self.props)))
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
        # we are dealing with bins. We do this to get the outer edges.
        self.xmin = self.x[0] - self.dx / 2.0
        self.xmax = self.x[-1] + self.dx / 2.0
        self.data = data

    @staticmethod
    def average_data(data, err=False):
        average = np.average(data, axis=0)
        if err is True:
            raise NotImplementedError

        return average

    def create_combined_property(
        self,
        data,
        name,
        bounds=(None, None),
        method="add",
        factor=None,
        average=True,
        *args,
        **kwargs
    ):
        self._set_data(bounds)

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
            combined_quantity = copy.deepcopy(self._get_data(data[0], bounds)[-1])
            data = data[1:]

        method_dict = {
            "add": np.add,
            "sub": np.subtract,
            "div": np.divide,
            "mul": np.multiply,
        }
        for item in data:
            _, ydata = self._get_data(item, bounds)
            method_dict.get(method)(combined_quantity, ydata, out=combined_quantity)

        if average is True:
            combined_quantity = self.average_data(combined_quantity)

        if factor is not None:
            combined_quantity *= float(factor)

        self.combined_properties.append(name)
        self.combined_properties_data.append(combined_quantity)

    def _get_pos(self, prop):
        if prop in self.props:
            return self.props.index(prop)
        if prop in self.combined_properties:
            return self.combined_properties.index(prop) + len(self.props)
        raise RuntimeError('Property "{:s}" not found'.format(prop))

    def _get_tminmax(self, tmin, tmax):
        if tmin is not None or tmax is not None:
            if tmin is not None and tmax is None:
                tmax = self.t_full[-1]
                if tmin < self.t_full[0]:
                    tmin = self.t_full[0]
            if tmin is None and tmax is not None:
                tmin = self.t_full[0]
                if tmax > self.t_full[-1]:
                    tmax = self.t_full[-1]
        else:
            tmin, tmax = self.t_full[0], self.t_full[-1]

        return tmin, tmax

    def plot_profile(self, ax, data, bounds=(None, None), color=None):

        t, ydata = self._get_data(data, bounds)
        if np.shape(ydata) != np.shape(self.x):
            ydata = self.average_data(ydata)

        ax.plot(self.x, ydata, color=color, lw=2)

    def get_data(self, data, bounds=(None, None)):
        """
        Public method to retrieve t, y data from Chunks to feed to other
        ChunkData instances. For example to use create_combined_property
        to combine properties from different files.

        Returns:
        --------
            t (array-like): vector with time-step information
            ydata (array-like): matrix of shape (n_times, n_x)
        """
        return self._get_data(data, bounds=bounds)

    def _get_data(self, data, bounds):
        if isinstance(data, str):
            index = self._get_pos(data)
            if index >= len(self.props):
                index -= len(self.props)
                ydata = self.combined_properties_data[index]
            else:
                self._set_data(bounds)  # This sets self.data
                data = self.data
                ydata = data[:, :, index]
            t = self.t
        elif isinstance(data, tuple) and isinstance(data[1], np.ndarray):
            t, ydata = data
        elif isinstance(data, np.ndarray):
            ydata = data
            t = np.arange(len(data[0]))
        else:
            raise TypeError("Data must either be at sring or numpy array")

        return t, ydata
        # return copy.deepcopy(ydata)

    def get_stats(self, data, xbounds=(0, 1), bounds=(None, None)):
        x0, xE = xbounds
        t, ydata = self._get_data(data, bounds)
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
