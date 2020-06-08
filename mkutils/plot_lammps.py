from .plot_sims import PlotSims
import numpy as np
import warnings


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

    def get_data(self, prop, bounds=None):
        pos = self._get_pos(prop)
        return self._get_data(pos, bounds=bounds)

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
