from .plot_sims import PlotSims
import numpy as np
import warnings

class PlotGromacs(PlotSims):
    def __init__(self, infile="energy.xvg", statistics=None):
        self.timestep = 1e-5  # ns
        self.infile = infile
        self.properties = self.get_properties()
        self.data = np.loadtxt(self.infile, comments=["@", "#"])
        self.combined_properties = []
        self.combined_properties_data = []

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
            print(x,y)
        return x / 1000., y
