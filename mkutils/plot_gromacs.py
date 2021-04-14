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
        elif "Pressure" in prop or "Pres-" in prop:
            return "bar"
        elif "#Surf*SurfTen" in prop:
            return r"bar$\,$nm"
        elif "Box" in prop:
            return "nm"
        elif "Volume" in prop:
            return r"nm^3"
        elif "Density" in prop:
            return r"kg$\,$mol$^{-3}$"
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
        return x / 1000.0, y

    @staticmethod
    def _parse_line(line):
        sep = "  "
        variables = [""] * 6  # six variables to hold
        var = 0
        for i, char in enumerate(line):
            if line[i : i + len(sep)] != sep:
                variables[var] += char
                write = True
            elif line[i : i + len(sep)] == sep and write == True:
                if var > 3.5:
                    sep = "  "  # first and last variable eventually have spaces
                else:
                    sep = " "
                var += 1
                write = False
            else:
                pass

        variables[1:-1] = [float(variable) for variable in variables[1:-1]]
        data = {
            variables[0]: {
                "Average": variables[1],
                "Error": variables[2],
                "RMSD": variables[3],
                "Drift": variables[4],
                "Unit": variables[5].strip(")").strip("("),
            }
        }
        return data

    @staticmethod
    def get_gmx_stats(gmx_file):
        data = {}
        with open(gmx_file, "r") as f:
            switch = False
            for line in f:
                if line[:10] == "-" * 10 and not switch:
                    switch = True
                elif switch:
                    data.update(PlotGromacs._parse_line(line))
                else:
                    pass
        return data
