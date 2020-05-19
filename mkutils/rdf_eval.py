import numpy as np
import copy


class RDFEval:
    """
    Class designed to calculate solvation numbers
    First peak distances etc. from rdf's.
    """

    def __init__(self, rdfobj, slack=2.0):
        self.obj = rdfobj  # MDAnalysis object
        self.bins = rdfobj.bins  # bin middles
        self.rdf = rdfobj.rdf  #  rdf values
        self.max0 = None  # index of solvation shell radii
        self.min0 = None
        self.max1 = None
        self.min1 = None
        self.n0 = None  # coordination number
        self.n1 = None
        self.kbi = None
        self.rho = self._calc_rho()
        self.slack = slack  # Angstrom for LAMMPS

    def _calc_rho(self):
        """
        Calculates density for npt/nvt simulation over trajectory.
        Only the used frames are considered and only triclinic boxes
        We count the number of residues of the second species, i.e.
        the "target" of the rdf
        """
        uni = self.obj.u
        start = self.obj.start
        stop = self.obj.stop
        atoms = self.obj.g2.n_residues
        if atoms == 1:
            atoms = self.obj.g2.n_atoms

        # Averaging volume
        V = 0.0
        for frame in uni.trajectory[start:stop]:
            V += frame.dimensions[0] * frame.dimensions[0] * frame.dimensions[0]
        V /= stop - start
        return atoms / V

    def _get_r0r1(self):
        """
        Get the maximum, minimum values for looking for max/min values.
        A bit crude as there is no smoothing involved, therefore can fall
        for noisy data
        """
        # first peak should be the highest values
        max0 = np.amax(self.rdf)
        (ind0,) = np.where(self.rdf == max0)
        self.max0 = ind0[0]

        slack = np.int(self.slack / (self.bins[2] - self.bins[1]))

        # Look for appr. next valley by checking when the rdf increases
        # again. Can fail for noisy data
        i = 1
        while self.rdf[ind0[0] + i + 1] < self.rdf[ind0[0] + i]:
            i += 1

        # search for minimal value between the found minima and a slack region
        min0 = np.amin(self.rdf[ind0[0] : ind0[0] + i + slack])
        (indv0,) = np.where(self.rdf == min0)
        self.min0 = indv0[0]

        # Look for increase in similar fashion as for the minima
        i = 1
        while self.rdf[indv0[0] + i + 1] > self.rdf[indv0[0] + i]:
            i += 1

        max1 = np.amax(self.rdf[indv0[0] : indv0[0] + i + slack])
        (ind1,) = np.where(self.rdf == max1)
        self.max1 = ind1[0]

        i = 1
        while self.rdf[ind1[0] + i + 1] < self.rdf[ind1[0] + i]:
            i += 1

        min1 = np.amin(self.rdf[ind1[0] : ind1[0] + i + slack])
        (indv1,) = np.where(self.rdf == min1)
        self.min1 = indv1[0]

    def get_n0n1(self):
        """
        Collects positions of minimia/maxima and integrates for the 
        first two coordination numbers:

        coord = 4*pi*rho int_rmin^rmax g(r)*r^2 dr
        """
        self._get_r0r1()
        self.n0 = self._integrate(0, self.min0)
        self.n1 = self._integrate(self.min0, self.min1)

    def get_kbi(self):
        x2 = np.multiply(self.bins, self.bins)
        rdfm1 = copy.deepcopy(self.rdf)
        rdfm1 -= 1
        rdfm1x2 = np.multiply(rdfm1, x2)

        integral = np.trapz(rdfm1x2, self.bins)

        fac = 4.0 * np.pi
        rho = self.rho
        self.kbi = fac * rho * integral
        return self.kbi

    def _integrate(self, lower_bound, upper_bound):
        x2 = np.multiply(
            self.bins[lower_bound : upper_bound + 1],
            self.bins[lower_bound : upper_bound + 1],
        )
        y = np.multiply(self.rdf[lower_bound : upper_bound + 1], x2)
        # gr =  np.trapz(y,
        #                self.bins[lower_bound:upper_bound+1])
        dx = self.bins[lower_bound + 1] - self.bins[lower_bound]
        y = y * dx
        gr = np.sum(y)
        # print(self.bins[lower_bound:upper_bound+1])
        fac = 4.0 * np.pi
        rho = self.rho

        return fac * rho * gr

    def print_results(self, second_coord=False):
        """
        Print results for all minima/maxima coordinatin numbers.

        Calculates everything if necessary.
        """
        if None in [self.min0, self.min1, self.max0, self.max1, self.n0, self.n1]:
            self.get_n0n1()
        print("Rmax-1st:   {:.4f} nm".format(self.bins[self.max0] / 10.0))
        print("Rmin-1st:   {:.4f} nm".format(self.bins[self.min0] / 10.0))
        print("Coord#-1st: {:.4f}\n".format(self.n0))
        if second_coord:
            print("Rmax-2nd:   {:.4f} nm".format(self.bins[self.max1] / 10.0))
            print("Rmin-2nd:   {:.4f} nm".format(self.bins[self.min1] / 10.0))
            print("Coord#-2nd: {:.4f}\n".format(self.n1))

    def plot(self, ax, label="", color="k", lw=1, ls="-", alpha=1):
        """
        Plot with graphical representation of integral
        """
        ax.plot(
            self.bins, self.rdf, lw=lw, ls=ls, color=color, label=label, alpha=alpha
        )
        ylim = ax.get_ylim()
        ax.vlines(self.bins[self.max0], *ylim, linestyle="--", alpha=alpha)
        ax.vlines(self.bins[self.min0], *ylim, linestyle="--", alpha=alpha)
        ax.vlines(self.bins[self.max1], *ylim, linestyle="--", alpha=alpha)
        ax.vlines(self.bins[self.min1], *ylim, linestyle="--", alpha=alpha)
        ax.fill_between(
            self.bins[: self.min0 + 1],
            self.rdf[: self.min0 + 1],
            np.zeros(self.rdf[: self.min0 + 1].shape),
            color=color,
            alpha=0.6 * alpha,
        )
        ax.fill_between(
            self.bins[self.min0 : self.min1 + 1],
            self.rdf[self.min0 : self.min1 + 1],
            np.zeros(self.rdf[self.min0 : self.min1 + 1].shape),
            color=color,
            alpha=0.4 * alpha,
        )

        ax.set_ylabel("g(r) / -")
        ax.set_xlabel("r / $\AA$")
        ax.set_ylim(*ylim)
