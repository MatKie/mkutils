import numpy as np
from mkutils.plotting import PlotLAMMPS, ChunkData, save_to_file
from mkutils.plot_functions import create_fig
import os


def osm_press_ideal(rho_av, m_av, rho_err, m_err, T=298.15, nu=2):
    R = 8.314  # J/mol/K
    pi = R * T * rho_av * m_av * nu / 1e5
    rho_err = R * T * m_av * nu / 1e5 * rho_err
    m_err = R * T * rho_av * nu / 1e5 * m_err
    err = rho_err + m_err
    return pi, err


def osm_coeff(osmPideal_av, osmPideal_err, osmP_av, osmP_err):
    rel_err_ideal = osmPideal_err / osmPideal_av
    rel_err_real = osmP_err / osmP_av
    osmC_av = osmP_av / osmPideal_av
    osmC_err = osmC_av * (rel_err_ideal + rel_err_real)
    return osmC_av, osmC_err


def force_to_bar(force, wall_lenght):
    pressure = force / (wall_lenght ** 2) * (4.184 / 6.022) * 1e10
    pressure /= 1e5
    return pressure


def write_results(
    averages, errors, path=".",
):
    props = [
        "Molality/mol/kg",
        "osmP/bar",
        "osmC/-",
        "rho/kg/m3",
        "rhosolv/kg/m3",
        "dp_LRC",
        "osmP_LRC",
        "osmC_LRC",
        "osmPid/bar",
    ]
    with open(path, "a+") as f:
        header = "#"
        for item in props:
            header = "".join(
                [
                    header,
                    "{:<16s}{:<12s}".format(
                        item, "_".join(["err", item.split("/")[0]])
                    ),
                ]
            )
        header = "".join([header, "\n"])
        f.write(header)
        line = " "
        for av, err in zip(averages, errors):
            line = "".join([line, "{:<16.4f}{:<12.4f}".format(av, err)])
        f.write("".join([line, "\n"]))


def eval_wall(Wall, Wall0, xy, bounds, plot=False, path="."):
    Wall.create_combined_property(
        ["f_zwall1[1]", "f_zwall2[1]"], "WallForce", absolute_values=True
    )
    _, av_wall, err_wall, drift_wall = Wall.get_stats(props="WallForce", bounds=bounds)
    av_osmP = force_to_bar(av_wall, xy)
    err_osmP = force_to_bar(err_wall, xy)

    if plot is True:
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        Wall0.plot_timeseries(ax, "WallForce", alpha=0.5, blocksize=100)
        Wall0.timesep = 0.005
        Wall0.plot_mean(ax, "WallForce", bounds=(1, 20), err=True, color="grey")
        Wall0.plot_mean(ax, "WallForce", bounds=(20, 40), err=True, color="grey")
        Wall0.plot_mean(ax, "WallForce", bounds=(40, 60), err=True, color="grey")
        Wall0.plot_mean(ax, "WallForce", bounds=(60, 80), err=True, color="grey")
        Wall0.plot_mean(ax, "WallForce", bounds=(80, 100), err=True, color="grey")
        Wall0.plot_mean(ax, "WallForce", bounds=(100, 120), err=True, color="grey")
        ax.set_xlim(0, 120)
        save_to_file(os.path.join(path, "WallForce"))
    return av_osmP, err_osmP


def eval_densities(
    Solute,
    Solvent,
    Solute0,
    tmin,
    tmax,
    zi_h,
    zi_l,
    zo_h,
    zo_l,
    nu,
    MW_solute,
    MW_solvent,
    plot=False,
    path=".",
):
    solute_mol = Solute.combine_quantity(
        ["density/number"], tmin=tmin, tmax=tmax, average=False, factor=1000
    )
    solvent_mol = Solvent.combine_quantity(
        ["density/number"], tmin=tmin, tmax=tmax, average=False, factor=1000
    )

    # Conversion from number/nm3 to kg/m3
    solute_factor = float(1 / nu) * MW_solute * (1.0 / 6.022) * 10 ** 4  # kg/m^3
    solvent_factor = MW_solvent * (1.0 / 6.022) * 10 ** 4  # kg/m^3

    # We multiply to get the mass specific densities
    solute_data = solute_mol * solute_factor
    solvent_data = solvent_mol * solvent_factor
    density = np.add(solute_data, solvent_data)

    # rho_solute/(rho_solvent*MW_solute) = n_solute/m_solvent = molality
    molality = np.divide(solute_data, solvent_data)
    molality /= MW_solute

    # Calculate all the averages
    moldens_solute, moldens_solute_err = Solute0.get_stats(
        solute_mol, zi_l, zi_h, tmin=tmin, tmax=tmax
    )
    moldens_solv_in, moldens_solv_in_err = Solute0.get_stats(
        solvent_mol, zi_l, zi_h, tmin=tmin, tmax=tmax
    )
    moldens_solv_o, moldens_solv_o_err = Solute0.get_stats(
        solvent_mol, zo_l, zo_h, tmin=tmin, tmax=tmax
    )

    av_mol, err_mol = Solute0.get_stats(molality, zi_l, zi_h, tmin=tmin, tmax=tmax)
    av_solv = moldens_solv_o * solvent_factor
    err_solv = moldens_solv_o_err * solvent_factor
    av_solute = moldens_solute * solute_factor
    err_solute = moldens_solute_err * solute_factor

    # Smaller error with direct calculation
    av_brine, err_brine = Solute0.get_stats(density, zi_l, zi_h, tmin=tmin, tmax=tmax)

    if plot is True:
        fig, ax = create_fig(1, 1)
        ax = ax[0]
        Solute0.plot_profile(ax, density)
        Solute0.plot_profile(ax, solute_data)
        Solute0.plot_profile(ax, solvent_data)
        xl = 0.25 * (Solute0.xmax - Solute0.xmin) + Solute0.xmin
        xh = 0.75 * (Solute0.xmax - Solute0.xmin) + Solute0.xmin
        # print(xl, xh, x[1]-x[0])
        ax.axvline(xl, color="k", ls="--", lw=2)
        ax.axvline(xh, color="k", ls="--", lw=2)

        ax.legend(["Density", "Solute Density", "Solvent Density"])
        save_to_file(os.path.join(path, "Density"))

    return (
        av_mol,
        err_mol,
        av_brine,
        err_brine,
        av_solv,
        err_solv,
        av_solute,
        err_solute,
        moldens_solute,
        moldens_solute_err,
        moldens_solv_in,
        moldens_solv_in_err,
        moldens_solv_o,
        moldens_solv_o_err,
    )


def eval_LRC(Mix, moldens_solv_in, moldens_solv_o, moldens_solute):
    composition_out = [1, 0, 0]
    moldens = moldens_solv_in + moldens_solute
    xw = moldens_solv_in / moldens
    composition_in = [xw, (1 - xw) / 2.0, (1 - xw) / 2.0]
    po = Mix.LRC_p(moldens, composition_in)
    pi = Mix.LRC_p(moldens_solv_o, composition_out)
    dp = po - pi
    return dp
