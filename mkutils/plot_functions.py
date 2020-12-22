import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import sys
from scipy.interpolate import interp1d
import random

# For changing matplotlib default settings consult:
# https://matplotlib.org/1.3.1/users/customizing.html#matplotlibrc-sample

# Matplotlib figure documentation:
# http://matplotlib.org/api/figure_api.html?highlight=figure#module-matplotlib.figure

# create_fig() only supports 1 subplot
def create_fig(
    sp_row,
    sp_col,
    fig_width=8,
    fig_height=6,
    axes_label_size=24,
    axes_tick_size=20,
    legend_fontsize=16,
    marker_size=6,
    line_width=1,
    ticks_top_right=True,
    sharex=False,
    sharey=False,
):

    # ------------------------------ USER INPUT ------------------------------
    # Specify font
    font_style = "serif"

    # Specify legend
    legend_numpoint = 1
    legend_frameon = False

    # Specify marker and line size

    # Tick direction
    tick_direc = "out"

    # Tick size
    major_tick_size = 4
    minor_tick_size = 2
    major_tick_width = 1.1
    minor_tick_width = 1.1

    # Figure properties
    fig_size = (fig_width, fig_height)  # [width, height] in inches
    fig_dpi = 100  # Increase later
    fig_facecolor = "white"
    fig_edgecolor = "black"
    fig_edgelinewidth = 0
    fig_frameon = False
    fig_tightlayout = True
    fig_alpha = 1.0  # Transparency

    # Subplot properties
    splot_xscale = "linear"
    splot_yscale = "linear"
    splot_facecolor = "white"
    tick_direct = "out"
    border_linewidth = 1.1

    # ------------------------------ CODE ------------------------------
    # Set subscript size
    # mpl.mathtext.SHRINK_FACTOR = 0.7
    # mpl.mathtext.GROW_FACTOR = 1.0 / 0.7

    # Use text
    # mpl.rcParams['text.usetex']=True

    # Set marker and line size
    mpl.rcParams["lines.linewidth"] = line_width
    mpl.rcParams["lines.markersize"] = marker_size

    # Set font
    mpl.rcParams["font.family"] = font_style
    mpl.rcParams["axes.labelsize"] = axes_label_size
    mpl.rcParams["xtick.labelsize"] = axes_tick_size
    mpl.rcParams["ytick.labelsize"] = axes_tick_size

    # Set legend properties
    mpl.rcParams["legend.fontsize"] = legend_fontsize
    mpl.rcParams["legend.numpoints"] = legend_numpoint
    mpl.rcParams["legend.frameon"] = legend_frameon

    # Set tick direction
    mpl.rcParams["xtick.direction"] = tick_direc
    mpl.rcParams["ytick.direction"] = tick_direc

    # Set tick size
    mpl.rcParams["xtick.major.size"] = major_tick_size
    mpl.rcParams["ytick.major.size"] = major_tick_size
    mpl.rcParams["xtick.minor.size"] = minor_tick_size
    mpl.rcParams["ytick.minor.size"] = minor_tick_size
    mpl.rcParams["xtick.major.width"] = major_tick_width
    mpl.rcParams["ytick.major.width"] = major_tick_width
    mpl.rcParams["xtick.minor.width"] = minor_tick_width
    mpl.rcParams["ytick.minor.width"] = minor_tick_width

    # Set axes border line thickness
    mpl.rcParams["axes.linewidth"] = border_linewidth

    # Create and format figure
    fig1 = plt.figure(
        num=None,
        figsize=fig_size,
        dpi=fig_dpi,
        facecolor=fig_facecolor,
        edgecolor=fig_edgecolor,
        linewidth=fig_edgelinewidth,
        frameon=fig_frameon,
        tight_layout=fig_tightlayout,
    )
    fig1.patch.set_alpha(fig_alpha)

    # Define array of subplots
    ax = []

    # Loop over all subplots
    for i in range(sp_row * sp_col):
        # Create and format subplot
        newax = fig1.add_subplot(
            sp_row, sp_col, i + 1, xscale=splot_xscale, yscale=splot_yscale
        )

        # ,label=str(random.randint(1,10001))
        ax.append(newax)
        ax[i].patch.set_facecolor(splot_facecolor)

        # Turn off ticks right and top
        if not ticks_top_right:
            ax[i].yaxis.tick_left()
            ax[i].xaxis.tick_bottom()

    return fig1, ax


# Set major and minor tick marks on the x and y axis.
# x_major and x_minor represent the invervals
# Good options for tick formatter are NullFormatter(),
# ScalarFormatter() or StrMethodFormatter
def set_ticks(x_major, x_minor, y_major, y_minor, x_major_format, y_major_format, ax):
    xmajorLocator = MultipleLocator(x_major)
    xminorLocator = MultipleLocator(x_minor)
    ymajorLocator = MultipleLocator(y_major)
    yminorLocator = MultipleLocator(y_minor)
    xmajorFormatter = FormatStrFormatter(x_major_format)
    ymajorFormatter = FormatStrFormatter(y_major_format)
    return (
        ax.xaxis.set_major_locator(xmajorLocator),
        ax.xaxis.set_minor_locator(xminorLocator),
        ax.yaxis.set_major_locator(ymajorLocator),
        ax.yaxis.set_minor_locator(yminorLocator),
        ax.xaxis.set_major_formatter(xmajorFormatter),
        ax.yaxis.set_major_formatter(ymajorFormatter),
    )


def create_legend(legendon, legend_loc, ax):
    if legendon == True:
        return ax.legend(loc=legend_loc)
    else:
        return 0


def get_color(i):
    color_list = [
        "#000000",
        "#4477AA",
        "#66CCEE",
        "#228833",
        "#CCBB44",
        "#EE6677",
        "#AA3377",
        "#BBBBBB",
    ]

    if i > (len(colorlist) - 0.5):
        raise ValueError("Index out of range")

    return color_list[i]


def get_marker(i):
    marker_list = ["o", "^", "s", "D"]

    if i > (len(colorlist) - 0.5):
        raise ValueError("Index out of range")

    return marker_list[i]


# Label line with line2D label data
def labelLine(ax, line_num, xy, label, y_adjust):
    xdata = ax.lines[line_num].get_xdata()
    ydata = ax.lines[line_num].get_ydata()

    # Find coordinates from the approximate xy coordinates specified
    index = -1
    tol = 0.10
    for i in range(len(xdata)):
        error = abs((xdata[i] - xy[0]) / xy[0]) + abs((ydata[i] - xy[1]) / xy[1])
        if error < tol:
            index = i
            break

    if index == -1:
        print("No x,y coordinates found within set tolerance")
        return 0

    xy = [xdata[index], ydata[index] + y_adjust]

    # Compute the slope
    dx = xdata[index + 1] - xdata[index - 1]
    dy = ydata[index + 1] - ydata[index - 1]

    ang = np.arctan(dy / dx) * 180.0 / np.pi

    # Transform to screen co-ordinates
    pt = np.array([xy[0], xy[1]]).reshape((1, 2))
    trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

    return ax.text(
        xy[0],
        xy[1],
        label,
        rotation=trans_angle,
        fontsize=10,
        ha="center",
        va="center",
        zorder=2.5,
        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"),
    )


# Spline data to smoothen
def spline(x, y, order, x_points):
    f_new = interp1d(x, y, kind=order)
    x_new = np.linspace(x[0], x[-1], x_points)
    y_new = f_new(x_new)
    return x_new, y_new


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
