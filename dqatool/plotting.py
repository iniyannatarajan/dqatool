from casatools import table
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.time import Time
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file
from bokeh.layouts import gridplot
from dqatool.constants import SECONDS_IN_DAY, XX_CORRID, YY_CORRID, DEFAULT_MARKER_SIZE, DEFAULT_ALPHA, HZ_IN_GHZ, SPEED_OF_LIGHT
from typing import Literal, List
from dqatool.logging_config import get_logger

tb = table()

logger = get_logger(__name__)

# Set Seaborn style for all plots
sns.set_theme(style="whitegrid")

def plot_amp_vs_time(ms_path: str, colname: Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA'] = 'DATA', 
                        corrs: List[Literal[0, 1, 2, 3]] = [0], in_notebook: bool = False, save_path: str = 'amp_vs_time.html') -> None:
    """
    Plot visibility amplitudes vs. time as scatter points for the specified correlations.

    This function visualizes the amplitude of visibility data over time for the given 
    correlations in a CASA MeasurementSet (MS). The data extracted from the specified 
    column(s) are plotted using Bokeh. Each correlation is displayed in a separate subplot.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MeasurementSet (MS).
    colname : Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA']
        Column name to extract visibility data from.
    corrs : List[Literal[0, 1, 2, 3]]
        List of correlation indices to plot (e.g., 0 for XX/RR, 1 for XY/RL, etc.).
    save_path : str, optional
        Path to save the generated plot as an image file.

    Raises
    ------
    ValueError
        If any of the specified correlation indices are invalid.
    """
    # Decide how to render the plot
    if in_notebook:
        output_notebook()
    else:
        logger.info(f"Saving plot to {save_path}")
        output_file(save_path)

    # Load data from MS
    tb.open(ms_path)
    data = tb.getcol(colname)    # shape: (nRow, nChan, nCorr)
    flags = tb.getcol("FLAG")   # same shape
    times = tb.getcol("TIME")   # shape: (nRow,)
    tb.close()
    data = np.transpose(data, axes=(2, 1, 0))
    flags = np.transpose(flags, axes=(2, 1, 0))

    # Convert times to datetime
    times_dt = Time(times / SECONDS_IN_DAY, format='mjd').to_datetime()

    # Load polarization labels from POLARIZATION table
    tb.open(f"{ms_path}::POLARIZATION")
    corr_types = tb.getcol("CORR_TYPE")[:, 0]  # Get correlation types (e.g., [5, 6, 7, 8])
    tb.close()

    # Map correlation types to labels
    corr_labels = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
    labels = [corr_labels.get(corr, f"Corr {corr}") for corr in corr_types]

    # Define colors for each correlation
    colors = ["blue", "red", "green", "orange"]

    # Create subplots for each correlation
    plots = []
    for i, corr in enumerate(corrs):
        if corr < 0 or corr >= data.shape[2]:
            raise ValueError(f"Invalid correlation index: {corr}")

        # Apply flags (masks) to data
        amp = np.abs(data[:, :, corr])
        amp = np.where(flags[:, :, corr], np.nan, amp)

        # Get rid of NaNs
        nchan = amp.shape[1]
        xs = np.repeat(times_dt, nchan)
        ys = amp.flatten()
        mask = ~np.isnan(ys)
        xs, ys = xs[mask], ys[mask]

        # Create a Bokeh figure for this correlation
        label = labels[corr] if corr < len(labels) else f"Corr {corr}"
        color = colors[i % len(colors)]  # Cycle through colors if more correlations are provided
        p = figure(
            title=f"Visibility Amplitude vs. Time ({label})",
            x_axis_type="datetime",
            x_axis_label="Time (UTC)",
            y_axis_label="Amplitude",
            width=450,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            output_backend="webgl"
        )
        p.scatter(xs, ys, marker="circle", size=DEFAULT_MARKER_SIZE, color=color, alpha=DEFAULT_ALPHA)
        plots.append(p)

    # Arrange plots in a 2x2 grid
    grid = gridplot([plots[i:i + 2] for i in range(0, len(plots), 2)])

    # Show the grid of plots
    show(grid)

def plot_amp_vs_freq(ms_path: str, colname: Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA'] = 'DATA', 
                     corrs: List[Literal[0, 1, 2, 3]] = [0], in_notebook: bool = False, save_path: str = 'amp_vs_time.html') -> None:
    """
    Plot visibility amplitudes vs. frequency, one subplot per correlation,
    arranged in a 2×2 grid.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MS.
    colname : Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA']
        Column name to extract visibility data from.
    corrs : List[Literal[0, 1, 2, 3]]
        List of correlation indices to plot (0=XX/RR, 1=XY/RL, 2=YX/LR, 3=YY/LL).

    Raises
    ------
    ValueError
        If any correlation index is invalid or more than four are requested.
    """
    # Decide how to render the plot
    if in_notebook:
        output_notebook()
    else:
        logger.info(f"Saving plot to {save_path}")
        output_file(save_path)

    # Load visibilities and flags
    tb.open(ms_path)
    data = tb.getcol(colname)    # (nRow, nChan, nCorr)
    flags = tb.getcol("FLAG")    # same shape
    tb.close()
    data = np.transpose(data, axes=(2, 1, 0))
    flags = np.transpose(flags, axes=(2, 1, 0))

    # Load frequencies
    tb.open(f"{ms_path}::SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")[:, 0] / HZ_IN_GHZ  # GHz
    tb.close()

    # Load correlation types & labels
    tb.open(f"{ms_path}::POLARIZATION")
    corr_types = tb.getcol("CORR_TYPE")[:, 0]
    tb.close()

    # Map correlation types to labels
    corr_labels = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
    labels = [corr_labels.get(ct, f"Corr {ct}") for ct in corr_types]

    # Validate
    if len(corrs) > 4:
        raise ValueError("Can plot at most 4 correlations in a 2×2 grid.")
    for corr in corrs:
        if corr < 0 or corr >= data.shape[2]:
            raise ValueError(f"Invalid correlation index: {corr}")

    # Prepare one figure per requested correlation
    tools = "pan,wheel_zoom,box_zoom,reset,save,hover"
    colors = ["blue", "red", "green", "orange"]
    figs: List[figure | None] = []

    for idx, corr in enumerate(corrs):
        # mask & flatten
        amp = np.abs(data[:, :, corr])
        amp = np.where(flags[:, :, corr], np.nan, amp)
        xs = np.tile(freqs, amp.shape[0])
        ys = amp.flatten()
        mask = ~np.isnan(ys)
        xs, ys = xs[mask], ys[mask]

        # make figure
        p = figure(
            title=f"{labels[corr]}: Amplitude vs Frequency",
            x_axis_label="Frequency (GHz)",
            y_axis_label="Amplitude",
            width=450,
            height=350,
            tools=tools,
            output_backend="webgl"
        )
        p.scatter(xs, ys,
                  marker="circle",
                  size=DEFAULT_MARKER_SIZE,
                  color=colors[idx % len(colors)],
                  alpha=DEFAULT_ALPHA)
        figs.append(p)

    # pad to 4 panels
    while len(figs) < 4:
        figs.append(None)

    # arrange in 2x2 grid
    grid = gridplot([[figs[0], figs[1]],
                     [figs[2], figs[3]]])

    show(grid)

def plot_uv_coverage(ms_path: str, in_notebook: bool = False, save_path: str = 'uv_coverage.html') -> None:
    """
    Plot UV coverage of a CASA MeasurementSet.

    Parameters
    ----------
    ms_path : str
        Path to the CASA measurement set.
    """
    # Decide how to render the plot
    if in_notebook:
        output_notebook()
    else:
        logger.info(f"Saving plot to {save_path}")
        output_file(save_path)

    # Load data from MS
    tb.open(ms_path)
    uvw = tb.getcol("UVW")
    tb.close()
    uvw = np.transpose(uvw, axes=(1, 0))  # (nRow, 3) -> (3, nRow)

    # Get the frequency from the SPECTRAL_WINDOW table
    tb.open(f"{ms_path}::SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")[:, 0]  # (nChan,)
    tb.close()

    f_ref = freqs.mean()

    # Get u's and v's
    u = uvw[:, 0]
    v = uvw[:, 1]

    # Convert uv from metres to wavelengths
    # u, v in wavelengths
    u = u / (SPEED_OF_LIGHT / f_ref)  # m to wavelengths
    v = v / (SPEED_OF_LIGHT / f_ref)  # m to wavelengths

    # Plot with bokeh
    p = figure(
        title="UV Coverage",
        x_axis_label=r"u ($$\lambda$$)",
        y_axis_label=r"v ($$\lambda$$)",
        width=600,
        height=550,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        output_backend="webgl"
    )

    # (u, v) in blue and (-u, -v) in red
    p.scatter(u, v, marker="circle", size=DEFAULT_MARKER_SIZE, color="blue", alpha=DEFAULT_ALPHA, legend_label="(u, v)")
    p.scatter(-u, -v, marker="circle", size=DEFAULT_MARKER_SIZE, color="red", alpha=DEFAULT_ALPHA, legend_label="(-u, -v)")

    p.legend.location = "top_right"
    show(p)

def plot_custom(x, y, x_label: str = 'X', y_label: str = 'Y', title: str = 'Title') -> None:
    """
    Plot any supplied x and y axes using Bokeh.

    Parameters
    ----------
    x : array-like
        Data for the x-axis.
    y : array-like
        Data for the y-axis.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    title : str
        Title of the plot.
    """
    # Render inline in a notebook
    output_notebook()

    # Plot with bokeh
    p = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        width=900,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        output_backend="webgl"
    )

    # Scatter plot
    p.scatter(x, y, marker="circle", size=DEFAULT_MARKER_SIZE, color="blue", alpha=0.5)

    show(p)
