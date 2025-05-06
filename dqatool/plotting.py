from casacore import tables
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.time import Time
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from .constants import SECONDS_IN_DAY, XX_CORRID, YY_CORRID, DEFAULT_MARKER_SIZE, DEFAULT_ALPHA, HZ_IN_GHZ, SPEED_OF_LIGHT
from typing import Literal, List

# Set Seaborn style for all plots
sns.set_theme(style="whitegrid")

def plot_amp_vs_time(ms_path: str, colname: Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA'] = 'DATA', corrs: List[Literal[0, 1, 2, 3]] = [0]) -> None:
    """
    Plot visibility amplitudes vs. time as scatter points for the specified correlations.

    This function visualizes the amplitude of visibility data over time for the given 
    correlations in a CASA MeasurementSet (MS). The data extracted from the specified 
    column(s) are plotted using Bokeh.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MeasurementSet (MS).
    colname : Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA']
        Column name to extract visibility data from.
    corrs : List[Literal[0, 1, 2, 3]]
        List of correlation indices to plot (e.g., 0 for XX/RR, 1 for XY/RL, etc.).

    Raises
    ------
    ValueError
        If any of the specified correlation indices are invalid.
    """
    # Render inline in a notebook
    output_notebook()

    # Load data from MS
    vis = tables.table(ms_path, ack=False)
    data = vis.getcol(colname)    # shape: (nRow, nChan, nCorr)
    flags = vis.getcol("FLAG")   # same shape
    times = vis.getcol("TIME")   # shape: (nRow,)
    vis.close()

    # Convert times to datetime
    times_dt = Time(times / SECONDS_IN_DAY, format='mjd').to_datetime()

    # Load polarization labels from POLARIZATION table
    pol_table = tables.table(f"{ms_path}/POLARIZATION", ack=False)
    corr_types = pol_table.getcol("CORR_TYPE")[0]  # Get correlation types (e.g., [5, 6, 7, 8])
    pol_table.close()

    # Map correlation types to labels
    corr_labels = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
    labels = [corr_labels.get(corr, f"Corr {corr}") for corr in corr_types]

    # Prepare the Bokeh figure
    p = figure(
        title="Visibility Amplitude vs. Time",
        x_axis_type="datetime",
        x_axis_label="Time (UTC)",
        y_axis_label="Amplitude",
        width=900,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        output_backend="webgl"
    )

    # Define colors for each correlation
    colors = ["blue", "red", "green", "orange"]

    # Plot data for each correlation in `corrs`
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

        # Add scatter plot for this correlation
        label = labels[corr] if corr < len(labels) else f"Corr {corr}"
        color = colors[i % len(colors)]  # Cycle through colors if more correlations are provided
        p.scatter(xs, ys, marker="circle", size=DEFAULT_MARKER_SIZE, color=color, alpha=DEFAULT_ALPHA, legend_label=label)

    # Configure legend
    p.legend.location = "top_right"

    # Show the plot
    show(p)

def plot_amp_vs_freq(ms_path: str, colname: Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA'] = 'DATA', corrs: List[Literal[0, 1, 2, 3]] = [0]) -> None:
    """
    Plot visibility amplitudes vs. frequency as scatter points for the specified correlations.

    This function visualizes the amplitude of visibility data as a function of frequency
    for the given correlations in a CASA MeasurementSet (MS). The data extracted from the 
    specified column(s) are plotted using Bokeh.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MS.
    colname : Literal['DATA', 'MODEL_DATA', 'CORRECTED_DATA']
        Column name to extract visibility data from.
    corrs : List[Literal[0, 1, 2, 3]]
        List of correlation indices to plot (e.g., 0 for XX/RR, 1 for XY/RL, etc.).

    Raises
    ------
    ValueError
        If any of the specified correlation indices are invalid.
    """
    # Render inline in a notebook
    output_notebook()

    # Load data from MS
    vis = tables.table(ms_path, ack=False)
    data = vis.getcol(colname)    # shape: (nRow, nChan, nCorr)
    flags = vis.getcol("FLAG")   # same shape
    vis.close()

    # Load frequency information from SPECTRAL_WINDOW table
    spw = tables.table(f"{ms_path}/SPECTRAL_WINDOW", ack=False)
    freqs = spw.getcol("CHAN_FREQ")[0]  # shape: (nChan,)
    spw.close()

    # Convert frequencies from Hz to GHz
    freqs = freqs / HZ_IN_GHZ

    # Load polarization labels from POLARIZATION table
    pol_table = tables.table(f"{ms_path}/POLARIZATION", ack=False)
    corr_types = pol_table.getcol("CORR_TYPE")[0]  # Get correlation types (e.g., [5, 6, 7, 8])
    pol_table.close()

    # Map correlation types to labels
    corr_labels = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
    labels = [corr_labels.get(corr, f"Corr {corr}") for corr in corr_types]

    # Prepare the Bokeh figure
    p = figure(
        title="Visibility Amplitude vs. Frequency",
        x_axis_label="Frequency (GHz)",
        y_axis_label="Amplitude",
        width=900,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        output_backend="webgl"
    )

    # Define colors for each correlation
    colors = ["blue", "red", "green", "orange"]

    # Plot data for each correlation in `corrs`
    for i, corr in enumerate(corrs):
        if corr < 0 or corr >= data.shape[2]:
            raise ValueError(f"Invalid correlation index: {corr}")

        # Apply flags (masks) to data
        amp = np.abs(data[:, :, corr])  # Amplitude for the selected correlation
        amp = np.where(flags[:, :, corr], np.nan, amp)  # Mask flagged data

        # Flatten the data for plotting
        xs = np.tile(freqs, amp.shape[0])  # Repeat frequencies for each row
        ys = amp.flatten()  # Flatten the amplitude data

        # Remove NaNs
        mask = ~np.isnan(ys)
        xs, ys = xs[mask], ys[mask]

        # Add scatter plot for this correlation
        label = labels[corr] if corr < len(labels) else f"Corr {corr}"
        color = colors[i % len(colors)]  # Cycle through colors if more correlations are provided
        p.scatter(xs, ys, marker="circle", size=DEFAULT_MARKER_SIZE, color=color, alpha=DEFAULT_ALPHA, legend_label=label)

    # Configure legend
    p.legend.location = "top_right"

    # Show the plot
    show(p)

def plot_uv_coverage(ms_path: str) -> None:
    """
    Plot UV coverage of a CASA MeasurementSet.

    Parameters
    ----------
    ms_path : str
        Path to the CASA measurement set.
    """
    # render inline in a notebook
    output_notebook()

    # Load data from MS
    vis = tables.table(ms_path, ack=False)
    # Get the UVW coordinates
    uvw = vis.getcol("UVW")
    vis.close()

    # Get the frequency from the SPECTRAL_WINDOW table
    spw = tables.table(f"{ms_path}::SPECTRAL_WINDOW", ack=False)
    freqs = spw.getcol("CHAN_FREQ")[0]  # (nChan,)
    spw.close()

    # Get u's and v's
    u = uvw[:, 0]
    v = uvw[:, 1]

    # Convert uv from metres to wavelengths
    # u, v in wavelengths
    u = u / (SPEED_OF_LIGHT / freqs[0])  # m to wavelengths
    v = v / (SPEED_OF_LIGHT / freqs[0])  # m to wavelengths

    # Plot with bokeh
    p = figure(
        title="UV Coverage",
        x_axis_label="U (m)",
        y_axis_label="V (m)",
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
