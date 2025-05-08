from casatools import table
import numpy as np
from astropy.time import Time
from dqatool.logging_config import get_logger
from scipy.stats import median_abs_deviation as mad
from dqatool.constants import DEFAULT_NSIGMA, DEFAULT_WINDOW_SIZE, DEFAULT_SD_SCALE, SECONDS_IN_DAY
from itertools import combinations
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

tb = table()

# Create a logger for this file
logger = get_logger(__name__)

def compute_baseline_rfi(bldata: np.ndarray, bltimes: np.ndarray, window_size: int, nsigma: float, sdscale: float) -> np.ndarray:
    """
    Compute RFI-affected time indices for a given baseline, over all correlations.

    This function identifies time indices where the amplitude of the data deviates
    significantly from the rolling median by more than `nsigma` times the scaled MAD.

    Parameters
    ----------
    bldata : np.ndarray
        Array of shape (n_times, n_chan, n_corr) containing baseline data.
        NaN values indicate already-flagged data.
    bltimes : np.ndarray
        Array of shape (n_times,) containing time values for the baseline.
    window_size : int
        Size of the rolling window used for median and MAD computation.
    nsigma : float
        Threshold multiplier for flagging outliers. Times at which the 
        amplitude exceeds median ± `nsigma` × MAD are flagged.
    sdscale : float
        Scale factor applied to the MAD to make it comparable to the 
        standard deviation (e.g., 1.4826 for Gaussian noise).

    Returns
    -------
    np.ndarray
        Array of time values (from `bltimes`) corresponding to RFI-affected 
        indices in any correlation.

    Notes
    -----
    - A leave-one-out strategy is used when computing the rolling statistics: for each
      time sample, the window excludes that sample to avoid bias.
    - The function computes thresholds for each correlation independently and flags
      time indices where any correlation exceeds its respective threshold.
    """
    n_times, n_chan, n_corr = bldata.shape
    half_w = window_size // 2

    # Compute per-time, per-corr mean amplitude, shape -> (n_times, n_corr)
    data_mean = np.nanmean(np.abs(bldata), axis=1)

    # Allocate arrays for rolling median & MAD
    med = np.zeros((n_times, n_corr))
    mad_arr = np.zeros((n_times, n_corr))

    # For each time and each corr compute leave‑one‑out window
    for i in range(n_times):
        w0 = max(0, i - half_w)
        w1 = min(n_times, i + half_w + 1)
        if (w1 - w0) < window_size:
            if w0 == 0:
                w1 = min(n_times, window_size)
            else:
                w0 = max(0, n_times - window_size)

        # Extract the entire block for all corrs, shape (win, n_chan, n_corr)
        block = np.abs(bldata[w0:w1, :, :])
        # Flatten block and compute offset for i
        block = block.reshape(-1, n_corr)   # shape ((w1−w0)*n_chan, n_corr)
        offset = (i - w0) * n_chan

        for c in range(n_corr):
            # leave‑one‑out data for this corr
            loo = np.concatenate([block[:offset, c], block[offset + n_chan:, c]])
            med[i, c]       = np.nanmedian(loo)
            mad_arr[i, c]   = mad(loo, nan_policy="omit", scale=sdscale)

    # Find outliers
    thres_hi = med + nsigma * mad_arr
    thres_lo = med - nsigma * mad_arr

    # Mask where any correlation exceeds its own threshold
    is_bad = np.zeros(n_times, dtype=bool)
    for c in range(n_corr):
        is_bad |= (data_mean[:, c] > thres_hi[:, c]) | (data_mean[:, c] < thres_lo[:, c])

    bad_times = bltimes[is_bad]

    return bad_times

def apply_flags_to_ms(ms_path: str, antenna1: np.ndarray, antenna2: np.ndarray, 
                      flag_times_bldict: dict[tuple[int,int], np.ndarray]) -> None:
    """
    Apply the RFI flags stored in flag_times_bldict back to the MS by overwriting
    the FLAG and FLAG_ROW columns.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MeasurementSet.
    antenna1 : np.ndarray
        The ANTENNA1 column (shape: nRows,) originally read from the MS.
    antenna2 : np.ndarray
        The ANTENNA2 column (shape: nRows,) originally read from the MS.
    flag_times_bldict : dict[(int,int), np.ndarray]
        Mapping from (ant1,ant2) → array of time values to flag (MJD seconds).
    """
    # reopen to read existing flags
    tb.open(ms_path)
    alltimes = tb.getcol("TIME")
    flag_row = tb.getcol("FLAG_ROW")
    flag      = tb.getcol("FLAG")
    tb.close()

    # for each baseline and each time, set both FLAG_ROW and FLAG
    for (ant1, ant2), times in flag_times_bldict.items():
        for tval in times:
            # find the index in the MS corresponding to this baseline & time
            idx = np.where(
                (antenna1 == ant1) &
                (antenna2 == ant2) &
                (alltimes  == tval)
            )[0]
            if idx.size == 0:
                # no matching row ‑‑ perhaps time rounding; skip
                continue
            flag_row[idx] = True
            flag[:, :, idx] = True

    # write back into the MS
    tb.open(ms_path, nomodify=False)
    tb.putcol("FLAG_ROW", flag_row)
    tb.putcol("FLAG",     flag)
    tb.close()

    logger.info("Flags updated in the MS.")

def detect_rfi_1d(ms_path: str, window_size: int = DEFAULT_WINDOW_SIZE, nsigma: int = DEFAULT_NSIGMA, sdscale: float = DEFAULT_SD_SCALE,
                  overwriteflags: bool = False, flagfile: str = "rfi_flags.txt") -> None:
    """
    Detect radio-frequency interference (RFI) in a MeasurementSet (MS) by analysing 
    baseline data using a rolling median and median absolute deviation (MAD) in the 
    time domain.

    This function processes each baseline (antenna pair) in the MS, computes the 
    time-series of channel-averaged amplitudes for all correlations, and flags time 
    samples where the amplitudes deviate from the rolling median by more than 
    `nsigma` × MAD. Detected flags can optionally be written back into the MS, and 
    a CASA-compatible flag file is always generated.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MeasurementSet (MS).
    window_size : int, optional
        Number of time samples in the sliding window used to compute the rolling 
        median and MAD (default is `DEFAULT_WINDOW_SIZE`).
    nsigma : int, optional
        Threshold multiplier for flagging outliers. Time samples where the amplitude 
        exceeds median ± `nsigma` × MAD are flagged (default is `DEFAULT_NSIGMA`).
    sdscale : float, optional
        Scale factor applied to the MAD to make it comparable to the standard deviation 
        (default is `DEFAULT_SD_SCALE`, typically 1.4826 for Gaussian noise).
    overwriteflags : bool, optional
        If True, update the FLAG and FLAG_ROW columns in the MS with the detected RFI 
        flags. If False, the MS is left unmodified (default is False).
    flagfile : str, optional
        Filename for the output CASA-compatible flag list. Each line specifies the 
        antenna pair and time range of a detected RFI event (default is "rfi_flags.txt").

    Returns
    -------
    None
        All outputs are written to `flagfile` and optionally to the MS (if 
        `overwriteflags=True`). No value is returned.

    Notes
    -----
    - Data flagged in the original MS are ignored (masked to NaN) before computing 
      statistics.
    - A leave-one-out strategy is used when computing the rolling statistics: for each 
      time sample, the window excludes that sample to avoid bias.
    - The output flagfile uses the CASA flag file syntax (`antenna='i&j' timerange='HH:MM:SS'`) 
      to record baseline and time information.

    Examples
    --------
    Detect RFI without modifying the MS and write flags to "my_flags.txt":
    
    >>> detect_rfi_1d("mydata.ms", window_size=25, nsigma=5, sdscale=1.4826, overwriteflags=False, flagfile="my_flags.txt")

    Detect RFI and update the MS in place:

    >>> detect_rfi_1d("mydata.ms", overwriteflags=True)
    """
    # Load data from MS
    tb.open(ms_path)
    antenna1 = tb.getcol("ANTENNA1")
    antenna2 = tb.getcol("ANTENNA2")
    
    # Assign values to some common variables
    antstart = 0
    antstop = len(np.union1d(antenna1, antenna2))
    total_baselines = (antstop - antstart) * (antstop - antstart - 1) // 2
    baseline_iter = combinations(range(antstart, antstop), 2)

    half_wsize = window_size // 2

    # Dictionary to hold flagged times for each baseline
    # The keys are tuples of (antenna1, antenna2) and the values are arrays of times
    flag_times_bldict = {}

    for ant1, ant2 in tqdm(baseline_iter, total=total_baselines, desc="Processing baselines"):
        with logging_redirect_tqdm():
            bltab = tb.query(f"ANTENNA1=={ant1} AND ANTENNA2=={ant2}")
            bldata = bltab.getcol("DATA")
            # Check if the baseline data is empty and if so, skip to the next baseline
            if bldata.size == 0:
                logger.warning(f"No data found for baseline {ant1}-{ant2}")
                continue

            bltimes = bltab.getcol("TIME")
            blflag = bltab.getcol("FLAG")
            bltab.close()

            # To match the casacore.tables array shapes, we need to transpose the data
            bldata = np.transpose(bldata, axes=(2, 1, 0))  # shape = (n_times, n_chan, 4)
            blflag = np.transpose(blflag, axes=(2, 1, 0))  # shape = (n_times, n_chan, 4)

            # mask flagged visibilities
            bldata[blflag] = np.nan
            if np.all(np.isnan(bldata)):
                logger.debug(f"All data are flagged for baseline {ant1}-{ant2}. Skipping...")
                continue

            # compute RFI times
            bad = compute_baseline_rfi(bldata, bltimes, window_size, nsigma, sdscale)
            if bad.size:
                flag_times_bldict[(ant1, ant2)] = bad

    # Close the MS table
    tb.close()

    # # Write flags to MS if requested
    if overwriteflags:
        apply_flags_to_ms(ms_path, antenna1, antenna2, flag_times_bldict)
    else:
        logger.info("Flags not written to the MS. Use overwriteflags=True to write flags.")

    # Write the flags to a file in a format compatible with CASA
    with open(flagfile, "w") as f:
        for (ant1, ant2), times in flag_times_bldict.items():
            # Convert times to datetime
            times_dt = Time(times / SECONDS_IN_DAY, format='mjd').to_datetime()
            for tval in times_dt:
                f.write(f"antenna='{ant1}&{ant2}' timerange='{tval.strftime('%H:%M:%S')}'\n")
        
    logger.info(f"RFI detection completed and flags written to {flagfile}.")

