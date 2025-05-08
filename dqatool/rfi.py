from casatools import table
import numpy as np
from astropy.time import Time
from dqatool.logging_config import get_logger
from scipy.stats import median_abs_deviation as mad
from dqatool.constants import DEFAULT_NSIGMA, DEFAULT_WINDOW_SIZE, DEFAULT_SD_SCALE, SECONDS_IN_DAY
from itertools import combinations
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
#import matplotlib.pyplot as plt

tb = table()

# Create a logger for this file
logger = get_logger(__name__)

def detect_rfi_1d(ms_path: str, window_size: int = DEFAULT_WINDOW_SIZE, nsigma: int = DEFAULT_NSIGMA, sdscale: float = DEFAULT_SD_SCALE,
                  overwriteflags: bool = False, flagfile: str = "rfi_flags.txt") -> None:
    """
    Detect radio-frequency interference (RFI) in a MeasurementSet by baseline using a 
    rolling median and median absolute deviation (MAD) in the time domain.

    This function scans each baseline (antenna pair) in the MS, computes the time-series
    of channel-averaged amplitudes for the XX and YY correlations, and flags time samples
    whose amplitudes deviate from the rolling median by more than `nsigma` × MAD.
    Detected flags can optionally be written back into the MS, and a CASA-compatible
    flag file is always generated.

    Parameters
    ----------
    ms_path : str
        Path to the CASA MS.
    window_size : int, optional
        Number of time samples in the sliding window used to compute the rolling median
        and MAD (default is `DEFAULT_WINDOW_SIZE`).
    nsigma : int, optional
        Threshold multiplier for flagging outliers. Times at which the amplitude exceeds
        median ± `nsigma` * MAD are flagged (default is `DEFAULT_NSIGMA`).
    sdscale : float, optional
        Scale factor applied to the MAD to make it comparable to the standard deviation
        (default is `DEFAULT_SD_SCALE`, typically 1.4826 for Gaussian noise).
    overwriteflags : bool, optional
        If True, update the FLAG and FLAG_ROW columns in the MS with the
        detected RFI flags. If False, the MS is left unmodified (default False).
    flagfile : str, optional
        Filename for the output CASA-compatible flag list. Each line lists the
        antenna pair and time range of a detected RFI event (default "rfi_flags.txt").

    Returns
    -------
    None
        All outputs are written to `flagfile` and optionally to the MS (if `overwriteflags=True`).
        No value is returned.

    Notes
    -----
    - Data flagged in the original MS are ignored (masked to NaN) before computing
      statistics.
    - A leave-one-out strategy is used when computing the rolling statistics: for each
      time sample, the window excludes that sample to avoid bias.
    - The output flagfile uses the flag file syntax understood by CASA (`antenna='i&j' timerange='HH:MM:SS'`)
      to record baseline and time information.

    Examples
    --------
    Detect RFI without modifying the MS and write flags to "my_flags.txt":
    
    >>> detect_rfi_1d("mydata.ms", window_size=25, nsigma=5,
    ...               sdscale=1.4826, overwriteflags=False,
    ...               flagfile="my_flags.txt")

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
            #bltimeoffsets = bltimes - bltimes[0]
            blflag = bltab.getcol("FLAG")
            bltab.close()

            # To match the casacore.tables array shapes, we need to transpose the data
            bldata = np.transpose(bldata, axes=(2, 1, 0))  # shape = (n_times, n_chan, 4)
            blflag = np.transpose(blflag, axes=(2, 1, 0))  # shape = (n_times, n_chan, 4)

            # mask flagged visibilities
            bldata[blflag] = np.nan
            if np.all(np.isnan(bldata)):
                #logger.warning(f"All data are flagged for baseline {ant1}-{ant2}. Skipping...")
                continue

            # average over channels for each time, parallel hands only
            dataxx = np.nanmean(np.abs(bldata[:, :, 0]), axis=1)
            datayy = np.nanmean(np.abs(bldata[:, :, 3]), axis=1)

            n_times, n_chan = bldata.shape[0], bldata.shape[1]

            # allocate stats arrays
            medianxx   = np.zeros(n_times)
            medianyy   = np.zeros(n_times)
            dataxx_mad = np.zeros(n_times)
            datayy_mad = np.zeros(n_times)

            # compute rolling median and MAD values given the window size
            # handle edge cases for the first and last few points
            for ntime in range(n_times):
                # Ensure that the windows stay the same size for data points at the edges
                w_start = max(0, ntime - half_wsize)
                w_end   = min(n_times, ntime + half_wsize + 1)

                # pad the edges
                if w_end - w_start < window_size:
                    if w_start == 0:
                        w_end = min(n_times, window_size)
                    elif w_end == n_times:
                        w_start = max(0, n_times - window_size)
        
                # Reshape data to compute time slices to skip
                segment_xx = np.abs(bldata[w_start:w_end, :, 0]).reshape(-1)
                segment_yy = np.abs(bldata[w_start:w_end, :, 3]).reshape(-1)

                # Compute the indices of time slices to skip
                idx_offset = ntime - w_start
                t0 = idx_offset * n_chan
                t1 = t0 + n_chan

                # stitch together the data segments
                loo_xx = np.concatenate([segment_xx[:t0], segment_xx[t1:]])
                loo_yy = np.concatenate([segment_yy[:t0], segment_yy[t1:]])

                # Compute the median and MAD for the current time slice
                medianxx[ntime]   = np.nanmedian(loo_xx)
                medianyy[ntime]   = np.nanmedian(loo_yy)
                dataxx_mad[ntime] = mad(loo_xx, nan_policy='omit', scale=sdscale, axis=None)
                datayy_mad[ntime] = mad(loo_yy, nan_policy='omit', scale=sdscale, axis=None)

            # Set thresholds for outlier detection
            thres_up_xx = medianxx + nsigma * dataxx_mad
            thres_lo_xx = medianxx - nsigma * dataxx_mad
            thres_up_yy = medianyy + nsigma * datayy_mad
            thres_lo_yy = medianyy - nsigma * datayy_mad

            # Find outliers in dataxx and datayy
            outxx = np.where((dataxx > thres_up_xx) | (dataxx < thres_lo_xx))[0]
            outyy = np.where((datayy > thres_up_yy) | (datayy < thres_lo_yy))[0]

            flag_times_bldict[(ant1, ant2)] = bltimes[np.union1d(outxx, outyy)]

            # # Plot the results                
            # plt.figure(figsize=(12, 6))
            # plt.plot(bltimeoffsets, dataxx, label='XX')
            # plt.plot(bltimeoffsets, dataxx_mad, label='MAD XX', color='blue')
            # plt.scatter(bltimeoffsets[outxx], dataxx[outxx], color='cyan', label='Outliers XX')

            # plt.plot(bltimeoffsets, datayy, label='YY')
            # plt.plot(bltimeoffsets, datayy_mad, label='MAD YY', color='red')
            # plt.scatter(bltimeoffsets[outyy], datayy[outyy], color='orange', label='Outliers YY')
            
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # plt.title(f'Outliers for baseline {ant1}-{ant2}')
            # plt.legend()
            # plt.grid()
            # plt.show()

    # Close the MS table
    tb.close()

    # Write flags to MS if requested
    if overwriteflags:
        # Read relevant arrays from MS
        tb.open(ms_path)
        alltimes = tb.getcol("TIME")
        flag_row = tb.getcol("FLAG_ROW")
        flag = tb.getcol("FLAG")
        tb.close()

        # Update flag arrays
        for (ant1, ant2), times in flag_times_bldict.items():
            for timeval in times:
                idx = np.where((antenna1 == ant1) & (antenna2 == ant2) & (alltimes  == timeval))[0]
                flag_row[idx] = True
                flag[:, :, idx] = True

        tb.open(ms_path, nomodify=False)
        tb.putcol("FLAG_ROW", flag_row)
        tb.putcol("FLAG", flag)
        tb.close()

        logger.info("Flags updated in the MS.")
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

