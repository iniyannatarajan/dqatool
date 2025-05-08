from casatasks import tclean
from casatools import table
import bdsf
from dqatool.logging_config import get_logger
from dqatool.constants import DEFAULT_IMAGING_PARAMS, SECONDS_IN_DAY
import pandas as pd
import datetime
from math import ceil
from astropy.time import Time

tb = table()

# Create a logger for this file
logger = get_logger(__name__)

def make_image(ms_path: str, image_name: str, imager_name: str = "tclean", imager_params: dict = DEFAULT_IMAGING_PARAMS) -> None:
    """
    Parse and log the arguments for the imager.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set (MS).
    image_name : str
        Name of the output image file.
    imager_name : str, optional
        Name of the imager to be used. Default is "tclean".
    imager_params : dict, optional
        Dictionary of parameters for the imager. Default is DEFAULT_IMAGING_PARAMS.
    """
    logger.info(f"Using imager: {imager_name}")
    logger.info(f"MS Name: {ms_path}")
    logger.info(f"Image Name: {image_name}")
    logger.info(f"Imager Parameters: {imager_params}")

    # Create an image using the specified imager
    if imager_name == "tclean":
        tclean(vis=ms_path, imagename=image_name, **imager_params)
    else:
        logger.error(f"Imager {imager_name} is not supported.")
        raise ValueError(f"Imager {imager_name} is not supported.")
    
    logger.info(f"Image {image_name} created successfully.")

def image_time_chunks(ms_path: str, chunk_minutes: float, imager_name: str = 'tclean', out_prefix: str = 'timechunk', imager_params: dict = DEFAULT_IMAGING_PARAMS) -> None:
    """
    Divide a Measurement Set into contiguous time chunks and image each chunk.

    Parameters
    ----------
    ms_path : str
        Path to the input Measurement Set (MS).
    chunk_minutes : float
        Duration of each time chunk in minutes.
    imager_name : str, optional
        Name of the imager to be used. Default is 'tclean'.
    out_prefix : str, optional
        Prefix for the output image names. Default is 'timechunk'.
    imager_params : dict, optional
        Dictionary of parameters for the imager. Default is DEFAULT_IMAGING_PARAMS.

    Notes
    -----
    This function calculates the observation time range from the Measurement Set,
    divides it into chunks of the specified duration, and images each chunk using
    the specified imager. The resulting images are named sequentially with the
    provided prefix.
    """
    # Check if the imager is supported
    if imager_name != "tclean":
        logger.error(f"Imager {imager_name} is not supported.")
        raise ValueError(f"Imager {imager_name} is not supported.")

    # Find scan start and end times (in CASA MJD seconds)
    tb.open(ms_path)
    times = tb.getcol('TIME')
    tb.close()

    t_start = times.min()
    t_end = times.max()

    # Compute chunk length in seconds and the number of chunks
    chunk_secs = chunk_minutes * 60.0
    n_chunks = int(((t_end - t_start) / chunk_secs) + 0.9999)

    # Loop over chunks to image
    for chunk_id in range(n_chunks):
        start_sec = t_start + chunk_id * chunk_secs
        end_sec   = min(t_start + (chunk_id+1) * chunk_secs, t_end)
        tstart = Time(start_sec / SECONDS_IN_DAY, format='mjd').to_datetime()
        tend = Time(end_sec / SECONDS_IN_DAY, format='mjd').to_datetime()

        imgname = f"{out_prefix}_{chunk_id:02d}"
        trange = f"{tstart.strftime('%H:%M:%S')}~{tend.strftime('%H:%M:%S')}"

        logger.info(f"Imaging chunk {chunk_id+1}/{n_chunks} ({imgname}), timerange={trange}")
        tclean(vis=ms_path, imagename=imgname, timerange=trange, **imager_params)

def make_source_catalog(image_name: str, catalog_name: str, mean_map: str = 'map', rms_map: bool = True,
                        thresh: str = 'hard', thresh_isl: int = 3, thresh_pix: int = 7,
                        catalog_format: str = 'ascii', catalog_type: str = 'srl', clobber: bool = False) -> None:
    """
    Generate a source catalog from the image using BDSF.

    Parameters
    ----------
    image_name : str
        Name of the input image file.
    catalog_name : str
        Name of the output catalog file.
    mean_map : str, optional
        Type of background mean map to compute, recognizable by BDSF. Default is 'map'.
    rms_map : bool, optional
        Whether to compute and use a 2-D RMS map for source detection. Default is True.
    catalog_format : str, optional
        Format of the output catalog. Default is 'ascii'.
    catalog_type : str, optional
        Type of catalog to generate. Default is 'srl' (source list).
    clobber : bool, optional
        Whether to overwrite the catalog file if it already exists. Default is False.
    """
    logger.info(f"Generating source catalog from {image_name}...")

    # Identify bright sources
    img = bdsf.process_image(image_name, mean_map=mean_map, rms_map=rms_map, thresh=thresh,
                             thresh_isl=thresh_isl, thresh_pix=thresh_pix, advanced_opts=True,
                             fittedimage_clip=3.0, group_tol=0.5, group_by_isl=False)

    # Write out catalog
    img.write_catalog(outfile=catalog_name, format=catalog_format, clobber=clobber, catalog_type=catalog_type)

    logger.info(f"Source catalog {catalog_name} created successfully.")

def display_catalog(catalog_name: str) -> None:
    """
    Display the brightest 20 sources from the source catalog.

    This function reads the source catalog, processes it to extract relevant
    columns, and displays the top 20 sources sorted by their total flux in
    descending order.

    Parameters
    ----------
    catalog_name : str
        Name of the catalog file to be displayed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the top 20 brightest sources with columns:
        'Source_id', 'RA', 'DEC', and 'Total_flux'.
    """
    logger.info(f"Displaying catalog {catalog_name}...")

    # Before reading in the catalog, pre-process the file to remove any lines that start with '#'
    with open(catalog_name) as f:
        for _ in range(5):
            f.readline()
        raw = f.readline().strip()
    names = raw.lstrip("#").strip().split()

    # Now read in the catalog, passing names manually
    df = pd.read_csv(catalog_name, names=names, skiprows=6, sep='\s+', header=None, comment=None)
    # Strip any leading ‘#’ from all column names
    df.columns = df.columns.str.lstrip("#")

    df2 = df[["Source_id", "RA", "DEC", "Total_flux"]]

    # Sort by Total_flux descending and take the top 20
    brightest20 = df2.sort_values("Total_flux", ascending=False).head(20)

    return brightest20
