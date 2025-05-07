from casatasks import tclean
import bdsf
from dqatool.logging_config import get_logger
from dqatool.constants import DEFAULT_IMAGING_PARAMS
import pandas as pd

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
