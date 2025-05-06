from casatasks import tclean
from dqatool.logging_config import get_logger
from dqatool.constants import DEFAULT_IMAGING_PARAMS

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


