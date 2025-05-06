import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("dqatool.log")  # Log to a file
    ]
)

# Create a logger for the module
def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the specified name.

    Args:
        name (str): Name of the logger, typically __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)