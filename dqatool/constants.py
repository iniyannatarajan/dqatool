# Constant definitions for dqatool

# constants for rfi submodule
DEFAULT_NSIGMA = 3 # threshold multiplier for cutoff
DEFAULT_WINDOW_SIZE = 9  # 8*15 s
DEFAULT_SD_SCALE = 1.4826  # scale factor for MAD to obtain standard deviation

# constants for plotting submodule
SECONDS_IN_DAY = 86400.0
XX_CORRID = 0
YY_CORRID = 3
DEFAULT_MARKER_SIZE = 2
DEFAULT_ALPHA = 0.3
HZ_IN_GHZ = 1e9
SPEED_OF_LIGHT = 299792458  # m/s

# constants for imtools submodule
DEFAULT_IMAGING_PARAMS = {
    "datacolumn": "data",
    "imsize": [4096, 4096],
    "cell": "2arcsec",
    "stokes": "I",
    "gridder": "wproject",
    "wprojplanes": 128,
    "deconvolver": "hogbom",
    "niter": 100000,
    "threshold": "30uJy",  # SARAO sensitivity calculator
    "pblimit": -0.1  # tclean documentation
}
