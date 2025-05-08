import numpy as np
from casatools import image
from dqatool.logging_config import get_logger

ia = image()

logger = get_logger(__name__)

def get_image_stats(imagename: str, rms_boxes: dict[str, tuple] | None = None) -> dict:
    """
    Compute global stats and per‑box RMS on a CASA image,
    discovering at runtime which axes are X (RA) and Y (Dec).

    Parameters
    ----------
    imagename : str
        CASA image rootname.
    rms_boxes : dict[str, tuple], optional
        Map from box name → tuple of two slice objects (x‑slice, y‑slice).
        These are applied on the RA and Dec axes, whichever positions
        they occupy.  All other axes (spectral, polarization, …) are
        left un‑sliced (slice(None)).

    Returns
    -------
    stats : dict
        Keys 'mean','median','std','rms','min','max' plus 'rms_<boxname>'
        for each box.
    """
    # Load image and header summary
    ia = image()
    ia.open(imagename)
    data    = ia.getchunk()
    summary = ia.summary() # to extract axis names
    ia.close()

    # Compute global image stats
    flat = data.ravel()
    stats = {
        'mean':   float(flat.mean()),
        'median': float(np.median(flat)),
        'std':    float(flat.std(ddof=0)),
        'rms':    float(np.sqrt(np.mean(flat**2))),
        'min':    float(flat.min()),
        'max':    float(flat.max())
    }

    # Find the RA and Dec axes. In a CASA image, the axis names are ['Right Ascension','Declination','Stokes','Frequency'].
    axisnames = summary.get('axisnames', [])
    ra_idx = next(
        (i for i,name in enumerate(axisnames)
         if ('ASCENSION' in name.upper() or name.upper().strip() == 'RA')),
        None
    )
    dec_idx = next(
        (i for i,name in enumerate(axisnames)
         if ('DECLINATION' in name.upper() or name.upper().strip() == 'DEC')),
        None
    )

    # If RA/Dec cannot be found, raise error
    if ra_idx is None or dec_idx is None:
        raise RuntimeError(f"Could not identify RA/Dec axes in {axisnames}")

    # Compute RMS for each box
    if rms_boxes:
        ndim = data.ndim
        for bname, box in rms_boxes.items():
            if len(box) != 2:
                raise ValueError(f"Box '{bname}' must be a 2‑tuple of slices")
            # build full index tuple of length ndim
            full = [slice(None)] * ndim
            full[ra_idx]  = box[0]
            full[dec_idx] = box[1]
            region = data[tuple(full)]
            stats[f'rms_{bname}'] = float(np.sqrt(np.mean(region.ravel()**2)))

    return stats
