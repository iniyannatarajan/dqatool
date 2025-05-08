import numpy as np
import pytest

# import the module under test
import dqatool.stats as stats_mod
from dqatool.stats import get_image_stats

# --- DummyImage tool to stand in for casatools.image ---
class DummyImage:
    def __init__(self):
        # create simple 4×4 array for testing
        self.data = np.arange(16, dtype=float).reshape(4,4)

    def open(self, path):
        # ignore path
        pass

    def getchunk(self):
        # return a 2‑D array
        return self.data
    
    def summary(self):
        # declare the two axes as RA and Dec
        return {'axisnames': ['Right Ascension','Declination']}
    
    def close(self):
        pass

@pytest.fixture(autouse=True)
def patch_image_tool(monkeypatch):
    """
    Before each test, patch stats_mod.image() so it returns our DummyImage.
    """
    monkeypatch.setattr(stats_mod, 'image', lambda: DummyImage())

def test_global_stats_only():
    out = get_image_stats("ignored.image", rms_boxes=None)
    flat = DummyImage().data.ravel()
    # expected values
    exp = {
        'mean':   float(flat.mean()),
        'median': float(np.median(flat)),
        'std':    float(flat.std(ddof=0)),
        'rms':    float(np.sqrt(np.mean(flat**2))),
        'min':    float(flat.min()),
        'max':    float(flat.max())
    }
    assert out == exp

def test_per_box_rms_and_global_unchanged():
    boxes = {
        'small': (slice(0,2), slice(0,2)),    # top‑left 2×2
        'mid':   (slice(1,3), slice(1,3)),    # central 2×2
    }
    out = get_image_stats("ignored.image", rms_boxes=boxes)

    # check global still present
    assert 'rms' in out and out['rms'] == pytest.approx(np.sqrt(np.mean(DummyImage().data.ravel()**2)))

    # compute expected per‑box rms
    data = DummyImage().data
    for name, (sx, sy) in boxes.items():
        region = data[sx, sy].ravel()
        expected_rms = np.sqrt(np.mean(region**2))
        assert out[f'rms_{name}'] == pytest.approx(expected_rms)

def test_bad_box_raises_value_error():
    # box tuple must be length‑2
    bad = {'too_many': (slice(0,1), slice(0,1), slice(0,1))}
    with pytest.raises(ValueError):
        get_image_stats("ignored.image", rms_boxes=bad)
