import pandas as pd
import pytest
import pathlib
import os

from dqatool.imtools import display_catalog

DATA_DIR = pathlib.Path(__file__).parent / "data"

def test_display_catalog_returns_top_20(tmp_path):

    catalog_file = os.path.join(DATA_DIR, "sample_catalog.txt")

    # Run the function
    df = display_catalog(str(catalog_file))

    # Perform some basic checks
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Source_id", "RA", "DEC", "Total_flux"]
    assert len(df) == 20
    
    # Check that the DataFrame is sorted by Total_flux in descending order
    assert df["Total_flux"].is_monotonic_decreasing
