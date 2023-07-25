import pathlib
from typing import Union

import numpy as np
import pandas as pd


def df_f64_f32(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert all columns of df with float64 to float32.
    https://stackoverflow.com/questions/30494569/how-to-force-pandas-read-csv-to-use-float32-for-all-float-columns
    """
    float64_cols = df.select_dtypes(include='float64').columns
    mapper = {col_name: np.float32 for col_name in float64_cols}
    return df.astype(mapper)


def to_gz_csv(df: pd.DataFrame, path: Union[str, pathlib.Path], **kwargs) -> None:
    """ Write dataframe to gzip compressed REPRODUCIBLE csv file """
    path = pathlib.Path(path)
    if not path.suffix != ".gz":
        # Add gz suffix if not set, yet.
        path = path.parent / f"{path.stem}.gz"

    # Disable inclusion of modification time in GZIP compression for reproducibility.
    df.to_csv(path_or_buf=path, compression={"method": "gzip", "mtime": 0}, **kwargs)
