import pathlib
from typing import Tuple

import pandas as pd

import piml.config
from piml.utils import df_f64_f32, to_gz_csv


def split_test_train(df: pd.DataFrame, test_interval: Tuple[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split into test and training. """
    df_test_mask = df["TIME"].between(*test_interval, inclusive="left")
    df_test = df.loc[df_test_mask]
    df_train = df.loc[~df_test_mask]

    return df_train, df_test


def write_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame, base_name: str, train_test_dir: pathlib.Path) -> None:
    """ Write train dataset, test dataset, and list of features to csv files. """
    to_gz_csv(
        df_f64_f32(df_train),
        train_test_dir / f"{base_name}_TRAIN.csv.gz",
        index=False
    )
    to_gz_csv(
        df_f64_f32(df_test),
        train_test_dir / f"{base_name}_TEST.csv.gz",
        index=False
    )


def validate_dataset(df: pd.DataFrame, dim_vars: piml.config.DimVars):
    required_vars = dim_vars.dim_all_str + ["TIME", "DAY_YEAR"]
    missing_vars = set(required_vars) - set(df.columns)
    if len(missing_vars) > 0:
        raise ValueError(f"Validation failed! The following variables are missing: {missing_vars}.")


if __name__ == '__main__':
    ws = piml.Workspace.auto()

    for f in ws.data_processed.glob("*.csv.gz"):
        # Read
        print(f"Reading {f.name}... ", end=" ")
        df = pd.read_csv(f)
        df = df.rename(columns=ws.config.data.col_to_var)

        # Validate
        validate_dataset(df, ws.dim_vars)
        print("Valid! ", end=" ")

        # Split into test and training
        test_interval = ws.config.data.test_interval
        test_interval = (str(test_interval[0]), str(test_interval[1]))
        print(f"Performing train/test split with test interval {test_interval[0]} -- {test_interval[1]}... ", end=" ")
        df_train, df_test = split_test_train(df, test_interval)

        # Print diagnostics
        print(f"Test ratio: {len(df_test) / len(df):.2f}.", end=" ")
        print(f"Number of days in test: {len(df_test['DAY_YEAR'].unique())}.", end=" ")

        # Create base name
        base_name = str(f.name)
        for s in f.suffixes:
            base_name = base_name.replace(s, "")
        base_name += f"_{test_interval[0]}_{test_interval[1]}"

        # Write to disk
        write_dataset(df_train, df_test, base_name=base_name, train_test_dir=ws.data_train_test)
        print("Done!")
