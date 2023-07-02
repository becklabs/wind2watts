import os
import argparse

from wind2watts.fit.util import fit_to_df
from wind2watts.data.util import validate_lat_long
from wind2watts.wind.providers import OpenMeteoProvider
from wind2watts.wind.util import add_wind_data


def build_dataset(
    input_dir: str,
    output_dir: str,
    time_col: str,
    lat_col: str,
    long_col: str,
    elevation_column: str,
    skip_existing: bool,
    verbose: bool,
):
    def condprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    os.makedirs(output_dir, exist_ok=True)

    provider = OpenMeteoProvider()

    # Add wind_data to each fit file
    for fit_file in os.listdir(input_dir):
        if not fit_file.endswith(".fit"):
            continue
        output_name = fit_file.split(".")[0]
        output_file = os.path.join(output_dir, output_name + ".csv")

        df = fit_to_df(os.path.join(input_dir, fit_file))
        # Check that all columns exist
        if not all(
            col in df.columns for col in [time_col, lat_col, long_col, elevation_column]
        ):
            condprint(f"Skipping {fit_file} because it is missing columns")
            continue

        # Check that df has not already been processed
        if os.path.exists(output_file) and skip_existing:
            condprint(f"Skipping {fit_file} because it has already been processed")
            continue

        df = df[df[lat_col].notnull() & df[long_col].notnull() & df[time_col].notnull()]

        lat = df[lat_col].values
        long = df[long_col].values
        if not validate_lat_long(lat, long).all():
            df[lat_col] = df[lat_col] / 1e7
            df[long_col] = df[long_col] / 1e7
            condprint(f"Corrected lat/long for {fit_file}")

        if not validate_lat_long(df[lat_col].values, df[long_col].values).all():
            condprint(f"Skipping {fit_file} because it has invalid lat/long")
            continue

        condprint(f"Getting wind data for {fit_file}")
        df_with_wind = add_wind_data(df, time_col, lat_col, long_col, provider)
        df_with_wind.to_csv(output_file, index=False)
        condprint(f"Saved {output_file}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="../data/fit/strava_export/",
        help="The input directory where the data is stored.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/dataframes/strava_export/",
        help="The output directory where the dataframes will be stored.",
    )

    parser.add_argument(
        "--time_col",
        type=str,
        default="timestamp",
        help="The name of the time column in the dataframe.",
    )

    parser.add_argument(
        "--lat_col",
        type=str,
        default="position_lat",
        help="The name of the latitude column in the dataframe.",
    )

    parser.add_argument(
        "--long_col",
        type=str,
        default="position_long",
        help="The name of the longitude column in the dataframe.",
    )

    parser.add_argument(
        "--ele_col",
        type=str,
        default="enhanced_altitude",
        help="The name of the elevation column in the dataframe.",
    )

    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=True,
        help="A flag to skip processing files that already exist in the output directory.",
    )

    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="A flag to print verbose output.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    build_dataset(
        args.input_dir,
        args.output_dir,
        args.time_col,
        args.lat_col,
        args.long_col,
        args.ele_col,
        args.skip_existing,
        args.verbose,
    )


if __name__ == "__main__":
    main()
