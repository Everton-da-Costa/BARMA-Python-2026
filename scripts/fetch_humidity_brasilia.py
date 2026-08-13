"""Fetch daily relative humidity for Brasília from NASA POWER and aggregate to monthly.

Downloads daily RH2M (Relative Humidity at 2 Meters) observations from NASA
POWER, spanning 1999-01-01 to today minus two months to avoid NaN values from
NASA POWER. Replaces the API's -999.0 sentinel with NaN, scales percentages to
fractions (RH2M / 100), and aggregates daily means to month-end values via pandas
resampling.

Output files
------------
data/raw/raw_data_dict.json : untouched API response (NASA POWER JSON format)
data/processed/y_full_monthly_data.csv : month-end means, fractions in (0, 1)

Notes
-----
NASA POWER citation: NASA Langley Research Center (LaRC) POWER Project,
funded through the NASA Earth Science Directorate Applied Science Program.
https://power.larc.nasa.gov

Direct execution: `python scripts/fetch_humidity_brasilia.py`
"""

import json
from pathlib import Path

import pandas as pd
import requests

# Setting the directory to export files
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# -------------------------------------------------------------------------------------
# 1. Functions definitions
# -------------------------------------------------------------------------------------


def build_api_params(
    latitude: float, longitude: float, start: str, end: str
) -> dict[str, str | float]:
    """Construct the NASA POWER API parameter dict."""
    return {
        "parameters": "RH2M",
        "community": "AG",
        "latitude": latitude,
        "longitude": longitude,
        "start": start,
        "end": end,
        "format": "JSON",
    }


def fetch_nasa_humidity(api_params: dict[str, str | float], timeout: int = 30) -> dict:
    """Hit NASA POWER and return the parsed JSON"""
    response = requests.get(
        url="https://power.larc.nasa.gov/api/temporal/daily/point",
        params=api_params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def json_to_daily_df(raw_data_dict: dict) -> pd.DataFrame:
    """Extract RH2M from a NASA POWER JSON response and return a daily DataFrame."""

    # NASA POWER returns a nested dictionary. The daily data lives inside:
    # properties -> parameter -> RH2M (Relative Humidity at 2 Meters).
    daily_data_dict: dict = raw_data_dict["properties"]["parameter"]["RH2M"]

    # Convert the dictionary {"YYYYMMDD": value} into a Pandas Series
    daily_series: pd.Series = pd.Series(daily_data_dict)

    # The index is currently string dates ("19990101"). Convert them to proper Datetimes
    daily_series.index = pd.to_datetime(daily_series.index, format="%Y%m%d")

    # Convert to DataFrame for easier manipulation
    daily_df: pd.DataFrame = daily_series.reset_index()
    daily_df.columns = ["date", "RH2M"]

    # Note: The API returns -999.0 for missing values.
    # We should replace those with NaN (Not a Number)
    daily_df["RH2M"] = daily_df["RH2M"].replace(-999.0, pd.NA)

    n_missing = daily_df["RH2M"].isna().sum()
    if n_missing > 0:
        # With the 2-month lookback, missing values shouldn't occur.
        # If they do, something has changed upstream.
        print(daily_df[daily_df["RH2M"].isna()])

        raise ValueError(
            f"NASA POWER returned {n_missing} missing values (-999.0)."
            f"Aborting to avoid silently corrupted aggregation."
        )

    return daily_df


def aggregate_to_monthly(daily_df: pd.DataFrame) -> pd.Series:
    """Aggregate daily RH2M to monthly means and convert to proportion."""
    df = daily_df.copy()

    df["RH2M"] = df["RH2M"] / 100
    df_indexed = df.set_index("date")
    monthly_df = df_indexed.resample("ME").mean().reset_index()

    # Convert the monthly_data_df into a Pandas Series
    monthly_series: pd.Series = pd.Series(
        data=monthly_df["RH2M"].values, index=monthly_df["date"]
    )

    return monthly_series


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # 2. Define coordinates, start and end of the time series
    # ---------------------------------------------------------------------------------
    # Define coordinates for Brasília, Brazil
    latitude: float = -15.7797
    longitude: float = -47.9257

    # NASA POWER hourly data begins 1981;
    # We use 1999 to match the published Brasilia series.
    START_DATE: str = "19990101"

    # Empirically, NASA POWER returns NaN values for the most recent ~2 months;
    # the root cause (processing lag, caching, sensor delay?) is not documented
    # in the API reference. Capping the end date at today minus 2 months avoids
    # the issue. Revisit if the API behavior changes.

    today = pd.Timestamp.today()
    target_end: pd.Timestamp = today - pd.offsets.MonthEnd(2)
    END_DATE: str = target_end.strftime("%Y%m%d")

    # ---------------------------------------------------------------------------------
    # 3. Use the function to get the data
    # ---------------------------------------------------------------------------------

    api_params: dict[str, str | float] = build_api_params(
        latitude=latitude, longitude=longitude, start=START_DATE, end=END_DATE
    )

    raw_data_dict: dict = fetch_nasa_humidity(api_params=api_params)
    daily_df: pd.DataFrame = json_to_daily_df(raw_data_dict)
    monthly_series: pd.Series = aggregate_to_monthly(daily_df)

    # ---------------------------------------------------------------------------------
    # 4. Export raw and processed data
    # ---------------------------------------------------------------------------------
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Export the raw data in json format
    with open(RAW_DIR / "raw_data_dict.json", "w") as f:
        json.dump(raw_data_dict, f, indent=2)

    monthly_series.to_csv(
        PROCESSED_DIR / "y_full_monthly_data.csv", index=True, index_label="Month"
    )
