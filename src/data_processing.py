# ==========================================================
# Data preprocessing script
# ==========================================================
# DATA COLUMN LIST:
# Monthly series from raw_data.csv
# X1 inflation / base index
# X2 USD-yen exchange rate / raw series
# X3 chained PPI / in [YoY % change]
# X4 PPI index / base index
# X5 SPPI / in [YoY % change]
# X6 import price index / base index
# X7 import energy price index / base index
# X8 Nikkei 225 index / raw series
# X9 unemployment rate / in [%]
# X10 HH income / in [YoY % change]
# X11 consumption activity index / base index
# X12 shadow interest rate / in [%]
# X13 loan size / raw series
# X14 Industrial production index / base index
# X15 monetary base / raw series

# Monthly but from other files
# X16 tourist arrivals / raw series
# X17 ESP inflation expectation / in [%]

# Quarterly series from raw_data_quarter.csv
# X18 Tankan output price change / net percentage responses
# X19 Tankan input price change / net percentage responses
# X20 real GDP / raw series
# X21 BOJ output gap / in [%]
# X22 HH disposable income / in [YoY % change]
# X23 government consumption / raw series
# X24 Government investment / raw series

# ==========================================================

import pathlib
import numpy as np
import pandas as pd

#import sys
#sys.path.append("../")

# ==========================================================
# X16 Tourist arrivals data part
# Merge tourist arrivals by countries to get total arrivals
# ==========================================================
# define paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "my_config.yaml"

tourist_data_file = ROOT / "data" / "tourist.csv"
raw_tourist_data = pd.read_csv(tourist_data_file, parse_dates=False).fillna(0)
raw_tourist_data = raw_tourist_data.drop(columns=["Growth Rate(%)"])
# rename arrivals column
raw_tourist_data = raw_tourist_data.rename(columns={"Visitor Arrivals": "arrivals"})

# in the raw file, month column has a dot at the end (abbrev.)
raw_tourist_data["Month_clean"] = raw_tourist_data["Month (abbr)"].str.replace(".", "", regex=False)

raw_tourist_data["Date"] = pd.to_datetime(
    raw_tourist_data["Year"].astype(str) + "-" + raw_tourist_data["Month_clean"],
    format="%Y-%b"
)
raw_tourist_data = raw_tourist_data.set_index("Date").sort_index()

# clean arrivals column
raw_tourist_data["arrivals"] = (
    raw_tourist_data["arrivals"]
      .astype(str)
      .str.replace(",", "", regex=False)
      .str.strip()
)
raw_tourist_data["arrivals"] = pd.to_numeric(raw_tourist_data["arrivals"], errors="coerce").fillna(0)

# group by date and sum over countries, save to new csv
grouped_tourist_data = raw_tourist_data.groupby(level="Date")["arrivals"].sum()
grouped_tourist_data.index.name = "Date"
grouped_tourist_data.to_csv(ROOT / "data" / "processed_tourist_arrivals.csv")

# ==========================================================
# X17 ESP series from another file
# ==========================================================
esp_data_file = ROOT / "data" / "ESP_inflation_exp.csv"
esp_data = pd.read_csv(esp_data_file) # no index column
# raw dataframe has a year column and a month column: form into Date index
esp_data["Date"] = pd.to_datetime(
    esp_data["year"].astype(str) + "-" + esp_data["month"].astype(str),
    format="%Y-%m"
)
esp_data = esp_data.set_index("Date").sort_index().drop(columns=["year", "month"])

# ==========================================================
# Process all other monthly data series
# ==========================================================
data_file = ROOT / "data" / "raw_data.csv"
raw_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
raw_data.index = pd.to_datetime(raw_data.index, format="%b-%y")
raw_data.index.name = "Date"

# add tourist arrivals series
raw_data = raw_data.join(grouped_tourist_data.rename("X16"), how="left")
raw_data["X16"] = raw_data["X16"].fillna(0)

# add ESP inflation expectation series
raw_data = raw_data.join(esp_data, how="left")
raw_data["X17"] = raw_data["X17"].fillna(0)

# ==========================================================
# Add all other quarterly data series
# ==========================================================
# read quarterly data
data_file = ROOT / "data" / "raw_data_quarter.csv"
raw_data_quarter = pd.read_csv(data_file, index_col=0)

raw_data_quarter.index = ( # deal with the quarter date index
    pd.to_datetime(raw_data_quarter.index, format="%b-%y")
    .to_period('Q')
    .to_timestamp(how='end')
    .normalize() # set time to midnight
    .map(lambda ts: ts.replace(day=1)) # set to first day of last month of quarter
)
raw_data_quarter.index.name = "Date"

# join quarterly data to monthly data by forward filling
# rule: for the first two months of the quarter, use last quarter's data
# for the last month of the quarter, use current quarter's data
# this is because of the data release timing
monthly_idx = pd.date_range(
    start=raw_data_quarter.index.min(),
    end=raw_data_quarter.index.max(),
    freq='MS' # Month Start frequency
)
q_to_m = (
    raw_data_quarter
    .reindex(monthly_idx, method="ffill")
)

# join to raw_data
raw_data = raw_data.join(q_to_m, how="left")
# now the data is complete

# ==========================================================
# Convert series into unified formats
# ==========================================================
# if in YoY % change format, good to go
# otherwise, log_diff the series to convert to YoY % change and X100 for percentage format

# lists of column names for different processing
list_good_series = ['X3', 'X5', 'X9', 'X10', 'X12', "X17", "X18", "X19", "X21", "X22"] # ready-to-go series
list_need_series = [col for col in raw_data.columns if col not in list_good_series]

# process each series accordingly
processed_data = pd.DataFrame(index=raw_data.index)
for col in raw_data.columns:
    if col in list_need_series:
        # need to log_diff and X100
        series = pd.to_numeric(raw_data[col], errors="coerce")
        series = series.where(series > 0) # avoid log(0) problem
        series_transformed = series.apply(lambda x: np.log(x)).diff(12) * 100.0
        processed_data[col] = series_transformed
    else:
        # good to go
        series_transformed = raw_data[col]
        processed_data[col] = series_transformed

# ==== save processed data ====
processed_data_file = ROOT / "data" / "processed_data.csv"
processed_data.to_csv(processed_data_file)