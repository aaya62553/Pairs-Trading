

from datetime import datetime
import os
import pandas as pd
import pandas_market_calendars as mcal

def get_data_from_cache(ticker, start_date, end_date, cache_dir="data_cache"):
  
    if not os.path.isdir(cache_dir):
        return None
    target_start = int(start_date.year)
    target_end = int(end_date.year)
    best_file = None
    best_span = None
    # Expected filename format: <TICKER>_<START_YEAR>_<END_YEAR>.csv
    for filename in sorted(os.listdir(cache_dir)):
        if not filename.endswith(".csv"):
            continue
        stem = os.path.splitext(filename)[0]
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        file_ticker, start_year_str, end_year_str = parts
        if file_ticker != ticker:
            continue
        try:
            start_year = int(start_year_str)
            end_year = int(end_year_str)
        except ValueError:
            continue
        if start_year <= target_start and target_end <= end_year:
            span = end_year - start_year
            if best_span is None or span < best_span:
                best_span = span
                best_file = filename
    if best_file is None:
        return None
    data = pd.read_csv(
        os.path.join(cache_dir, best_file),
        index_col=0,
        parse_dates=True
    )
    if data.empty:
        return None
    data.index = pd.to_datetime(data.index, utc=True)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    data = data[(data.index >= start_ts) & (data.index <= end_ts)]
    return data if not data.empty else None
  
def get_universe_by_period(universe, formation_period, trading_period, start_date, end_date=datetime(2025,1,1)):
    filtered_universe = []
    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index  # jours ouverts bourse
    periods = []

    # Build non-overlapping windows with exact sizes:
    # [period_start ... formation_end] then [trading_start ... trading_end]
    max_start_idx = len(trading_days) - formation_period - trading_period
    if max_start_idx < 0:
        return filtered_universe

    for i in range(0, max_start_idx + 1, trading_period):
        period_start = trading_days[i]
        formation_end = trading_days[i + formation_period - 1]
        trading_start = trading_days[i + formation_period]
        trading_end = trading_days[i + formation_period + trading_period - 1]
        periods.append([period_start, formation_end, trading_start, trading_end])
    
    for period_start, formation_end, trading_start, trading_end in periods:
        dict_period = {
            "period_start": period_start,
            "formation_end": formation_end,
            "trading_start": trading_start,
            "trading_end": trading_end,
        }
        for sector in universe["GICS Sector"].unique():
            sector_tickers = universe[(universe["start"] <= period_start) & (universe["end"] >= trading_end) & (universe["GICS Sector"] == sector)]["ticker"].unique()
            if len(sector_tickers) >= 2:
                dict_period[sector] = sector_tickers.tolist()

        filtered_universe.append(dict_period)

    return filtered_universe