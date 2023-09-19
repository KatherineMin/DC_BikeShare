import os
import pandas as pd
from dotenv import dotenv_values

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pytz import timezone

import GetStationData
import GetBikeShareData

config = {
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}
current_wd_path = os.getcwd()
data_wd_path = f"{current_wd_path}/DC BikeShare Data"
today_date = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d")
today_date = pd.to_datetime(today_date)


def update_station_data():
    print("Fetch past station data")
    retrieved_station_df = pd.read_csv(f"{data_wd_path}/active_station.csv")

    for date_col in retrieved_station_df.filter(regex='date').columns:
        retrieved_station_df[date_col] = pd.to_datetime(retrieved_station_df[date_col])

    print("Fetch today's station data")
    updated_time, active_station_df = GetStationData.get_station_data()

    if pd.to_datetime(datetime.strftime(updated_time, '%Y-%m-%d')) >= today_date:
        print(f"Update station data at {updated_time}")
        retrieved_station_df = pd.concat([retrieved_station_df, active_station_df], ignore_index=True)
        retrieved_station_df.to_csv(f"{data_wd_path}/active_station.csv", index=None)


def fetch_past_usage_data():
    print("Fetch past usage data")
    retrieved_usage_df = pd.read_csv(f"{data_wd_path}/bike_share_log.csv")

    for date_col in retrieved_usage_df.filter(regex='date').columns:
        retrieved_usage_df[date_col] = pd.to_datetime(retrieved_usage_df[date_col])

    return retrieved_usage_df


def prepare_usage_data():

    retrieved_usage_df = fetch_past_usage_data()
    max_date = retrieved_usage_df['date'].max()

    today_date = datetime.now(tz=timezone("US/Eastern")).strftime("%Y-%m-%d")

    year_month_list = [f"{date.year}{date.month}" if date.month > 9 else f"{date.year}0{date.month}" for date in pd.date_range(pd.to_datetime(max_date) + relativedelta(months=1), pd.to_datetime(today_date), freq='M')]

    has_new = 0

    if len(year_month_list) > 0:

        for ym in year_month_list:
            print(ym)
            try:
                new_data = GetBikeShareData.get_usage_data(ym)
                new_data = new_data.reset_index()
                retrieved_usage_df = pd.concat([retrieved_usage_df, new_data], axis=0, ignore_index=True)
                has_new += 1
            except:
                print(f"Data of {ym} is not downloadable yet.")

    if has_new > 0:
        retrieved_usage_df.to_csv(f"{data_wd_path}/bike_share_log.csv", index=None)

    return has_new, retrieved_usage_df
