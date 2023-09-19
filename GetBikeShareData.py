import os
import pandas as pd
from datetime import datetime

current_wd_path = os.getcwd()
data_wd_path = f"{current_wd_path}/DC BikeShare Data"


def breakdown_datetime(df):
    """
    Add breakdowns of date&hour as features
    :param df: dataset for forecasting
    :return: the dataset appended time component features
    """

    df['year'] = df.index.year
    df['season'] = pd.cut(x=df.index.month, bins=[1, 2, 5, 8, 11, 12],
                          labels=['winter', 'spring', 'summer', 'fall', 'winter'], ordered=False,
                          include_lowest=True)
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['day_of_year'] = df.index.dayofyear
    df['hour'] = df.index.hour

    return df


def add_features(df):
    """
    Add holidays and active station count columns
    :param df: dateset for forecasting
    :return: the dataset appended features
    """
    import holidays
    us_dc_holidays = holidays.country_holidays('US', subdiv='DC',
                                               years=[y for y in range(min(df['year']), max(df['year'])+1, 1)])

    us_dc_holidays_df = pd.DataFrame(us_dc_holidays.items(), columns=['date', 'holiday'])
    us_dc_holidays_df = us_dc_holidays_df.sort_values(by='date').reset_index(drop=True)
    us_dc_holidays_df['date'] = pd.to_datetime(us_dc_holidays_df['date'])

    df['date'] = pd.to_datetime(df['date'])
    df['is_holiday'] = df.merge(
        us_dc_holidays_df,
        on='date',
        how='left'
    )['holiday'].apply(lambda x: 0 if pd.isna(x) else 1).values

    active_station_df = pd.read_csv(f"{data_wd_path}/active_station.csv")
    active_station_df['date'] = pd.to_datetime(active_station_df['date'])

    df['active_station_cnt'] = df.merge(
        active_station_df,
        on='date',
        how='left'
    )['active_station_cnt'].values

    return df


def get_usage_data(year_month):
    import requests
    import zipfile
    import io

    """
    Fetch bike share usage log from
    :param year_month:
    :return:
    """
    DC_BIKESHARE_USAGE_API = f"https://s3.amazonaws.com/capitalbikeshare-data/{year_month}-capitalbikeshare-tripdata.zip"
    CSV_FILE = f"{year_month}-capitalbikeshare-tripdata.csv"

    print("API request")
    response = requests.get(DC_BIKESHARE_USAGE_API)
    zip_content = io.BytesIO(response.content)

    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        csv_file = zip_ref.open(CSV_FILE)
        df = pd.read_csv(csv_file)

    def summarize_data(df):
        for time_col in df.filter(regex='ed_at').columns:
            df[time_col] = pd.to_datetime(df[time_col])

        df['duration'] = df.apply(lambda row: (row['ended_at'] - row['started_at']).seconds, axis=1)
        df = df.loc[df['duration'] >= 60].reset_index(drop=True)

        df['start_date_hour'] = df['started_at'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:00:00'))

        summarized_df = df.groupby('start_date_hour').size().to_frame('ride_cnt').reset_index()
        summarized_df.set_index('start_date_hour', inplace=True)
        summarized_df.index = pd.to_datetime(summarized_df.index)

        summarized_df = summarized_df.asfreq(freq='h', fill_value=0)

        summarized_df['date'] = summarized_df.index.date

        return summarized_df

    bike_usage_df = summarize_data(df)

    bike_usage_df = breakdown_datetime(bike_usage_df)

    bike_usage_df = add_features(bike_usage_df)

    bike_usage_df = bike_usage_df.reset_index()

    return bike_usage_df
