import os
from dotenv import dotenv_values

print('Get configuration')
config = {
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}


def get_station_data():
    import pandas as pd
    from datetime import datetime
    import requests

    print('API request')
    response = requests.get(config['DC_BIKESHARE_STATION_API'])

    if response.status_code == 200:
        print('Successfully responded')
        updated_time = datetime.fromtimestamp(response.json()['last_updated'])
        data_list_df = pd.DataFrame(response.json()['data']['en']['feeds'])

        # print('Get general information about stations')
        # station_info_url = data_list_df.loc[data_list_df['name']=='station_information', 'url'].values[0]
        # station_info = requests.get(station_info_url).json()
        # station_info_df = pd.DataFrame(station_info['data']['stations'])
        # print('General information about stations is loaded as station_info_df.')

        print('Get current status of stations')
        station_stat_url = data_list_df.loc[data_list_df['name'] == 'station_status', 'url'].values[0]
        station_stat = requests.get(station_stat_url).json()
        station_stat_df = pd.DataFrame(station_stat['data']['stations'])
        print('Current status of stations is loaded as station_stat_df.')

        if len(station_stat_df) > 0:
            active_station_cnt = station_stat_df.loc[station_stat_df['is_installed'].isin([1, True]), 'station_id'].nunique()

            active_station_df = pd.DataFrame({
                'date': [pd.to_datetime(datetime.strftime(updated_time, '%Y-%m-%d'))],
                'update_time': [pd.to_datetime(updated_time)],
                'active_station_cnt': active_station_cnt
            })

            return updated_time, active_station_df

    else:
        print(f"API request failed. Code: {response.status_code}")
