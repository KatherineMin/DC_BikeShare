import os
from dotenv import dotenv_values
import warnings

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

print('Get a current working directory')
current_wd_path = os.getcwd()
print(f'Current working directory is: {current_wd_path}')

print('Import datasets')
data_v1_raw = pd.read_csv(f'{current_wd_path}/DC BikeShare Data/bks_tripdata_v1.csv', low_memory=False)
data_v2_raw = pd.read_csv(f'{current_wd_path}/DC BikeShare Data/bks_tripdata_v2.csv', low_memory=False)

# Quick check on imported datasets
print(f'Size of data_v1_raw: {data_v1_raw.shape}')
print(f'Size of data_v2_raw: {data_v2_raw.shape}')

print(f'Columns of data_v1_raw:\n{data_v1_raw.columns}\n\nColumns of data_v2_raw:\n{data_v2_raw.columns}')

print(f'Data type of data_v1_raw:\n{data_v1_raw.dtypes}\n\nData type of data_v2_raw:\n{data_v2_raw.dtypes}')

print(f'Nulls in data_v1_raw:\n{data_v1_raw.isna().sum()}\n\nNulls in data_v2_raw:\n{data_v2_raw.isna().sum()}')

print(f'Snippet of data_v1_raw\n{data_v1_raw.head(3)}')
print(f'Snippet of data_v2_raw\n{data_v2_raw.head(3)}')


# Rearrange values in the datasets
print('Convert date string to datetime')

print(f"Convert {data_v1_raw.filter(regex='_date').columns} in data_v1_raw")
for date_col in data_v1_raw.filter(regex='_date').columns:
    data_v1_raw[date_col] = pd.to_datetime(data_v1_raw[date_col])

print(f"Convert {data_v2_raw.filter(regex='ed_at').columns} in data_v2_raw")
for date_col in data_v2_raw.filter(regex='ed_at').columns:
    data_v2_raw[date_col] = pd.to_datetime(data_v2_raw[date_col])

data_v1_raw['member_type'] = data_v1_raw['member_type'].apply(lambda x: x.lower())

# Make two datasets compatible
print("Rename some columns of data_v1_raw")
data_v1_raw.rename(columns={
    'start_date': 'started_at', 'end_date': 'ended_at',
    'start_station_number': 'start_station_id', 'end_station_number': 'end_station_id',
    'start_station': 'start_station_name', 'end_station': 'end_station_name'
}, inplace=True)

print('Calculate duration of travel of data_v2_raw')
data_v2_raw['duration'] = data_v2_raw.apply(lambda row: (row['ended_at'] - row['started_at']).seconds, axis=1)
print("Rename some columns of data_v2_raw")
data_v2_raw.rename(columns={'member_casual': 'member_type'}, inplace=True)
print("Append a bike_number column to data_v2_raw")
data_v2_raw['bike_number'] = None

# Combine two datasets
print('List columns for the combined datasets')
combined_cols = ['ride_id', 'started_at', 'ended_at', 'duration', 'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_type', 'rideable_type', 'bike_number']

print("Arrange columns of the datasets before merging")
data_v1_raw[list(set(combined_cols) - set(data_v1_raw.columns))] = None
data_v1_raw = data_v1_raw[combined_cols]
data_v2_raw = data_v2_raw[combined_cols]

print("Combine two datasets")
data_raw = pd.concat([data_v1_raw, data_v2_raw], axis=0).reset_index(drop=True)
print(f'Size of data_raw: {data_raw.shape}')

# Remove error rows following the guide on the official website
print('Remove test trips')
data_raw = data_raw.loc[data_raw['duration'] >= 60].reset_index(drop=True)
print(f'Size of data_raw after removal: {data_raw.shape}')

# Prepare Time Series data for prediction
print("Extract date and hour from datetime of departure")
data_raw['start_date_hour'] = data_raw['started_at'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:00:00'))
data_raw['start_date'] = data_raw['started_at'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))

print('Summarize a dataset into date & hour')
date_hour_df = data_raw.groupby('start_date_hour').size().to_frame('ride_cnt').reset_index()
date_hour_df = date_hour_df.merge(
    data_raw.groupby('start_date_hour').agg(
        sum_duration=pd.NamedAgg('duration', 'sum'),
        # sum_distance=pd.NamedAgg('distance', 'sum'),
        avg_duration=pd.NamedAgg('duration', 'mean'),
        # avg_distance=pd.NamedAgg('distance', 'mean')
    ).reset_index(),
    on='start_date_hour',
    how='left'
)

date_df = data_raw.groupby('start_date').size().to_frame('ride_cnt').reset_index()

print("Set date & hour column as an index of the dataset")
date_hour_df.set_index('start_date_hour', inplace=True)
date_hour_df.index = pd.to_datetime(date_hour_df.index)
date_hour_df = date_hour_df.asfreq(freq='h', fill_value=0)

date_df.set_index('start_date', inplace=True)
date_df.index = pd.to_datetime(date_df.index)
date_df = date_df.asfreq(freq='D', fill_value=0)

date_hour_df['ride_cnt'].plot(style='.', markersize=.8, title='Hourly Ride Counts', figsize=(15, 5))
plt.xlabel('Date Time')
plt.ylabel('Ride Counts')
plt.tight_layout()

date_df['ride_cnt'].plot(style='.', markersize=2, title='Daily Ride Counts', figsize=(15, 5), color='tab:purple')
plt.xlabel('Date')
plt.ylabel('Ride Counts')
plt.tight_layout()


# Create features

def breakdown_datetime(df):
    """
    Add breakdowns of date&hour as features
    :param df: dataset for forecasting
    :return: the dataset appended time component features
    """

    df['year'] = df.index.year
    df['season'] = pd.cut(x=df.index.month, bins=[1, 2, 5, 8, 11, 12],
                          labels=['winter', 'spring', 'summer', 'fall', 'winter'], ordered=False, include_lowest=True)
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['day_of_year'] = df.index.dayofyear
    df['hour'] = df.index.hour

    return df


date_hour_df = breakdown_datetime(date_hour_df)

print("Create features - add weather data")
print("Import weather datasets")
weather_v1_df = pd.read_csv(f'{current_wd_path}/DC Bikeshare Data/weather_v1.csv', low_memory=False)
weather_v2_df = pd.read_csv(f'{current_wd_path}/DC Bikeshare Data/weather_v2.csv', low_memory=False)

weather_df = pd.concat([weather_v1_df, weather_v2_df], axis=0).reset_index(drop=True)

print("Convert date columns of weather_df into date & hour")
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
weather_df['DATE'] = weather_df['DATE'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:00:00'))


filtered_weather_df = weather_df.groupby('DATE').size().to_frame('cnt').reset_index()

print('Summarize weather_df into date & hour')
for col in [col for col in weather_df.columns if col != 'DATE']:
    print(col)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            filtered_weather_df = filtered_weather_df.merge(
                weather_df.groupby('DATE')[col].apply(lambda x: np.nanmean(x)).reset_index(),
                on='DATE',
                how='left'
            )
    except:
        filtered_weather_df = filtered_weather_df.merge(
            weather_df.groupby('DATE')[col].sum().reset_index(),
            on='DATE',
            how='left'
        )

# filtered_weather_df['DATE'] = pd.to_datetime(filtered_weather_df['DATE'])
# filtered_weather_df.set_index('DATE', inplace=True)

# Transform some data values
def transform_hourlyWeatherType(df):
    col_name = 'HourlyPresentWeatherType'
    df = df.copy()

    print(f"Transform {col_name} from {df}")
    df.loc[df[col_name] != 0, col_name] = df.loc[df[col_name] != 0, col_name].apply(lambda x: x.replace('||', '|'))
    df.loc[df[col_name] != 0, col_name] = df.loc[df[col_name] != 0, col_name].apply(lambda x: x.replace(' ', ''))
    df.loc[df[col_name] != 0, col_name] = df.loc[df[col_name] != 0, col_name].apply(lambda x: re.sub(r'[0-9a-z+:|-]', ' ', x))
    df.loc[df[col_name] != 0, col_name] = df.loc[df[col_name] != 0, col_name].apply(lambda x: ' '.join(x.split()).split())
    df.loc[df[col_name] != 0, col_name] = df.loc[df[col_name] != 0, col_name].apply(lambda x: list(set(x)))

    df.loc[df[col_name] != 0, col_name] = df.loc[df[col_name] != 0, f"{col_name}_cnt"] = df.loc[df[col_name] != 0, col_name].apply(lambda x: len(x))
    df[f"{col_name}_cnt"] = df[f"{col_name}_cnt"].fillna(0)

    return df


filtered_weather_df = transform_hourlyWeatherType(filtered_weather_df)

# weather_code_map = {
#     'HZ': 'Haze', 'FU': 'Smoke', 'DU': 'Dust', 'BR': 'Mist', 'SQ': 'Squalls', 'FG': 'Fog', 'DZ': 'Drizzle', 'FZDZ': 'Freezing Drizzle', 'RA': 'Rain', 'FZRA': 'Freezing Rain',
#     'SN': 'Snow', 'PL': 'Ice Pellet', 'SG': 'Snow Grains', 'IC': 'Ice Crystals', 'SHRA': 'Rain Showers', 'SHSN': 'Snow Showers', 'HAIL': 'Hail', 'TS': 'Thunderstorm', 'TS+HAIL': 'Thunderstorm Hail', 'FC': 'Tornado','GR': 'Hail', 'GS': 'Small Hail', 'UP': 'Unknown Precipitation', 'VA': 'Volcanic Ash', 'SA': 'Sand', 'PY': 'Spray', 'FC': 'Funnel Cloud', 'SS': 'Sandstorm', 'DS': 'Duststorm', 'DRSN': 'Drifting Snow', 'BLSN': 'Blowing Snow', 'SHRASN': 'Shower of Rain and Snow', 'WIND': 'Wind', 'BLPY': 'Blowing Spray', 'GL': 'Glaze', 'MIFG': 'Ground Fog', 'FZFG': 'Freezing Fog'
# }


def transform_hourlyPrecipitation(df):
    col_name = 'HourlyPrecipitation'
    df = df.copy()

    print(f"Transform {col_name} from {df}")
    df.loc[~df[col_name].isna(), col_name] = df.loc[~df[col_name].isna(), col_name].apply(lambda x: re.sub(r'[a-z,A-Z]', '', str(x)))

    def untie_float(x):
        try:
            return float(str(x))
        except:
            numbers = re.findall(r'[0-9]{1}.[0-9]{2}', str(x))
            return sum([untie_float(n) for n in numbers])

    df.loc[~df[col_name].isna(), f"{col_name}_check"] = df.loc[~df[col_name].isna(), col_name].apply(lambda x: untie_float(x))
    df[f"{col_name}_check"] = df[f"{col_name}_check"].fillna(0)

    return df


filtered_weather_df = transform_hourlyPrecipitation(filtered_weather_df)

print("Summarize hourly weather data values")
filtered_weather_df['hourly_summary'] = filtered_weather_df.apply(lambda row: row['HourlyPresentWeatherType_cnt'] + 1 if row['HourlyPrecipitation_check'] > 0 else row['HourlyPresentWeatherType_cnt'], axis=1)


# Get Holiday data
us_dc_holidays = holidays.country_holidays('US', subdiv='DC', years=[y for y in range(min(date_hour_df['year']), max(date_hour_df['year'])+1, 1)])

us_dc_holidays_df = pd.DataFrame(us_dc_holidays.items(), columns=['date', 'holiday'])
us_dc_holidays_df = us_dc_holidays_df.sort_values(by='date').reset_index(drop=True)

us_dc_holidays_df['date'] = pd.to_datetime(us_dc_holidays_df['date'])


# Add weather & holiday components to date_hour_df
date_hour_df['date'] = pd.to_datetime(date_hour_df.index.date)

print("Create features - Add holiday factor to date_hour_df")
date_hour_df['is_holiday'] = date_hour_df.merge(
    us_dc_holidays_df,
    on='date',
    how='left'
)['holiday'].apply(lambda x: 0 if pd.isna(x) else 1).values

print("List weather related columns inclusive of date_hour_df")
weather_cols = ['HourlyPresentWeatherType', 'HourlyPresentWeatherType_cnt', 'HourlyPrecipitation', 'HourlyPrecipitation_check', 'hourly_summary']

print("Create features - Add weather factors to date_hour_df")
date_hour_df['weather_summary'] = date_hour_df.merge(
    filtered_weather_df[weather_cols],
    left_index=True,
    right_index=True,
    how='left'
)['hourly_summary'].apply(lambda x: 1 if x > 0 else 0).values

print("Create features - Add the number of available stations factor")
station_avail_df = pd.read_csv(f'{current_wd_path}/DC BikeShare Data/station_avail_df.csv')
station_avail_df['date'] = pd.to_datetime(station_avail_df['date'])

date_hour_df['active_station_cnt'] = date_hour_df.merge(
    station_avail_df[['date', 'adj_station_cnt']].rename(columns={'adj_station_cnt': 'active_station_cnt'}),
    on='date',
    how='left'
)['active_station_cnt'].values

date_hour_df = pd.get_dummies(date_hour_df).replace(to_replace=[True, False], value=[1, 0])

print("Start prediction")
print("Declare a target variable")
TARGET = 'ride_cnt'


def transform_target(df, target=TARGET):
    """
    Log transform to prevent negative predictive values
    :param df: dataset for forecasting
    :param target: column name of target variable in the dataset
    :return: the dataset with a column that contains log transformed target value
    """
    alpha = 0.001

    df[f"log_{target}"] = np.log(df[target]+alpha)

    return df


date_hour_df = transform_target(date_hour_df)

print("Reassign the target variable along with transform")
TARGET = f"log_{TARGET}"


# Create features - Add lagged target values to date_hour_df
def add_lags(df):
    """
    Add lagged values of the target variable to dataset
    :param df: dataset for forecasting
    :return: the dataset that has lagged values appended
    """
    date_hour_map = df[TARGET].to_dict()

    # Bring 8-week-ago target value
    df['lag_8w'] = (df.index - timedelta(days=7*8)).map(date_hour_map)  ## df['lag_8w'] = df[TARGET].shift(24*7*8)
    # Bring 1-year-ago target value
    df['lag_1y'] = (df.index - timedelta(days=7*52)).map(date_hour_map)  ## df['lag_1y'] = df[TARGET].shift(24*7*52)
    # Bring 2-year-ago target value
    df['lag_2y'] = (df.index - timedelta(days=7*52*2)).map(date_hour_map)  ## df['lag_2y'] = df[TARGET].shift(24*7*52*2)

    return df


date_hour_df = add_lags(date_hour_df)

print("List up input variables")
FEATURES = ['year', 'season_fall', 'season_spring', 'season_summer', 'season_winter', 'month', 'weekday', 'day_of_year', 'hour', 'is_holiday', 'weather_summary', 'ma_28', 'lag_8w', 'lag_1y', 'lag_2y']
FEATURES.remove('weather_summary')
date_hour_df.head()
FEATURES = ['year', 'season_fall', 'season_spring', 'season_summer', 'season_winter', 'month', 'weekday', 'day_of_year', 'hour', 'is_holiday', 'active_station_cnt', 'lag_8w', 'lag_1y', 'lag_2y']

# FEATURES.remove('ma_28')
# FEATURES.append('ma_7')


def evaluate_model(y_actual, y_pred):
    """
    Evaluate performance of model prediction by MAE, RMSE, MAPE, and MDAPE
    :param y_actual: Actual values of the target variable
    :param y_pred: Predicted target values
    :return: mae, rmse, mape, mdape
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_actual, y_pred)
    print(f"MAE: {mae}")

    # Calculate Root Mean Squared Error (RMSE)
    rmse = mean_squared_error(y_actual, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # Calculate Mean Absolute Percent Error (MAPE)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    print(f"MAPE: {mape:.2f}%")

    mdape = np.median(np.abs((y_actual - y_pred) / y_actual)) * 100
    print(f"MDAPE: {mdape:.2f}%")

    return mae, rmse, mape, mdape


print("Prediction - Hyperparameter tuning")

print("Cross Validation with Default setting")
tscv = TimeSeriesSplit(n_splits=5, test_size=24*28*2, gap=24)

for i, (train_idx, test_idx) in enumerate(tscv.split(date_hour_df)):
    train_df = date_hour_df.iloc[train_idx]
    test_df = date_hour_df.iloc[test_idx]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    xgb_reg = xgb.XGBRegressor(seed=704, booster='gbtree')
    xgb_reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)
    print(xgb_reg.get_params)

    y_pred = xgb_reg.predict(X_test)
    evaluate_model(y_test, y_pred)


print("Split train and test data")
train_df = date_hour_df.loc[date_hour_df.index < datetime(2023, 1, 1)]
test_df = date_hour_df.loc[date_hour_df.index >= datetime(2023, 1, 1)]

print(f"Target variable is {TARGET}.\nInput variables are {FEATURES}")
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

preds = {}
evaluations = {}

xgb_reg = xgb.XGBRegressor(seed=704, booster='gbtree')
xgb_reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

y_pred = xgb_reg.predict(X_test)
mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)

preds['Default'] = {'params': xgb_reg.get_params(),
                    'actuals': y_test.values,
                    'preds': y_pred}

evaluations['Default'] = {'params': xgb_reg.get_params(),
                          'MAE': mae,
                          'RMSE': rmse,
                          'MAPE': mape,
                          'MDAPE': mdape}


print("Hyperparameter Tuning with Grid Search CV")
tscv = TimeSeriesSplit(n_splits=5, test_size=24*28*2, gap=24)
tscv_idx = []
for i, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
    tscv_idx.append((train_idx, test_idx))

print("List up parameters to be considered")
params = {
    'n_estimators': [100, 500, 1000, 1500],
    'max_depth': np.arange(3, 13, 2),
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.5, 0.8, 1],
    'reg_alpha': [0.1, 1, 5, 10]
}

print("Hyperparameter Tuning with Grid Search CV")
xgb_reg = xgb.XGBRegressor(seed=704, booster='gbtree')
grid_search = GridSearchCV(estimator=xgb_reg,
                           param_distributions=params,
                           scoring='neg_mean_squared_error',
                           early_stopping_rounds=10,
                           n_iter=10,
                           cv=tscv_idx)

# Get the parameter combinations
combinations = grid_search.cv_results_['params']

# Get the mean test scores
mean_scores = grid_search.cv_results_['mean_test_score']

# Get the standard deviation of test scores
std_scores = grid_search.cv_results_['std_test_score']

# Iterate over the parameter combinations and their scores
grid_search_cv_df = pd.DataFrame(columns=['combination', 'mean_score', 'std_score'])
for i, (combination, mean_score, std_score) in enumerate(zip(combinations, mean_scores, std_scores)):
    grid_search_cv_df.loc[i] = [combination, mean_score, std_score]
    print(f"Combination: {combination}, Mean Score: {mean_score}, Std Score: {std_score}")

grid_search_cv_df.sort_values(by='mean_score', ascending=False)
# Get the best estimator and its corresponding hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

best_params
print(TARGET)
xgb_reg = xgb.XGBRegressor(seed=704, booster='gbtree',
                           learning_rate=0.01, max_depth=9, n_estimators=1000, reg_alpha=0.1, subsample=0.5)
xgb_reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

y_pred = xgb_reg.predict(X_test)
mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)

preds['GridSearchCV'] = {'params': xgb_reg.get_params(),
                         'actuals': y_test.values,
                         'preds': y_pred}

evaluations['GridSearchCV'] = {'params': xgb_reg.get_params(),
                               'MAE': mae,
                               'RMSE': rmse,
                               'MAPE': mape,
                               'MDAPE': mdape}

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)

preds['GridSearchCV'] = {'params': best_model.get_params(),
                         'actuals': y_test.values,
                         'preds': y_pred}

evaluations['GridSearchCV'] = {'params': best_model.get_params(),
                               'MAE': mae,
                               'RMSE': rmse,
                               'MAPE': mape,
                               'MDAPE': mdape}

# print("Reduce lists of hyperparameter for Grid Search CV based on result of Random Search CV")
# print(random_search_cv_df.sort_values(by='mean_score', ascending=False)['combination'].values)


print("Check feature importance")
fig, ax = plt.subplots(1, 2)
pd.DataFrame(data=xgb_reg.feature_importances_,
             index=xgb_reg.feature_names_in_,
             columns=['importance']).sort_values('importance').plot(kind='barh', title='Default - Feature Importance', ax=ax[0], color='tab:blue')
pd.DataFrame(data=grid_search.best_estimator_.feature_importances_,
             index=grid_search.best_estimator_.feature_names_in_,
             columns=['importance']).sort_values('importance').plot(kind='barh', title='Grid Search - Feature Importance', ax=ax[1], color='tab:green')
fig.tight_layout()


print("Decide the final model")
print("Comparing actual values against predicted values through line charts")

for k, val in preds.items():
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(f"{k} - Actual vs. Prediction")

    booster = val['params']['booster']
    learning_rate = val['params']['learning_rate']
    max_depth = val['params']['max_depth']
    n_estimators = val['params']['n_estimators']
    reg_alpha = val['params']['reg_alpha']
    subsample = val['params']['subsample']

    ax.set_title(
        f"'learning_rate': {0.3 if pd.isna(learning_rate) else learning_rate}, 'max_depth': {6 if pd.isna(max_depth) else max_depth}, "
        f"'n_estimators': {100 if pd.isna(n_estimators) else n_estimators}, 'reg_alpha': {0 if pd.isna(reg_alpha) else reg_alpha}, "
        f"'subsample': {1 if pd.isna(subsample) else subsample}"
    )
    ax.plot(X_test.index, val['actuals'], color='tab:blue', label='Actual')
    ax.plot(X_test.index, val['preds'], color='tab:orange', label='Prediction')
    ax.legend()
    fig.tight_layout()


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title("Actual vs. Prediction zoomed in 2023-02-20 - 2023-02-26")
ax.plot(X_test.index, preds['Default']['actuals'], color='tab:blue', linewidth=2.5, label='Actual')
ax.plot(X_test.index, preds['Default']['preds'], color='tab:green', label='Prediction of Default')
ax.plot(X_test.index, preds['GridSearchCV']['preds'], color='tab:purple', label='Prediction of GridSearchCV')
ax.set_xbound(lower=pd.to_datetime('2023-02-20'), upper=pd.to_datetime('2023-02-26'))
ax.legend()
fig.tight_layout()


print("Assessing errors")

for k, val in preds.items():
    # if k == 'default':
    #     fig, ax = plt.subplots(5, 1, figsize=(15, 5))
    #     fig.suptitle(f"{k} - Errors (= Actual - Prediction)")
    #     for fold, vals in val.items():
    #         n = int(fold.replace('fold_', ''))
    #         errors = vals['actual'] - vals['prediction']
    #         ax[n].plot(date_hour_df.iloc[tscv_idx[n][1]].index, errors, color='tab:red')
    #         ax[n].set_title(f'{fold} - Absolute Mean of errors: {round(abs(errors).sum()/len(errors), 2)}')
    #         ax[n].set_ylabel(f'{fold}')
    #
    #     fig.tight_layout()
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(f"{k} - Actual vs. Prediction")

    if k == 'default':
        errors = (y_test - y_pred).values
        ax.set_title(f"'learning_rate': {0.3 if pd.isna(xgb_reg.learning_rate) else xgb_reg.learning_rate}, 'max_depth': {6 if pd.isna(xgb_reg.max_depth) else xgb_reg.max_depth}, "
                     f"'n_estimators': {100 if pd.isna(xgb_reg.n_estimators) else xgb_reg.n_estimators}, 'reg_alpha': {0 if pd.isna(xgb_reg.reg_alpha) else xgb_reg.reg_alpha}, "
                     f"'subsample': {1 if pd.isna(xgb_reg.subsample) else xgb_reg.subsample}\nAbsolute Mean of errors: {round(abs(errors).sum() / len(errors), 2)}")
        ax.plot(X_test.index, errors, color='tab:red')

    else:
        errors = val['actual'] - val['pred']
        ax.set_title(f"{val['info']}\nAbsolute Mean of errors: {round(abs(errors).sum()/len(errors), 2)}")
        ax.plot(X_test.index, errors, color='tab:red')

    fig.tight_layout()
    ax.figure.savefig(f"{current_wd_path}/Chart/errors_{k}.png")


print("Save models")
import joblib

default_xgb_file = f"{current_wd_path}/Model/default_xgb.pkl"
joblib.dump(xgb_reg, default_xgb_file)

grid_search_xgb_file = f"{current_wd_path}Model/grid_search_xgb.pkl"
joblib.dump(xgb_reg, grid_search_xgb_file)

default_xgb = joblib.load(default_xgb_file)
y_pred = default_xgb.predict(X_test)

grid_search_xgb = joblib.load(grid_search_xgb_file)
grid_search_xgb


print("ARIMA modeling")

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


def check_stationarity(data):
    """
    Check stationarity of the data
    :param data:
    :return:
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, ax = plt.subplots(3, 1)
    # Run sequence plot
    data.plot(ax=ax[0])
    ax[0].set_ylabel('Ride Counts')
    # Autocorrelation plot
    plot_acf(data.dropna(), alpha=0.05, ax=ax[1], title=None)
    ax[1].set_ylabel('Autocorrelation')
    # Partial autocorrelation plot
    plot_pacf(data.dropna(), alpha=0.05, ax=ax[2], title=None)
    ax[2].set_ylabel('Partial autocorrelation')

    plt.show()
    plt.tight_layout()

    # Adfuller test
    adf_result = adfuller(data.dropna())
    print(f"ADF Statistic: {round(adf_result[0], 4)}")
    print(f"p-value: {round(adf_result[1], 4)}")
    print("Critical Values:")
    for key, val in adf_result[4].items():
        print(f"\t{key}: {val}")


result = seasonal_decompose(date_hour_df['log_ride_cnt'].diff().dropna(), model='addtive')
result = seasonal_decompose(date_hour_df['log_ride_cnt'].diff().diff().dropna(), model='addtive')

result.plot()
check_stationarity(date_hour_df['log_ride_cnt'])
check_stationarity(date_hour_df['log_ride_cnt'].diff())
check_stationarity(date_hour_df['log_ride_cnt'].diff().diff())
check_stationarity(date_hour_df['log_ride_cnt'].diff().diff().diff())

arima_model = ARIMA(endog=y_train.diff(), order=(0, 1, 0), freq='H')
arima_model_fit = arima_model.fit()

arima_model = ARIMA(endog=y_train.diff().diff(), order=(1, 2, 1), freq='H')
arima_model_fit = arima_model.fit()

arima_model = ARIMA(endog=y_train.diff().diff().diff(), order=(1, 3, 2), freq='H')
arima_model_fit = arima_model.fit()

y_pred = arima_model_fit.forecast(steps=len(y_test))
mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)

preds['ARIMA_010'] = {'params': arima_model_fit.model_orders,
                      'actuals': y_test.values,
                      'preds': y_pred.values}
evaluations['ARIMA_010'] = {'params': arima_model_fit.model_orders,
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape,
                            'MDAPE': mdape}

preds['ARIMA_121'] = {'params': arima_model_fit.model_orders,
                      'actuals': y_test.values,
                      'preds': y_pred.values}
evaluations['ARIMA_121'] = {'params': arima_model_fit.model_orders,
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape,
                            'MDAPE': mdape}

preds['ARIMA_132'] = {'params': arima_model_fit.model_orders,
                      'actuals': y_test.values,
                      'preds': y_pred.values}
evaluations['ARIMA_132'] = {'params': arima_model_fit.model_orders,
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape,
                            'MDAPE': mdape}


for k, val in preds.items():
    if 'ARIMA' in k:
        p, d, q = k.replace('ARIMA_', '')
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_title(f"ARIMA({p}, {d}, {q}) - Actual vs. Prediction")
        ax.plot(X_test.index, np.exp(val['actuals']), color='tab:blue', label='Actual')
        ax.plot(X_test.index, np.exp(val['preds']), color='tab:brown', label='Prediction')

        ax.legend()
        fig.tight_layout()


print("Prophet")

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot


prophet_df = y_train.reset_index().rename(columns={'start_date_hour': 'ds', 'log_ride_cnt': 'y'})


def is_winter(ds):
    """
    Add Winter factor to prophet dataframe
    :param ds: ds of prophet dataframe
    :return: 1 if it's Winter else 0
    """
    date = pd.to_datetime(ds)
    return int(date.month in [12, 1, 2])


def is_commute_hour(ds):
    """
    Add commute hour factor to prophet dataframe
    :param ds: ds of prophet dataframe
    :return: 1 if it's commute hour else 0
    """
    date = pd.to_datetime(ds)
    return int(date.weekday in range(0, 5) and date.hour in [7, 8, 17, 18])


prophet = Prophet()

# Add US holiday
prophet.add_country_holidays(country_name='US')

prophet_df['winter'] = prophet_df['ds'].apply(is_winter)
prophet_df['commute_hour'] = prophet_df['ds'].apply(is_commute_hour)

prophet.add_regressor('winter')
prophet.add_regressor('commute_hour')

prophet.fit(df=prophet_df)

future = prophet.make_future_dataframe(periods=len(y_test), freq='H')
future['winter'] = future['ds'].apply(is_winter)
future['commute_hour'] = future['ds'].apply(is_commute_hour)

prophet_forecast = prophet.predict(future)
prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

y_pred = prophet_forecast.iloc[-len(y_test):]['yhat'].values
mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)

fig, ax = plt.subplots(figsize=(15, 5))
fig = prophet.plot(prophet_forecast, ax=ax)
a = add_changepoints_to_plot(fig.gca(), prophet, prophet_forecast)
ax.scatter(y_test.index, y_test.values, color='tab:orange')
ax.axvline(y_test.index.min(), color='black', ls='--', label='Start Date of Test Set')
ax.set_title(f"Predictive {TARGET} by Prophet - RMSE: {round(rmse, 4)}")
plt.legend()
plt.tight_layout()

prophet.plot_components(prophet_forecast)
plt.tight_layout()
preds['Prophet'] = {'params': prophet.params,
                    'actuals': y_test.values,
                    'preds': prophet_forecast.iloc[-len(y_test):]['yhat'].values}
evaluations['Prophet'] = {'params': prophet.params,
                          'MAE': mae,
                          'RMSE': rmse,
                          'MAPE': mape,
                          'MDAPE': mdape}

preds['Prophet_reg'] = {'params': prophet.params,
                        'actuals': y_test.values,
                        'preds': prophet_forecast.iloc[-len(y_test):]['yhat'].values}
evaluations['Prophet_reg'] = {'params': prophet.params,
                              'MAE': mae,
                              'RMSE': rmse,
                              'MAPE': mape,
                              'MDAPE': mdape}

for k, val in evaluations.items():
    print(f"Evaluation metrics of {k}")
    print(f"MAE: {val['MAE']:.4f}")
    print(f"RMSE: {val['RMSE']:.4f}")
    print(f"MAPE: {val['MAPE']:.2f}%")
    print(f"MDAPE: {val['MDAPE']:.2f}%")


preds['GridSearchCV'].keys()
preds['GridSearchCV']['params']
preds['Prophet']['params']
color_palette = sns.color_palette()

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))
fig.suptitle("Actual vs. Prediction")

for i, (k, val) in enumerate(preds.items()):

    if 'ARIMA' not in k:
        de_tr_actual = np.exp(val['actuals']) - 0.001
        de_tr_pred = np.exp(val['preds']) - 0.001

        if 'Prophet' not in k:
            booster = val['params']['booster']
            learning_rate = val['params']['learning_rate']
            max_depth = val['params']['max_depth']
            n_estimators = val['params']['n_estimators']
            reg_alpha = val['params']['reg_alpha']
            subsample = val['params']['subsample']

            ax[i].set_title(
                f"XGBoost Regressor {i} - "
                f"'learning_rate': {0.3 if pd.isna(learning_rate) else learning_rate}, 'max_depth': {6 if pd.isna(max_depth) else max_depth}, "
                f"'n_estimators': {100 if pd.isna(n_estimators) else n_estimators}, 'reg_alpha': {0 if pd.isna(reg_alpha) else reg_alpha}, "
                f"'subsample': {1 if pd.isna(subsample) else subsample}"
            )
            ax[i].plot(X_test.index, de_tr_actual, color='tab:blue', label='Actual')
            ax[i].plot(X_test.index, de_tr_pred, color='tab:orange', label='Prediction')
            ax[i].legend()

        else:
            i -= 3
            ax[i].set_title(f"{k.replace('_reg', ' w/ regressor')}")
            ax[i].plot(X_test.index, de_tr_actual, color='tab:blue', label='Actual')
            ax[i].plot(X_test.index, de_tr_pred, color='tab:orange', label='Prediction')
            ax[i].legend()

fig.tight_layout()


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))
fig.suptitle("Actual vs. Prediction zoomed in 2023-02-20 - 2023-02-26")

for i, (k, val) in enumerate(preds.items()):

    if 'ARIMA' not in k:
        de_tr_actual = np.exp(val['actuals']) - 0.001
        de_tr_pred = np.exp(val['preds']) - 0.001

        if 'Prophet' not in k:
            booster = val['params']['booster']
            learning_rate = val['params']['learning_rate']
            max_depth = val['params']['max_depth']
            n_estimators = val['params']['n_estimators']
            reg_alpha = val['params']['reg_alpha']
            subsample = val['params']['subsample']

            ax[i].set_title(
                f"XGBoost Regressor {i} - "
                f"'learning_rate': {0.3 if pd.isna(learning_rate) else learning_rate}, 'max_depth': {6 if pd.isna(max_depth) else max_depth}, "
                f"'n_estimators': {100 if pd.isna(n_estimators) else n_estimators}, 'reg_alpha': {0 if pd.isna(reg_alpha) else reg_alpha}, "
                f"'subsample': {1 if pd.isna(subsample) else subsample}"
            )
            ax[i].plot(X_test.index, de_tr_actual, color='tab:blue', linewidth=2.5, label='Actual')
            ax[i].plot(X_test.index, de_tr_pred, color='tab:orange', label='Prediction')
            ax[i].set_xbound(lower=pd.to_datetime('2023-02-20'), upper=pd.to_datetime('2023-02-26'))
            ax[i].legend()

        else:
            i -= 3
            ax[i].set_title(f"{k.replace('_reg', ' w/ regressor')}")
            ax[i].plot(X_test.index, de_tr_actual, color='tab:blue', linewidth=2.5, label='Actual')
            ax[i].plot(X_test.index, de_tr_pred, color='tab:orange', label='Prediction')
            ax[i].set_xbound(lower=pd.to_datetime('2023-02-20'), upper=pd.to_datetime('2023-02-26'))
            ax[i].legend()

fig.tight_layout()
