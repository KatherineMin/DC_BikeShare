import os
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from pytz import timezone

import GetBikeShareData
import RetrainXGB


current_wd_path = os.getcwd()
data_wd_path = f"{current_wd_path}/DC BikeShare Data"
model_wd_path = f"{current_wd_path}/Model"


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


def transform_cv_results(dict):
    weights = [0.05, 0.1, 0.2, 0.3, 0.35]
    mae, rmse, mape, mdape = 0, 0, 0, 0

    for i, w in zip(range(5), weights):
        fold_metrics = dict[f"fold{i}"]
        mae += w * fold_metrics['MAE']
        rmse += w * fold_metrics['RMSE']
        mape += w * fold_metrics['MAPE']
        mdape += w * fold_metrics['MDAPE']

    return mae, rmse, mape, mdape


def append_performance_log(performance_dict):
    from csv import DictWriter

    field_names = ['model', 'MAE', 'RMSE', 'MAPE', 'MDAPE', 'run_date']

    with open(f"{data_wd_path}/performance_log.csv", 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        dictwriter_object.writerow(performance_dict)
        f_object.close()


def append_prediction_log(prediction_df):
    past_prediction_df = pd.read_csv(f"{data_wd_path}/prediction_log.csv")
    updated_prediction_df = pd.concat([past_prediction_df, prediction_df], axis=0, ignore_index=True)
    updated_prediction_df.to_csv(f"{data_wd_path}/prediction_log.csv", index=None)


def train_model(df):
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from prophet import Prophet

    import joblib
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 300)

    df.set_index(keys='start_date_hour', inplace=True)

    last_datetime = pd.to_datetime(df.index.max())

    future_periods = pd.date_range(start=last_datetime + timedelta(hours=1), periods=24 * 7 * 8, freq='H')
    predictions = pd.DataFrame(index=future_periods, columns=['ride_cnt', 'log_ride_cnt'])
    predictions.index.name = 'start_date_hour'
    predictions['date'] = pd.to_datetime(predictions.index.date)

    predictions = GetBikeShareData.breakdown_datetime(predictions)
    predictions = GetBikeShareData.add_features(predictions)
    predictions['active_station_cnt'] = predictions['active_station_cnt'].ffill()
    print(predictions)

    print("Start tuning features")
    df = pd.get_dummies(df).replace(to_replace=[True, False], value=[1, 0])
    predictions[['ride_cnt', 'log_ride_cnt']] = predictions[['ride_cnt', 'log_ride_cnt']].fillna(-9)
    predictions = pd.get_dummies(predictions).replace(to_replace=[True, False], value=[1, 0])
    predictions[['ride_cnt', 'log_ride_cnt']] = np.nan

    def log_transformation(df, target='ride_cnt'):
        new_target = f"log_{target}"
        alpha = 0.001

        df[new_target] = np.log(df[target] + alpha)
        return new_target, df

    target, df = log_transformation(df)
    prophet_df = df[target].reset_index().rename(columns={'start_date_hour': 'ds', 'log_ride_cnt': 'y'})

    print("Append prediction rows to dataframe")
    off_seasons = [col for col in df.filter(regex='season_') if col not in predictions.filter(regex='season_').columns]
    predictions[off_seasons] = 0
    print(predictions.head())
    print(df.columns)
    predictions = predictions[df.columns.tolist()]
    df = pd.concat([df, predictions], axis=0)
    print(df.head())

    def add_lags(df):
        """
        Add lagged values of the target variable to dataset for XGBoost Model
        :param df: dataset for forecasting
        :return: the dataset that has lagged values appended
        """
        date_hour_map = df[target].to_dict()

        # Bring 8-week-ago target value
        df['lag_8w'] = (df.index - timedelta(days=7 * 8)).map(date_hour_map)
        # Bring 1-year-ago target value
        df['lag_1y'] = (df.index - timedelta(days=7 * 52)).map(date_hour_map)
        # Bring 2-year-ago target value
        df['lag_2y'] = (df.index - timedelta(days=7 * 52 * 2)).map(date_hour_map)

        return df

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

    df = add_lags(df)

    predictions = df.loc[df[target].isna()]
    df = df.loc[~df[target].isna()]

    features = [col for col in df.columns if col not in ['date', 'ride_cnt', 'log_ride_cnt']]
    print(f"Input variables for the XGBoost model are {features}")

    prophet_df['winter'] = prophet_df['ds'].apply(is_winter)
    prophet_df['commute_hour'] = prophet_df['ds'].apply(is_commute_hour)

    # Default XGBoost
    xgb = xgb.XGBRegressor(seed=813, booster='gbtree')

    # Turned XGBoost
    tuned_xgb_file = f"{model_wd_path}/tuned_xgb.pkl"
    tuned_xgb = joblib.load(tuned_xgb_file)

    print("Start testing candidate models")
    cv_results = {}
    tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 28 * 2, gap=24)
    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):

        print("XGBoost Models")
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # Measure the performance of Default XGBoost
        xgb.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)
        y_pred = xgb.predict(X_test)
        mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)
        if i == 0:
            cv_results['default_xgb'] = {f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}}
        else:
            cv_results['default_xgb'].update({f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}})

        # Measure the performance of Turned XGBoost
        tuned_xgb.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=100)
        y_pred = tuned_xgb.predict(X_test)
        mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)
        if i == 0:
            cv_results['tuned_xgb'] = {f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}}
        else:
            cv_results['tuned_xgb'].update({f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}})

        print("Prophet Models")
        train_df = prophet_df.iloc[train_idx]

        # Measure the performance of Default Prophet
        prophet = Prophet()
        prophet.fit(df=train_df[['ds', 'y']])
        future = prophet.make_future_dataframe(periods=len(test_idx), freq='H', include_history=False)
        forecast = prophet.predict(future)
        y_pred = forecast['yhat'].values
        mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)
        if i == 0:
            cv_results['default_prophet'] = {f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}}
        else:
            cv_results['default_prophet'].update({f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}})

        # Measure the performance of Prophet with Regressor
        prophet_reg = Prophet()
        prophet_reg.add_country_holidays(country_name='US')
        prophet_reg.add_regressor('winter')
        prophet_reg.add_regressor('commute_hour')
        prophet_reg.fit(df=train_df)
        future = prophet_reg.make_future_dataframe(periods=len(test_idx), freq='H', include_history=False)
        future['winter'] = future['ds'].apply(is_winter)
        future['commute_hour'] = future['ds'].apply(is_commute_hour)
        forecast = prophet_reg.predict(future)
        y_pred = forecast['yhat'].values
        mae, rmse, mape, mdape = evaluate_model(y_test, y_pred)
        if i == 0:
            cv_results['prophet_reg'] = {f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}}
        else:
            cv_results['prophet_reg'].update({f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}})

    performance = {}
    for model in cv_results.keys():
        mae, rmse, mape, mdape = transform_cv_results(cv_results[model])
        performance[model] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}

    performance_df = pd.DataFrame.from_dict(performance, orient='index')
    performance_df.index.name = 'model'

    # Pick the best performing model by MAE & MDAPE
    performance_df.sort_values(by=['MAE', 'MDAPE'], ascending=[True, True], inplace=True) # MAE >> RMSE && MDAPE >> MAPE
    print(f"Check performance of the models:\n{performance_df}")

    run_date = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d")
    today_performance = performance_df.reset_index().iloc[0].to_dict()

    past_performance = pd.read_csv(f"{data_wd_path}/performance_log.csv").iloc[-1].to_dict()

    comp_mae = 1 + (today_performance['MAE'] - past_performance['MAE']) / past_performance['MAE']
    comp_mdape = 1 + (today_performance['MDAPE'] - past_performance['MDAPE']) / past_performance['MDAPE']

    if (comp_mae < 0.95) & (comp_mdape < 0.95):
        print(f"Check df before passing it to retrain_xgb\n{df.head()}")
        new_tuned_xgb, new_performance = RetrainXGB.retrain_xgb(df, features, target)

        new_comp_mae = 1 + (new_performance['MAE'] - today_performance['MAE']) / today_performance['MAE']
        new_comp_mdape = 1 + (new_performance['MDAPE'] - today_performance['MDAPE']) / today_performance['MDAPE']

        if (new_comp_mae >= 1) | (new_comp_mdape >= 1):
            best_model_type = 'xgb'
            best_model = new_tuned_xgb
            new_performance.update({'run_date': run_date})
            append_performance_log(new_performance)

        else:
            best_model_type = today_performance['model']
            best_model = locals()[today_performance['model']]
            today_performance.update({'run_date': run_date})
            append_performance_log(today_performance)

    else:
        best_model_type = today_performance['model']
        best_model = locals()[today_performance['model']]
        today_performance.update({'run_date': run_date})
        append_performance_log(today_performance)

    if 'xgb' in best_model_type:
        y_pred = best_model.predict(predictions[features])
        predictions['tr_pred'] = y_pred

        pred_result = predictions.reset_index()[['start_date_hour', 'tr_pred']]
        pred_result['pred'] = np.exp(pred_result['tr_pred'] - 0.001).round().astype(int)
        pred_result['run_date'] = run_date
        pred_result['model'] = best_model_type

    else:
        prophet_predictions = predictions.reset_index().rename(columns={'start_date_hour': 'ds'})
        prophet_predictions = prophet_predictions[['ds']]

        if '_reg' in best_model_type:
            prophet_predictions['winter'] = prophet_predictions['ds'].apply(is_winter)
            prophet_predictions['commute_hour'] = prophet_predictions['ds'].apply(is_commute_hour)

            y_pred = best_model.predict(prophet_predictions)['yhat']

            prophet_predictions['tr_preds'] = y_pred
            pred_result = prophet_predictions.rename(columns={'ds': 'start_date_hour'})
            pred_result['preds'] = np.exp(pred_result['tr_pred'] - 0.001).round().astype(int)
            pred_result['run_date'] = run_date
            pred_result['model'] = best_model_type

        else:
            y_pred = best_model.predict(prophet_predictions)['yhat']

            prophet_predictions['tr_preds'] = y_pred
            pred_result = prophet_predictions.rename(columns={'ds': 'start_date_hour'})
            pred_result['preds'] = np.exp(pred_result['tr_pred'] - 0.001).round().astype(int)
            pred_result['run_date'] = run_date
            pred_result['model'] = best_model_type

    append_prediction_log(pred_result)

## test it out!!