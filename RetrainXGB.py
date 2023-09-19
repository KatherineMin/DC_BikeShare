
def retrain_xgb(df, features, target):

    import ForecastUsage

    import pandas as pd
    import numpy as np
    from datetime import timedelta

    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import RandomizedSearchCV

    print(f"See if df is still indexed by start_date_hour\n{df.head()}")
    if np.issubdtype(df.index.dtype, np.datetime64):
        max_datetime = df.index.max()
    else:
        df.set_index(keys='start_date_hour', inplace=True)
        df.index = pd.to_datetime(df.index)
        max_datetime = df.index.max()

    train_df = df.loc[df.index < (max_datetime - timedelta(days=28*2))]
    test_df = df.loc[df.index >= (max_datetime - timedelta(days=28*2))]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 28 * 2, gap=24)
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

    print("Hyperparameter Tuning with Randomized Search CV")
    xgb_reg = xgb.XGBRegressor(seed=813, booster='gbtree')
    random_search = RandomizedSearchCV(estimator=xgb_reg,
                                       param_distributions=params,
                                       scoring='neg_mean_squared_error',
                                       n_iter=10,
                                       cv=tscv_idx)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    cv_results = {}
    tscv = TimeSeriesSplit(n_splits=5, test_size=24 * 28 * 2, gap=24)
    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print("XGBoost Models")
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # Measure the performance of the selected best model
        best_model.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), (X_test, y_test)],
                       verbose=100)
        y_pred = best_model.predict(X_test)

        mae, rmse, mape, mdape = ForecastUsage.evaluate_model(y_test, y_pred)

        if i == 0:
            cv_results['tuned_xgb'] = {f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}}
        else:
            cv_results['tuned_xgb'].update({f"fold{i}": {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}})

    performance = {}
    for model in cv_results.keys():
        mae, rmse, mape, mdape = ForecastUsage.transform_cv_results(cv_results[model])
        performance[model] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'MDAPE': mdape}

    performance_df = pd.DataFrame.from_dict(performance, orient='index')
    performance_df.index.name = 'model'

    performance_df.sort_values(by=['MAE', 'MDAPE'], ascending=[True, True], inplace=True)

    return best_model, performance_df.reset_index().iloc[0].to_dict()
