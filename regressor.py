import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt


def get_prediction_lm(X_train, y_train, X_val):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    prediction = lm.predict(X_val)
    
    return prediction


def get_prediction_rf(X_train, y_train, X_val):
    rf = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=1234)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_val)

    return prediction


def get_prediction_lgbm(X_train, y_train, X_val):
    gbm = lgb.LGBMRegressor(n_estimators=50,
                            silent=True, 
                            num_leaves = 31,
                            learning_rate=0.1,
                            min_child_samples=10,
                            random_state=1234)
    gbm.fit(X_train, y_train)
    prediction = gbm.predict(X_val)

    return prediction
