import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def print_rmse(prediction, y_val):
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, prediction))
    print('RandomForest RMSE: ', rmse)

    
def show_scatter(prediction, y_val):
    # plotting a chart
    plt.scatter(y_val, prediction)


def get_eval_data():
    # Boston データセットを読み込む
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    return X_train, X_test, y_train, y_test
    

def main():
    X_train, X_test, y_train, y_test = get_eval_data()
    
    return print(X_train)


    # # 上記のパラメータでモデルを学習する
    # model = lgb.LGBMRegressor()
    # model.fit(X_train, y_train)

    # # テストデータを予測する
    # y_pred = model.predict(X_test)

    # # RMSE を計算する
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(rmse)


if __name__ == '__main__':
    main()