import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_models(train_data_x, train_data_y, test_data_x, test_data_y):
    iris = KFold(n_splits=5, shuffle=True, random_state=42)
    i = 0
    RF_models = {}
    CB_models = {}
    LGBM_models = {}
    GBM_models = {}
    mlp_models = {}

    RF_score = []
    CB_score = []
    LGBM_score = []
    GBM_score = []
    mlp_score = []

    print('Start Learning ML/DL Models...')
    for train_idx, test_idx in iris.split(train_data_x, train_data_y):
        RF_models[i] = RandomForestRegressor(random_state=42)
        CB_models[i] = CatBoostRegressor(random_state=42)
        LGBM_models[i] = LGBMRegressor(random_state=42)
        GBM_models[i] = GradientBoostingRegressor(random_state=42)
        mlp_models[i] = MLPRegressor(random_state=42)

        RF_models[i].fit(train_data_x.iloc[train_idx].values, train_data_y.iloc[train_idx].values.ravel())
        CB_models[i].fit(train_data_x.iloc[train_idx].values, train_data_y.iloc[train_idx].values.ravel())
        LGBM_models[i].fit(train_data_x.iloc[train_idx].values, train_data_y.iloc[train_idx].values.ravel())
        GBM_models[i].fit(train_data_x.iloc[train_idx].values, train_data_y.iloc[train_idx].values.ravel())
        mlp_models[i].fit(train_data_x.iloc[train_idx].values, train_data_y.iloc[train_idx].values.ravel())

        rfpred = RF_models[i].predict(train_data_x.iloc[test_idx].values)
        cbpred = CB_models[i].predict(train_data_x.iloc[test_idx].values)
        lgbmpred = LGBM_models[i].predict(train_data_x.iloc[test_idx].values)
        gbmpred = GBM_models[i].predict(train_data_x.iloc[test_idx].values)
        mlp_pred = mlp_models[i].predict(train_data_x.iloc[test_idx].values)

        i += 1

        RF_score.append(mean_squared_error(train_data_y.iloc[test_idx], rfpred, squared=False))
        CB_score.append(mean_squared_error(train_data_y.iloc[test_idx], cbpred, squared=False))
        LGBM_score.append(mean_squared_error(train_data_y.iloc[test_idx], lgbmpred, squared=False))
        GBM_score.append(mean_squared_error(train_data_y.iloc[test_idx], gbmpred, squared=False))
        mlp_score.append(mean_squared_error(train_data_y.iloc[test_idx], mlp_pred, squared=False))
        print('train progress ' + str(i) + '/5')
        
    print('')
        
        

    print('----Validation RMSE----')
    print(f"{np.mean(RF_score):.2f}", 'RF')
    print(f"{np.mean(CB_score):.2f}", 'CB')
    print(f"{np.mean(LGBM_score):.2f}", 'LGBM')
    print(f"{np.mean(GBM_score):.2f}", 'GBM')
    print(f"{np.mean(mlp_score):.2f}", 'MLP')

    RF_test = []
    CB_test = []
    LGBM_test = []
    GBM_test = []
    mlp_test = []

    for i in range(5):
        rf_pred = RF_models[i].predict(test_data_x)
        cb_pred = CB_models[i].predict(test_data_x)
        lgbm_pred = LGBM_models[i].predict(test_data_x)
        gbm_pred = GBM_models[i].predict(test_data_x)
        mlp_pred = mlp_models[i].predict(test_data_x)

        RF_test.append(mean_squared_error(rf_pred, test_data_y, squared=False))
        CB_test.append(mean_squared_error(cb_pred, test_data_y, squared=False))
        LGBM_test.append(mean_squared_error(lgbm_pred, test_data_y, squared=False))
        GBM_test.append(mean_squared_error(gbm_pred, test_data_y, squared=False))
        mlp_test.append(mean_squared_error(mlp_pred, test_data_y, squared=False))

    print('-----test RMSE-----')
    print(np.mean(RF_test), 'RF')
    print(np.mean(CB_test), 'CB')
    print(np.mean(LGBM_test), 'LGBM')
    print(np.mean(GBM_test), 'GBM')
    print(np.mean(mlp_test), 'MLP')

    for model in GBM_models.values():
        ser = pd.Series(model.feature_importances_, index=train_data_x.columns)
        # 내림차순 정렬을 이용한다
        top7 = ser.sort_values(ascending=False)[:7]
        print(top7)
        print('-------------------------------')
        
    