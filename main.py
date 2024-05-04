import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import mean_squared_error
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from .preprocess_ import preprocess

data_set, selected_for_x, target = preprocess("./aiml_test_data.xlsx")

X = data_set[selected_for_x].copy()
Y = data_set[target].copy()

pca = PCA(n_components = 2, random_state=42)
scaler = StandardScaler()
PCA_Data = pd.DataFrame(pca.fit_transform(scaler.fit_transform(pd.concat([X, Y], axis=1))))

PCA_Data.columns = ['pca1', 'pca2']
PCA_Data

# 엘보우 메서드
def elbow_method(data, max_clusters=10):
    sse = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_init=20, n_clusters=k, init='k-means++', random_state=0)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    return sse

# 클러스터 개수 결정
def determine_clusters(data, max_clusters=10):
    sse = elbow_method(data, max_clusters)
    plt.plot(range(1, max_clusters+1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()

# 실행
determine_clusters(PCA_Data)

PCA_Data.plot.scatter(x='pca1', y='pca2')

# KMeans
kmeans_4 = KMeans(n_init=20, n_clusters=5, init='k-means++', random_state=42)
kmeans_4.fit(PCA_Data)
PCA_Data = pd.concat([PCA_Data ,pd.DataFrame(kmeans_4.predict(PCA_Data), columns=['KmeansC'])], axis=1)

ax = sns.scatterplot(x='pca1', y='pca2', hue='KmeansC', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# GMM
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(PCA_Data[['pca1', 'pca2']].copy())
PCA_Data['gmmC'] = gmm.predict(PCA_Data[['pca1', 'pca2']].copy())

ax = sns.scatterplot(x='pca1', y='pca2', hue='gmmC', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# DBSCAN
DBSCAN_model = DBSCAN(eps=0.087, min_samples=8)
DBSCAN_model.fit((PCA_Data[['pca1', 'pca2']].copy()))
# eps=0.087, min_samples=8

PCA_Data['DBSCAN'] = DBSCAN_model.fit_predict(PCA_Data[['pca1', 'pca2']].copy())

ax = sns.scatterplot(x='pca1', y='pca2', hue='DBSCAN', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# BIRCH
birchmodel = Birch(n_clusters=5)
birchmodel.fit((PCA_Data[['pca1', 'pca2']].copy()))
PCA_Data['birchC'] = birchmodel.fit_predict(PCA_Data[['pca1', 'pca2']].copy())

ax = sns.scatterplot(x='pca1', y='pca2', hue='birchC', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


X['cluster'] = PCA_Data['birchC']

total_data = pd.concat([X, Y], axis=1)

total_data['cluster'].value_counts()

total_data.groupby('cluster')['paid_amount'].mean()

total_data.columns

total_data.groupby('cluster').mean()

total_data.columns[1]

total_data.columns

total_data.groupby('cluster').sum()

plt.figure(figsize=(12, 8))

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.hist(total_data[total_data['cluster'] == 4][total_data.columns[i]], bins=10)
    plt.title(total_data.columns[i])

plt.show()

plt.figure(figsize=(12, 8))

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.hist(total_data[total_data['cluster'] == 3][total_data.columns[i]], bins=10)
    plt.title(total_data.columns[i])

plt.show()

X = total_data[selected_for_x].copy()
Y = total_data[target].copy()

# 2024년 4월 test data 나머지 train data로 사용
train_data = total_data[~((total_data['year']==2024) & (total_data['month']==4))]
test_data = total_data[(total_data['year']==2024) & (total_data['month']==4)]

train_data_x, train_data_y = train_data[selected_for_x].copy(), train_data[target].copy()
test_data_x, test_data_y = test_data[selected_for_x].copy(), test_data[target].copy()

train_data_x

train_data_y

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

    RF_models[i].fit(train_data_x.iloc[train_idx], train_data_y.iloc[train_idx])
    CB_models[i].fit(train_data_x.iloc[train_idx], train_data_y.iloc[train_idx])
    LGBM_models[i].fit(train_data_x.iloc[train_idx], train_data_y.iloc[train_idx])
    GBM_models[i].fit(train_data_x.iloc[train_idx], train_data_y.iloc[train_idx])
    mlp_models[i].fit(train_data_x.iloc[train_idx], train_data_y.iloc[train_idx])

    rfpred = RF_models[i].predict(train_data_x.iloc[test_idx])
    cbpred = CB_models[i].predict(train_data_x.iloc[test_idx])
    lgbmpred = LGBM_models[i].predict(train_data_x.iloc[test_idx])
    gbmpred = GBM_models[i].predict(train_data_x.iloc[test_idx])
    mlp_pred = mlp_models[i].predict(train_data_x.iloc[test_idx])

    i += 1

    RF_score.append(mean_squared_error(train_data_y.iloc[test_idx], rfpred, squared=False))
    CB_score.append(mean_squared_error(train_data_y.iloc[test_idx], cbpred, squared=False))
    LGBM_score.append(mean_squared_error(train_data_y.iloc[test_idx], lgbmpred, squared=False))
    GBM_score.append(mean_squared_error(train_data_y.iloc[test_idx], gbmpred, squared=False))
    mlp_score.append(mean_squared_error(train_data_y.iloc[test_idx], mlp_pred, squared=False))
    print(str(i)+'/5')
    
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

# validation 에서 가장 좋은 성능 GBM이 test에서도 가장 좋은 걸 확인 (실제 데이터 에서는 test가 없음)

plt.figure(figsize=(12, 10))
sns.heatmap(total_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

for model in GBM_models.values():
    ser = pd.Series(model.feature_importances_, index=train_data_x.columns)
    # 내림차순 정렬을 이용한다
    top7 = ser.sort_values(ascending=False)[:7]
    print(top7)
    print('-------------------------------')