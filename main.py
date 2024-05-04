import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.preprocessing import StandardScaler
from preprocess_ import preprocess
from Modeling import train_models

data_set, selected_for_x, target = preprocess("aiml_test_data.xlsx")

X = data_set[selected_for_x].copy()
Y = data_set[target].copy()

# PCA 알고리즘 차원 축소
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

# 엘보우 메소드 실행
determine_clusters(PCA_Data)

PCA_Data.plot.scatter(x='pca1', y='pca2')

'''clustering 시각화'''
# KMeans
kmeans_4 = KMeans(n_init=20, n_clusters=5, init='k-means++', random_state=42)
kmeans_4.fit(PCA_Data)
PCA_Data = pd.concat([PCA_Data ,pd.DataFrame(kmeans_4.predict(PCA_Data), columns=['KmeansC'])], axis=1)
ax = sns.scatterplot(x='pca1', y='pca2', hue='KmeansC', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title('Kmeans')
plt.show()
# GMM
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(PCA_Data[['pca1', 'pca2']].copy())
PCA_Data['gmmC'] = gmm.predict(PCA_Data[['pca1', 'pca2']].copy())
ax = sns.scatterplot(x='pca1', y='pca2', hue='gmmC', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title('GMM')
plt.show()
# DBSCAN
DBSCAN_model = DBSCAN(eps=0.087, min_samples=8)
DBSCAN_model.fit((PCA_Data[['pca1', 'pca2']].copy()))
PCA_Data['DBSCAN'] = DBSCAN_model.fit_predict(PCA_Data[['pca1', 'pca2']].copy())
ax = sns.scatterplot(x='pca1', y='pca2', hue='DBSCAN', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title('DBSCAN')
plt.show()
# BIRCH
birchmodel = Birch(n_clusters=5)
birchmodel.fit((PCA_Data[['pca1', 'pca2']].copy()))
PCA_Data['birchC'] = birchmodel.fit_predict(PCA_Data[['pca1', 'pca2']].copy())
ax = sns.scatterplot(x='pca1', y='pca2', hue='birchC', data=PCA_Data, palette="deep")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title('BIRCH')
plt.show()

X['cluster'] = PCA_Data['birchC']

total_data = pd.concat([X, Y], axis=1)

# BIRCH cluster 3, 4 분석
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



# 2024년 4월 test data 나머지 train data로 사용
train_data = total_data[~((total_data['year']==2024) & (total_data['month']==4))]
test_data = total_data[(total_data['year']==2024) & (total_data['month']==4)]

train_data_x, train_data_y = train_data[selected_for_x].copy(), train_data[target].copy()
test_data_x, test_data_y = test_data[selected_for_x].copy(), test_data[target].copy()

train_models(train_data_x, train_data_y, test_data_x, test_data_y)