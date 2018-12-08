import numpy as np
import pandas as pd
import renders as rs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import itertools
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("Wholesalecustomersdata.csv")
data.drop(['Region', 'Channel'], axis=1, inplace=True)

stats = data.describe()
# print(stats)

# print(data.loc[[100, 200, 300], :])

# print(data.columns)

indices = [43, 12, 39]
samples = pd.DataFrame(data.loc[indices], columns = data.columns).reset_index(drop=True)
# print(samples)

mean_data = data.describe().loc['mean', :]
samples_bar = samples.append(mean_data)
samples_bar.index = indices + ['mean']
# samples_bar.plot(kind='bar', figsize=(14,8))
# plt.show()

percentiles = data.rank(pct=True)
percentiles = 100*percentiles.round(decimals=3)
percentiles = percentiles.iloc[indices]
# sns.heatmap(percentiles, vmin=1, vmax=99, annot=True)
# plt.show()

print(data.columns)
dep_vars = list(data.columns)
for var in dep_vars:
    new_data = data.drop([var], axis=1)
    new_feature = pd.DataFrame(data.loc[:, var])
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_train, y_train)
    score=dtr.score(X_test, y_test)
    # print('R2 score for {} as dependent variable: {}'.format(var, score))


# pd.scatter_matrix(data, alpha=0.3, figsize=(14,8), diagonal='kde')
# plt.show()

def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(df, interpolation='nearest')
    ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
# plot_corr(data)
# plt.show()

log_data = np.log(data)
log_samples = np.log(samples)
# pd.scatter_matrix(log_data, alpha=0.3, figsize=(14, 8), diagonal='kde')
# plt.show()

# print(log_samples)

# plot_corr(data)
# plot_corr(log_data)
# plt.show()

print(np.percentile(data.loc[:, 'Milk'], 25))

outliers_lst = []
for feature in log_data.columns:
    Q1 = np.percentile(log_data.loc[:, feature], 25)
    Q3 = np.percentile(log_data.loc[:, feature], 75)
    step = 1.5 * (Q3 - Q1)
    # print("data point considered outliers for the feature '{}': ".format(feature))
    outliers_rows = log_data.loc[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step)), :]
    outliers_lst.append(list(outliers_rows.index))
outliers = list(itertools.chain.from_iterable(outliers_lst))
uniq_outliers = list(set(outliers))
dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
# print('outliers list: \n', uniq_outliers)
# print('length of outliers list: \n', len(uniq_outliers))
# print('Duplicate list: \n', dup_outliers)
# print('Length of duplicates list: \n', len(dup_outliers))
good_data = log_data.drop(log_data.index[dup_outliers]).reset_index(drop=True)
# print('original shape of data: \n', data.shape)
# print('new shape of data: \n', good_data.shape)


# pca = PCA(n_components=6)
# pca.fit(good_data)
# pca_samples = pca.transform(log_samples)
# pca_results = rs.pca_results(good_data, pca)
# plt.show()
# print(pca_results)
# print(type(pca_results))
# print(pca_results['Explained Variance'].cumsum())
# print(pd.DataFrame(np.round(pca_samples, 4), columns=pca_results.index.values))

pca = PCA(n_components=2)
pca.fit(good_data)
reduced_data = pca.transform(good_data)
pca_samples = pca.transform(log_samples)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
# print(pd.DataFrame(np.round(pca_sample, 4), columns = ['Dimension 1', 'Dimension 2']))


range_n_clusters = list(range(2, 11))
print(range_n_clusters)

for n_clusters in range_n_clusters:
    clusterer = GaussianMixture(n_components=n_clusters).fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    centers = clusterer.predict(pca_samples)
    score = silhouette_score(reduced_data, preds, metric='mahalanobis')
    # print("for n_clusters = {}. the average silhouette_score is : {}".format(n_clusters, score))

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
# for cv_type in cv_types:
#     for n_components in n_components_range:
#         gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
#         gmm.fit(X)
#         bic.append(gmm.bic(X))
#         if bic[-1] < lowest_bic:
#             lowest_bic = bic[-1]
#             best_gmm = gmm


for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    centers = clusterer.cluster_centers_
    sample_preds = clusterer.predict(pca_samples)
    score = silhouette_score(reduced_data, preds, metric='euclidean')
    print(" for n_clusters = {} . the average silhouette_score is : {}".format(n_clusters, score))


clusterer = GaussianMixture(n_components=2).fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.means_
sample_preds = clusterer.predict(pca_samples)
rs.cluster_results(reduced_data, preds, centers, pca_samples)
# plt.show()

log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)
segments = ['Segment {}'.format(i) for i in range(0, len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns=data.columns)
true_centers.index = segments
# print(true_centers)
# print(true_centers - data.median())
# print(true_centers - data.mean())

for i, pred in enumerate(sample_preds):
    print("sample point", i, "predicted to be in cluster", pred)
# print(samples)
# print(dup_outliers)
rs.channel_results(reduced_data, dup_outliers, pca_samples)
plt.show()
