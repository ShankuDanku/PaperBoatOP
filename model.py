import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from unidecode import unidecode

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.encoding', 'UTF-8')
pd.set_option('display.width', 100)

stats_df = pd.read_csv('data/2021_stats.csv')

basic_stats = ['Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'MP']
per_100_stats = ['FGA', 'FT%', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

df = stats_df.loc[:, basic_stats + per_100_stats]

df['Player'] = df['Player'].apply(func=lambda name: unidecode(name))

shooting_df = pd.read_csv('data/2021_shooting.csv')

dist_metrics = ['0-3', '3-10', '10-16', '16-3P', '3P']
for metric in dist_metrics:
	df.insert(df.shape[1], metric + " FGA",  shooting_df[metric] * df['FGA'])

shoot_metrics = ['0-3.1', '3-10.1', '10-16.1', '16-3P.1', '3P.1']
for metric in shoot_metrics:
	df.insert(df.shape[1], metric[:-2] + " FG%",  shooting_df[metric])

df.fillna(0, inplace=True)

df = df[df['G'] >=20]
df = df[df['MP'] >= 500]

df.insert(7, 'MPG', df['MP']/df['G'])
df = df[df['MPG'] >= 25]

stats = list(df.columns[8:])

scaler = StandardScaler()

df[stats] = scaler.fit_transform(df[stats])


offensive_stats = ['FT%', 'FTA', 'ORB', 'AST', 'TOV', '0-3 FGA', '3-10 FGA', '10-16 FGA', '16-3P FGA', '3P FGA', 
'0-3 FG%', '3-10 FG%', '10-16 FG%', '16-3P FG%', '3P FG%']

wcss = []

for k in range(1, 15):
	model = KMeans(n_clusters=k)
	model.fit(df.loc[:, offensive_stats])
	wcss.append(model.inertia_)

sns.set_style('darkgrid')
plot.scatter(list(range(1, 15)), wcss, c='red')
plot.plot(list(range(1, 15)), wcss)
plot.show()


offensive_clusters = 8

model = KMeans(n_clusters=offensive_clusters)
model.fit(df.loc[:, offensive_stats])

df.insert(df.shape[1], 'OFF ARCH', model.labels_)

print("Silhoutte Score KMeans", silhouette_score(df.loc[:, offensive_stats], model.labels_))

pca = PCA(n_components=3)
pca_data = pd.DataFrame(pca.fit_transform(df.loc[:, offensive_stats]))
pca_data.insert(pca_data.shape[1], 3, model.labels_)

fig = plot.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

markers = ['v', 'x', 's', 'p', 'o', 'P', 'd', '1']

for arch in pca_data[3].unique():
	x = pca_data[pca_data[3] == arch][0]
	y = pca_data[pca_data[3] == arch][1]
	z = pca_data[pca_data[3] == arch][2]

	ax.scatter(x, y, z, marker=markers[arch], alpha=1, label=arch)
ax.legend()
plot.show()

# plot.figure(figsize=(8,22))
# plot.title('Hierarchical Clustering Dendrogram')
# dend = shc.dendrogram(shc.linkage(df.loc[:, offensive_stats], method='ward'),labels=list(df.Player),orientation='left')

# plot.yticks(fontsize=6)
# plot.xlabel('Height')

# plot.tight_layout()

# plot.savefig('dendogram.png')

model2 = AgglomerativeClustering(n_clusters=8)
model2.fit(df.loc[:, offensive_stats])

df.insert(df.shape[1], 'OFF ARCH 2', model2.labels_)

print("Silhoutte Score Agglomerative", silhouette_score(df.loc[:, offensive_stats], model2.labels_))

print(df.sort_values(by=['OFF ARCH']).loc[:, ['Player', 'OFF ARCH']])

# print(df.sort_values(by=['OFF ARCH']).loc[:, ['Player', 'OFF ARCH', 'OFF ARCH 2']])

# pca = PCA(n_components=3)
# pca_data = pd.DataFrame(pca.fit_transform(df.loc[:, offensive_stats]))
# pca_data.insert(pca_data.shape[1], 3, model2.labels_)

# fig = plot.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')

# markers = ['v', 'x', 's', 'p', 'o', 'P', 'd', '1']

# for arch in pca_data[3].unique():
# 	x = pca_data[pca_data[3] == arch][0]
# 	y = pca_data[pca_data[3] == arch][1]
# 	z = pca_data[pca_data[3] == arch][2]

# 	ax.scatter(x, y, z, marker=markers[arch], alpha=1, label=arch)
# ax.legend()
# plot.show()









