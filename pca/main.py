import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA

# データの読み込み
url = "https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/wine.txt"
urllib.request.urlretrieve(url, 'wine.txt')
df = pd.read_csv("wine.txt", sep = "\t", index_col = 0)
print(df.head())

# 前処理==========================================================================================================
# 散布図によるデータの外観
from pandas import plotting
plotting.scatter_matrix(
    df.iloc[:, 1:], figsize = (8, 8), c = list(df.iloc[:, 0]),
    alpha = 0.5
)
plt.show()
plt.close()

# 行列の標準化
dfs = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis = 0)
dfs = dfs[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash']]
print(dfs.head())

# 主成分分析======================================================================================================
# 実行
pca = PCA()
pca.fit(dfs)
# データを主成分空間に写像
feature = pca.transform(dfs)

# 主成分得点-------------------------------------------------------------------
pc = pd.DataFrame(
    feature,
    columns = ["PC{}".format(x + 1) for x in range(len(dfs.columns))]
).head()
print(pc)

# 第一、二主成分でプロット
plt.figure(figsize = (6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha = 0.8, c = list(df.iloc[:, 0]))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 各主成分のグラフ外観
plotting.scatter_matrix(
    pd.DataFrame(feature,columns = ["PC{}".format(x + 1) for x in range(len(dfs.columns))]),
    figsize = (8, 8),
    c = list(df.iloc[:, 0]),
    alpha = 0.5
)
plt.show()

# 寄与率と累積寄与率------------------------------------------------------------
# 寄与率
cr = pd.DataFrame(
    pca.explained_variance_ratio_,
    index = ["PC{}".format(x + 1) for x in range(len(dfs.columns))]
)
print("寄与率")
print(cr)

# 累積寄与率
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer = True))
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()

# 固有値と固有ベクトル----------------------------------------------------------
# PCAの固有値
eval = pd.DataFrame(
    pca.explained_variance_,
    index = ["PC{}".format(x + 1) for x in range(len(dfs.columns))]
)
print("PCAの固有値")
print(eval)

# 固有ベクトル
#evec = pd.DataFrame(
    #pca.components_,
    #columns = df.columns[1:],
    #index = ["PC{}".format(x + 1) for x in range(len(dfs.columns))]
#)
#print("PCAの固有ベクトル")
#print(evec)

# 第一、二主成分におけるパラメータの寄与度をプロット
plt.figure(figsize = (6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[2:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha = 0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()