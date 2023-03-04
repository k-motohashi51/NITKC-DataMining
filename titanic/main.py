import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder
import sklearn

# データを読み込む
train_set = pd.read_csv('data/train.csv')
test_set = pd.read_csv('data/test.csv')

fig = plt.figure(figsize = (12, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# PClass (旅客等級)
PClassPlot = train_set['Survived'].groupby(train_set['Pclass']).mean()
ax1.bar(x = PClassPlot.index, height = PClassPlot.values)
ax1.set_ylabel('Survival Rate')
ax1.set_xlabel('PClass')
ax1.set_xticks(PClassPlot.index)
ax1.set_yticks(np.arange(0, 1.1, 1))
ax1.set_title('Class and Survival Rate')

# Sex (性別)
GenderPlot = train_set['Survived'].groupby(train_set['Sex']).mean()
ax2.bar(x = GenderPlot.index, height = GenderPlot.values)
ax2.set_ylabel('Survival Rate')
ax2.set_xlabel('Gender')
ax2.set_xticks(GenderPlot.index)
ax2.set_yticks(np.arange(0, 1.1,.1))
ax2.set_title('Gender and Survival Rate')

# SibSp (同乗中の兄弟/配偶者の数)
SiblingPlot = train_set['Survived'].groupby(train_set['SibSp']).mean()
ax3.bar(x = SiblingPlot.index, height = SiblingPlot.values)
ax3.set_ylabel('Survival Rate')
ax3.set_xlabel('Total Siblings')
ax3.set_xticks(SiblingPlot.index)
ax3.set_yticks(np.arange(0, 1.1, .1))
ax3.set_title('Total Siblings and Survival Rate')

# Parch (同乗中の親/子供の数)
ParchPlot = train_set['Survived'].groupby(train_set['Parch']).mean()
ax4.bar(x = ParchPlot.index, height = ParchPlot.values, width = .8, color = 'Teal')
ax4.set_ylabel('Survival Rate')
ax4.set_xlabel('Number of Parents and Children abroad')
ax4.set_xticks(ParchPlot.index)
ax4.set_yticks(np.arange(0, 1.1,.1))
ax4.set_title('Number of Parents and Children abroad and Survival Rate')

# 欠損地 Age(年齢)は中央値で保管する
train_set['Age'].fillna(train_set['Age'].median(), inplace = True)

# 欠損値 Embarked (出発港)を補完するために，どの港が多いか調べる
# print(train_set['Embarked'].value_counts())
# Sのサウサンプトンが多いのでSで埋める
train_set['Embarked'].fillna('S', inplace = True)
#print(train_set.isnull().sum())

# 処理前の最初の5行を表示
#print(train_set['Sex'].head())
# Sexの値を処理
labelencoder = LabelEncoder()
train_set['Sex'] = Labelencoder.fit_trainsform(train_set['Sex'])

# 処理後の最初の5行を表示
print(train_set['Sex'].head())
