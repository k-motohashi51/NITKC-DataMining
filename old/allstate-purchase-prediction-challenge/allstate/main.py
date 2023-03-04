import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def main():
    train = pd.read_csv('data/train.csv/train.csv')
    test = pd.read_csv('data/test_v2.csv/test_v2.csv')

    train, test = preprocess(train, test)
    ans_list = multi(train, test)

    write_csv(test, ans_list)

# データの前処理
def preprocess(train, test):
    #print(train.info())
    #print(train['state'].value_counts())
    #print(train['car_value'].value_counts())
    #print(test.isnull().sum())

    # 欠損値補完
    train['car_value'].fillna('e', inplace = True)
    train['risk_factor'].fillna((train['risk_factor'].median()), inplace = True)
    train['C_previous'].fillna((train['C_previous'].median()), inplace = True)
    train['duration_previous'].fillna((train['duration_previous'].median()), inplace = True)
    test['location'].fillna((test['location'].median()), inplace = True)
    test['car_value'].fillna('e', inplace = True)
    test['risk_factor'].fillna((train['risk_factor'].median()), inplace = True)
    test['C_previous'].fillna((train['C_previous'].median()), inplace = True)
    test['duration_previous'].fillna((train['duration_previous'].median()), inplace = True)
    #print(test.isnull().sum())

    # 時間を抜く
    train = train.drop('time', axis = 1)
    test = test.drop('time', axis = 1)

    # 数値変換
    label_enc = LabelEncoder()
    train['state'] = label_enc.fit_transform(train['state'])
    train['car_value'] = label_enc.fit_transform(train['car_value'])
    test['state'] = label_enc.fit_transform(test['state'])
    test['car_value'] = label_enc.fit_transform(test['car_value'])

    #for col_name in train:
    #    print(col_name + '------------------')
    #    print(train[col_name].value_counts())

    #  rechord-typeが1のやつのときが実際に契約した時のやつ

    return train, test

    
def multi(train, test):
    train_X = train.drop(['A', 'B', 'C', 'D', 'E', 'F', 'G'], axis = 1)
    test_X = test.drop(['A', 'B', 'C', 'D', 'E', 'F', 'G'], axis = 1)
    train_y = train.A

    pred = [0] * 7

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state = 0)
    train_y = train.A
    clf = clf.fit(train_X, train_y)
    pred[0] = clf.predict(test_X)
    print('predictA Ok')
    train_y = train.B
    clf = clf.fit(train_X, train_y)
    pred[1] = clf.predict(test_X)
    print('predictB Ok')
    train_y = train.C
    clf = clf.fit(train_X, train_y)
    pred[2] = clf.predict(test_X)
    print('predictC Ok')
    train_y = train.D
    clf = clf.fit(train_X, train_y)
    pred[3] = clf.predict(test_X)
    print('predictD Ok')
    train_y = train.E
    clf = clf.fit(train_X, train_y)
    pred[4] = clf.predict(test_X)
    train_y = train.F
    clf = clf.fit(train_X, train_y)
    pred[5] = clf.predict(test_X)
    train_y = train.G
    clf = clf.fit(train_X, train_y)
    pred[6] = clf.predict(test_X)

    ans_list = []
    for i in range(len(test_X)):
        ans_str = ''
        for j in range(7):
            ans_str = ans_str + str(pred[j][i])
        print(ans_str)
        ans_list.append(ans_str)
    
    print(ans_list)

    return ans_list

def write_csv(df, ans_list):
    cid = df['customer_ID'].tolist()
    ans_df = pd.DataFrame({'customer_ID':cid, 'plan':ans_list})
    print(ans_df)
    ans_df.to_csv('Submission.csv', index = False)


if __name__ == '__main__':
    main()