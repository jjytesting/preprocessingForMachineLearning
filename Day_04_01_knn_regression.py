# http://archive.ics.uci.edu/ml/index.php
# http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (model_selection, neighbors, linear_model, preprocessing, metrics)

pd.set_option('display.width', 1000) #오른쪽으로 붙게
df = pd.read_csv('Data/winequality-red.csv', sep=';')
#print(df)

#pd.DataFrame.hist(df, figsize=(20,9))
#plt.show()
# 현재 상태는 각각 단위나 축의 범위가 달라서 비교하기 어려워 보인다. 이런거 전처리 이따 배울거임

all_quality= df.quality.values #품질만 뽑아서 list로 만들기
print(all_quality)
# print(all_quality.unique()) #에러
print((df.quality.unique()))
#345 품질 안좋다, 678 품질 좋다로 분류해보기

bad_quaility = (all_quality <= 5)
print(bad_quaility)

temp = df.groupby('quality').size()
print(temp)
print(' bad :', temp.iloc[:3].sum()) #index 기준
print('good :',temp.iloc[3:].sum())

print(' bad :', temp.loc[:5].sum()) #index에 들어있는 값 기준
print('good :',temp.loc[6:].sum()) #index에 들어있는 값 기준

def not_used():
    plt.figure(figsize=(20, 5))

    plt.subplot(1,2,1)
    plt.hist(all_quality)

    plt.subplot(1,2,2)
    plt.hist(bad_quaility)
    plt.show()

# (1599, 12)
x = df.drop('quality', axis = 1).values # dataframe에서 지정한 것을 제거하고 return 해주는 함수, 맨 마지막 quality만 사라지게 됨
y = (df.quality.values > 5)

print(x.shape, y.shape)

# 기본값은 75: 25 (학습 데이터 셋, 테스트 테이터 셋)
#data = model_selection.train_test_split()
data = model_selection.train_test_split(x, y, random_state=42)
#data = model_selection.train_test_split(x, y, random_state=42, train_size=0.75)
#data = model_selection.train_test_split(x, y, random_state=42, train_size=0.25, test_size=0.25) # 이런식으로 data 다 안쓸 수도 있음
train_x, test_x, train_y, test_y = data
print(train_x.shape, train_y.shape) # 75%일 때: (1199, 11) (1199,)
print(test_x.shape, test_y.shape) #75%일 때: (400, 11) (400,)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x,train_y)
print('score:', knn.score(test_x, test_y))

y_hat = knn.predict(test_x)
# print(y_hat)
print('score :', np.mean(y_hat == test_y))
#-------------------------------------------------#
# 데이터 전처리
x = preprocessing.scale(x) #전체 데이터 평균이랑 분산 맞춰줌
data = model_selection.train_test_split(x, y, random_state=42)
train_x, test_x, train_y, test_y = data
nn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x,train_y)
print('score:', knn.score(test_x, test_y))