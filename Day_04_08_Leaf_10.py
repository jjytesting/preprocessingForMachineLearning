# leaf keggle에서 가져온 것
# feature가 192개
# 나뭇잎을 수치화한 것
import numpy as np
import pandas as pd

from sklearn import preprocessing, model_selection
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import warnings
warnings.filterwarnings('ignore')

leaf = pd.read_csv('Data/leaf.csv')
#print(leaf.head())
#print(leaf.shape) # (990, 194)

#print(leaf.species)
#print(leaf.species.unique())
#print(len(leaf.species.unique()))

#문제
# species를 0~98 사이의 정수로 인코딩해 보세요.

le = preprocessing.LabelEncoder().fit(leaf.species) #deep learning에서는 Label Binarizer를 쓰고 여기서는 Encoder사용
label = le.transform(leaf.species)
#print(label)
#print(le.classes_)

#leaf 데이터셋으로부터 첫 번째와 두 번째 컬럼을 삭제하세요.
leaf = leaf.drop(['id', 'species'], axis=1)
# print(leaf.shape) # (990, 192) 2열 감소

#데이터 셋을 학습과 검증으로 7대 3으로 나누세요.
'''
data = model_selection.train_test_split(leaf, label, 
                                        train_size=0.7,
                                        test_size = 0.3, 
                                        random_state=42) #데이터가 편중되어 있을 수 있음
train_x, test_x, train_y, test_y = data
print('train shape :', train_x.shape, train_y.shape)
print(' test shape :', test_x.shape, test_y.shape)
#train shape : (693, 192) (693,)
# test shape : (297, 192) (297,)
'''

def show_result(train_x, test_x, train_y, test_y, classifiers):
    print('=' * 50)
    print('{:>30} : {:10} {}'.format('name', 'accuracy', 'loss'))

    for clf in classifiers:
        clf.fit(train_x, train_y)

        train_pred = clf.predict(test_x)

        train_loss = clf.predict_proba(test_x) #cost비용 예측, 딥러닝에서 이게 적으면 좋은거

        acc = accuracy_score(test_y, train_pred)
        loss = log_loss(test_y, train_loss)
        print('{:>30} : {:.4%} {:>9.5}'.format(clf.__class__.__name__, acc, loss))


classifiers = [KNeighborsClassifier(n_neighbors=3),
               SVC(kernel='rbf', C=0.025, probability=True),
               NuSVC(probability=True),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               AdaBoostClassifier(),
               GradientBoostingClassifier(),
               GaussianNB(),
               LinearDiscriminantAnalysis(), # 성능 가장 좋게 나옴 # 선형 판별 분석
               QuadraticDiscriminantAnalysis()]
sss = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=23) #데이터가 편중되지 않도록 자동으로 섞어줌
#10개로 쪼개서 10번 반복함

for train_index, test_index in sss.split(leaf, label):
    train_x, test_x = leaf.values[train_index], leaf.values[test_index]
    train_y, test_y = label[train_index], label[test_index]
    show_result(train_x, test_x, train_y, test_y, classifiers)

#for i in range(10):
#    show_result(train_x, test_x, train_y, test_y, classifiers)