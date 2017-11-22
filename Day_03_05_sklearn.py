# tensorflow 나오기 전에는 이걸로 머신러닝 했다.
from sklearn import datasets, svm, random_projection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
import pickle #binary 형식으로 데이터 저장 혹은 읽어옴
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

#http://pythonkim.tistory.com/78
# scipy‑1.0.0‑cp36‑cp36m‑win_amd64.whl
# numpy‑1.13.3+mkl‑cp36‑cp36m‑win_amd64.whl

def not_used_1():
    iris = datasets.load_iris()
    print(type(iris)) #dictionary에서 사용하는 <class 'sklearn.utils.Bunch'>
    print(iris.keys())

    # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    print(iris['target_names'])
    print(iris['feature_names']) # column
    print(iris['data'])
    print(iris['target'])
    print(iris['DESCR']) #설명

    print(type(iris['data'])) # <class 'numpy.ndarray'>

def outline(number):
    number = number.reshape(-1,8)

    for row in number:
        for col in row:
            ch = 1 if col>9 else ' '
            print(ch, end=' ')

        print()

def not_used_2():
    digits = datasets.load_digits()
    print(digits.keys())
    # ['data', 'target', 'target_names', 'images', 'DESCR']
    print(digits['images'])
    print(digits.data)
    print(digits.data.shape) # toy dataset (1797, 64)
    print(digits.data[0].reshape(-1,8))
    print(digits.data[1].reshape(-1,8))

    for i in range(10):
        print(digits.target[i])
        outline(digits.data[i])
        print('-'*50)


    print(digits.images.shape) # (1797, 8, 8)
    print(digits.data.shape) # (1797, 64) 사실을 같은 거임

def not_used_3():
    digits = datasets.load_digits()

    clf = svm.SVC(gamma=0.001, C=100.)
    fit = clf.fit(digits.data[:-1], digits.target[:-1])
    #print(clf)
    #print(fit)

    clf.fit(digits.data[:-1], digits.target[:-1]) #training

    pred = clf.predict(digits.data[-1:])
    print(pred)
    print(digits.target[-1])

    #____________________________________#

    s = pickle.dumps(clf) #string 변수로 저장
    clf2 = pickle.loads(s) # string 변수에서 읽어오기
    print(clf2.predict(digits.data[-1:]))
#not_used_3()


def not_used_4():
    digits = datasets.load_digits()

    train_count = int(len(digits.data) * 0.8)

    clf = svm.SVC(gamma=0.001, C=100.) #신경망 이전 알고리즘. 어떻게 되는지 원리 강사도 모름. 관심 없음. 목적은 간략한 알고리즘 사용 방법 알려주기임.
    clf.fit(digits.data[:train_count], digits.target[:train_count])

    y_hat = clf.predict(digits.data[train_count:]) #예측 값
    label = digits.target[train_count:] #정답
    print(y_hat)
    print(label)
    print(y_hat == label)
    print(np.mean(y_hat == label))

def not_used_5():
    # 문제
    # iris 데이터셋을  SVM에 적용해 봅니다.
    # 전체 데이터셋으로 학스바혹, 처음 3개에 대해서 예측해 봅시다.
    iris = datasets.load_iris()
    print(type(iris)) #dictionary에서 사용하는 <class 'sklearn.utils.Bunch'>
    print(iris.keys())

    clf = svm.SVC(gamma=0.001, C=100.)
    # fit = clf.fit(iris.data[:-1], iris.target[:-1])
    clf.fit(iris.data[:-1], iris.target[:-1])

    # 정수 레이블 -> 정수 결과
    pred = clf.predict(iris.data[:3])
    print(pred)
    print(iris.target[:3])

    print(iris['target_names'])
    print(iris['target'])
    #print(iris['target_names'][iris['target']])
    print(iris.target_names[iris.target])

    clf.fit(iris.data, iris.target)
    print(clf.predict(iris.data[:3]))

    # 문자열 레이블 -> 문자열 결과
    clf.fit(iris.data, iris.target_names[iris.target])
    print(clf.predict(iris.data[:3]))

def not_used_6():
    digits = datasets.load_digits()

    data = model_selection.train_test_split(digits.data, digits.target, train_size= 0.7) #실수는 %

    print(len(data))
    train_x, test_x, train_y, test_y = data
    print(train_x.shape, test_x.shape) #(1257, 64) (540, 64)
    print(train_y.shape, test_y.shape) #(1257,) (540,)

    data = model_selection.train_test_split(digits.data, digits.target, train_size= 1300) #정수는 갯수

    train_x, test_x, train_y, test_y = data
    print(train_x.shape, test_x.shape) #(1300, 64) (497, 64)
    print(train_y.shape, test_y.shape) #(1257,) (540,)

    clf = svm.SVC(gamma=0.001, C=100.)

    clf.fit(train_x, train_y)
    y_hat = clf.predict(test_x)

    print(y_hat) #예측
    print(test_y) #정답

    print(np.mean(y_hat == test_y)) #비교

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)

#scatter_matrix(df)
scatter_matrix(df, c=iris.target, hist_kwds={'bins':20}) #hist_kwds : 구간 20개로 쪼개기

plt.show()