from sklearn.preprocessing import (add_dummy_feature, Binarizer, Imputer,
                                   LabelBinarizer, LabelEncoder,
                                   MinMaxScaler)
import numpy as np

# 딥러닝하면서 데이터 전처리 하기 위한 좋은 것.. 이것들을 가지고 수업 기획한 것..

def use_add_dummy_feature():
    # 맨 앞의 칼럼에 1을 채워 줍니다. (bias)
    # 딥러닝 할 때 필요한 경우 많다.
    x = [[1, 0],
         [1, 0]]
    print(add_dummy_feature(x))


    x = [[3, 1, 0],
         [9, 1, 0]]
    print(add_dummy_feature(x))

    x = [[1, 0],
         [1, 0]]
    print(add_dummy_feature(x, value=7))

    # 2차원 데이터만 가능. (n_samples, n_features)
    # x = [1, 0]
    # print(add_dummy_feature(x))

def use_Binarizer():
    x = [[1., -1., 2.],
         [2., 0., 0.],
         [0., 1., -1]]

    binarizer = Binarizer() #계속 쓰기
    binarizer.fit(x)
    print(binarizer.transform(x))

    binarizer = Binarizer(threshold=1.5) #1.5 이하는 모두 0으로 만드는 것 # 한번만 쓰기
    print(binarizer.transform(x))

def use_Imputer():
    # 4 = (1+7) / 2
    # 5 = (2+3+10)/3
    x = [[1,2],
         [np.nan, 3],
         [7, 10]]

    # mean, median, most_frequent(최빈값, 최대 빈도)
    #imp = Imputer()
    imp = Imputer(strategy='mean', axis=0)
    imp.fit(x)
    print(imp.transform(x)) #nan 값 4로 넣어줌. 4가 default 값

    x = [[np.nan, 2],
         [6, np.nan],
         [7, 6]]
    print(imp.transform(x))

    print(imp.statistics_) #계산해서 값을 가지고 있음. learning, 첫 열에는 4를 넣겠다. 두번째 열에는 5를 넣겠다.
    print(imp.missing_values)

def use_LabelBinarizer():
    #one hot label? 한줄에 한개만 1
    x = [1, 2, 6, 4, 2]
    lb = LabelBinarizer()
    lb.fit(x)
    print(lb.transform(x))
    print(lb.classes_) # 어떤 숫자가 어떤 위치에 켜지는지 볼 수 있음

    lb2 = LabelBinarizer(sparse_output=True) #dense or sparse
    lb2.fit(x)
    print(lb2.transform(x))
    #뜨문 뜨문 sparse한 경우 압축해서 이렇게 만들기도 함. 근데 모든 배열이 다 되는 것은 아님.

    lb3 = LabelBinarizer(neg_label=-1, pos_label=2) #0과 1 대신 다른 값으로 변경할 수 있다. 지정 가능하다.
    lb3.fit(x)
    print(lb3.transform(x))
    print('-' * 50)

    lb4 = LabelBinarizer()
    print(lb4.fit_transform(['yes', 'no', 'no', 'yes'])) #fit(학습) transform(예측) 한번에 하자

    x = ['yes', 'no', 'no', 'yes', 'cancel']
    lb5 = LabelBinarizer()
    lb5.fit(x)
    print(lb5.fit_transform(x))
    print(lb5.classes_)

    inverse_x = lb5.transform(x)
    print(inverse_x)
    print(lb5.inverse_transform(x))
    print(inverse_x)
    print(lb5.inverse_transform(inverse_x))

use_LabelBinarizer()

def useLabelEncoder(): #binarizer와 사실은 같은거임
    x = [2, 1, 2, 6]
    le = LabelEncoder() #deep learning에서는 Label Binarizer를 쓰고 여기서는 Encoder사용
    le.fit(x) # 사후평가 predict, score, fit, transform 은 낼거임
    print(le.transform(x))
    print(le.classes_)

    inverse_x = le.transform(x)
    print(inverse_x)

    #원래 상태로 돌려보기
    print(le.inverse_transform(inverse_x))
    print('-' * 50)

    x = ['paris', 'tokyo', 'paris', 'amsterdam']
    le2 = LabelEncoder()
    le2.fit(x)
    print(le2.classes_)

    inverse_x = le2.transform(x)
    print(inverse_x)
    print(le.inverse_transform(inverse_x))
    print(le.inverse_transform(inverse_x))
    print(le.inverse_transform([0, 0, 1, 1 ,2 ,2]))

def use_MeanMaxScaler():
    x= [[1., -1., 2.],
        [2., 0., 0.],
        [0., 1., -1]]
    scaler = MinMaxScaler()
    scaler.fit()
    print(scaler.transform(x))