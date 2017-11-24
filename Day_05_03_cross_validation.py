from sklearn.datasets import  make_blobs, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score, KFold, LeaveOneOut, ShuffleSplit, GroupKFold)
import numpy as np

def simple_test():
    x, y = make_blobs(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    print('score :', logreg.score(x_test, y_test))
    #score : 0.88
    #신뢰할 수 없는 데이터 1번만 돌려서

def cross_validation():
    iris = load_iris()
    logreg = LogisticRegression()

    # 회귀 : KFold
    # 분류 : StratifiedKFold
    # cv는 기본 3번


    #기본 3번 반복.
    scores = cross_val_score(logreg, iris.data, iris.target)
    print('scores (3-folds) :', scores) #세겹
    print('  mean (3-folds) :', scores.mean())
    #scores : [ 0.96078431  0.92156863  0.95833333]


    #5번 반복
    scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
    print('scores (5-folds) :', scores) #세겹
    print('  mean (5-folds) :', scores.mean()) #세겹

    #여러번에 걸쳐서 test를 했는데 이정도면 믿을만하다고 할 수 있음 ? 근데 편차가 너무 커서 신뢰하기 어려움. 편차가 큰 이유는 분포가 한쪽으로 치우쳐 있거나 데이터가 너무 작아서 평가하기 어려웠거나 할 수 있다.

def usage_Kfold():
    iris = load_iris()

    # 총 150개의 data
    # 분할 갯수. 반복 횟수는 분할 갯수와 동일하다.
    sp1 = KFold()
    for train_index, test_index in sp1.split(iris.data, iris.target):
        print(len(train_index), len(test_index))
    print() #2/3은 train 사용, 1/3은 test가 씀

    sp2 = KFold(n_splits=5)
    for train_index, test_index in sp2.split(iris.data, iris.target):
        print(len(train_index), len(test_index))
    print() #4/5는 train 사용, 1/5는 test 사용

    #fold 순서
    sp3 = KFold()
    for train_index, test_index in sp3.split(iris.data, iris.target):
        print(test_index[:10]) #인덱스 출력
    print()
    #[0 1 2 3 4 5 6 7 8 9]
    #[50 51 52 53 54 55 56 57 58 59]
    #[100 101 102 103 104 105 106 107 108 109]

    sp4 = KFold(n_splits=5)
    for train_index, test_index in sp4.split(iris.data, iris.target):
        print(test_index[:10])
        print(iris.data[test_index])
    print()
    #[0 1 2 3 4 5 6 7 8 9]
    #[30 31 32 33 34 35 36 37 38 39]
    #[60 61 62 63 64 65 66 67 68 69]
    #[90 91 92 93 94 95 96 97 98 99]
    #[120 121 122 123 124 125 126 127 128 129]

    #KFold 내부 구조 확인
    sp5= KFold()
    folds = list(sp5.split(iris.data, iris.target)) # generator임, 반복문을 모두 반복한 결과를 lits로 만듬
    print(folds)
    print(len(folds))  # 3. n_splits가 3이니까
    print(folds[0])
    print(len(folds[0])) # 2. train과 test가 2개니까.
    train, test = folds[0]
    print(len(train))   #100, 150개를 3개로 나눠서 2개 사용.
    print(len(test))    #50, 150개를 3개로 나눠서 1개 사용.


def cv_detail():
    iris = load_iris()
    logreq = LogisticRegression()

    print('n_splits : 3')
    print(cross_val_score(logreq, iris.data, iris.target, cv = KFold())) #estimater : logistic regression
    # cv = 3       : [ 0.96078431 0.92156863 0.95833333]
    # cv = KFold() : [0. 0. 0.]
    # 데이터가 50개씩 나눠져 있기 때문에 나쁜 결과.
    print('n_splits : 5')
    print(cross_val_score(logreq, iris.data, iris.target, cv=KFold(n_splits=5)))  # estimater : logistic regression
    # [ 1.          0.93333333  0.43333333  0.96666667  0.43333333]

    print('n_splits: 3_shuffle')
    print(cross_val_score(logreq, iris.data, iris.target, cv = KFold(shuffle=True, random_state=0))) #shuffle이 없으면 random_state 의미 없음.

    # n_splits : 3 shuffle
    # [ 0.9 0.96, 0.96]


    print('n_splits: loocv')
    loocv = cross_val_score(logreq, iris.data, iris.target, cv = LeaveOneOut())

    print('n_splits:', len(loocv))
    print('  score :', loocv.mean())

    # n_splits: loocv
    # n_splits: 150
    # score: 0.953333333333

    print('n_splits: 150')
    spl150 = cross_val_score(logreq, iris.data, iris.target, cv=KFold(n_splits=150))

    print('n_splits:', len(spl150))
    print('  score :', spl150.mean())
    # 총 150개여서 150개로 나눈 다는 것은 leave one out이랑 결과가 같을 수 밖에 없음

def cv_shuffle_split():
    iris = load_iris()
    logreg = LogisticRegression()

    # sp = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10, random_state=0)
    # sp = ShuffleSplit(test_size=0.5, n_splits=10, random_state=0)
    sp = ShuffleSplit(train_size=0.5, n_splits=10, random_state=0)

    scores = cross_val_score(logreg, iris.data, iris.target, cv=sp)
    print(scores)
    print('mean :', scores.mean())
    #[0.84        0.93333333  0.90666667  1.          0.90666667  0.93333333
    # 0.94666667  1.          0.90666667  0.88]
    # mean : 0.925333333333

    print('train size :', sp.train_size)
    print(' test size :', sp.test_size)
    #train size: None
    #test size: 0.5
    #train size : 0.5
    #test size : default

def usage_ShuffleSplit():
    iris = load_iris()

    sp1 = ShuffleSplit(train_size=0.6, test_size=0.4, n_splits=3)
    for train_index, test_index in sp1.split(iris.data, iris.target):
        print(len(train_index), len(test_index)) #90 60
    print()

    # test_size를 전달하지 않으면 기본 15개 사용
    sp2 = ShuffleSplit(train_size=0.6, n_splits=3)
    for train_index, test_index in sp2.split(iris.data, iris.target):
        print(len(train_index), len(test_index)) #90 15 ?? test size의 default 값이 15개임. 나머지는 사용하지 않는다.
    print()

    # train_size를 전달하지 않으면 나머지 전부 사용
    sp3 = ShuffleSplit(test_size=0.6, n_splits=3)
    for train_index, test_index in sp3.split(iris.data, iris.target):
        print(len(train_index), len(test_index)) #60 90 일관성이 없음. 그래서 그냥 2개를 다 주는 게 나음.
    print()

    #전체 테스트셋을 모아서 하나로 출력
    total = []
    sp4 = ShuffleSplit(train_size=100, test_size=50, n_splits=3)
    for train_index, test_index in sp4.split(iris.data, iris.target):
        print(len(train_index), len(test_index))  # 100 50
        #print(test_index[:10])
        #print(iris.data[test_index])
        total += list(test_index)
    print(len(total))
    total_sorted = np.sort(total)
    print(total_sorted) # 150개 섞기 하면 중복 안할 것 같은데 중복이 많이 들어 있음
    # 아마 셔플하고 100개 50개 꺼내기 이런식으로 하는 것 같음
    # 중복된 숫자들 발생

def group_kfold():
    logreg = LogisticRegression()
    x, y = make_blobs(n_samples=12, random_state=0)

    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
    scores = cross_val_score(logreg, x, y, groups, cv = GroupKFold(n_splits=3))
    print('scores')
    print(scores)
    # scores
    # [0.75        0.8         0.66666667]

    # 3개로 분할
    # 분할 시 내가 원하는 그룹이 몰려다니게 하는 것, 위에서 설정한 groups 의 같은 숫자끼리 몰려다님
    sp1 = GroupKFold()
    for train_index, test_index in sp1.split(x, y, groups):
        print(train_index, test_index)
    print()
    #[ 0  1  2  7  8  9 10 11] [3 4 5 6]
    #[0 1 2 3 4 5 6] [ 7  8  9 10 11]
    #[ 3  4  5  6  7  8  9 10 11] [0 1 2]

    #4개로 분할
    sp2 = GroupKFold(n_splits=4)
    for train_index, test_index in sp2.split(x, y, groups):
        print(train_index, test_index)
    print()
    #[ 0  1  2  7  8  9 10 11] [3 4 5 6]
    #[0 1 2 3 4 5 6 7 8] [ 9 10 11]
    #[ 3  4  5  6  7  8  9 10 11] [0 1 2]
    #[ 0  1  2  3  4  5  6  9 10 11] [7 8]

    #5개로 분할 : 종류가 4개 밖에 없어서 5개는 불가
    '''
    sp3 = GroupKFold(n_splits=5)
    for train_index, test_index in sp3.split(x, y, groups):
        print(train_index, test_index)
    print()
    '''

    #어떤 형태 하나에 대해 검증하는 것이 아니라 돌려가면서 검증하므로 여러 데이터 셋에 대해서 검증
    #일반화 성능 향상 !!!!!

group_kfold()