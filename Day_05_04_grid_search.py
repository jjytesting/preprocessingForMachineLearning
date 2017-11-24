from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV) #CV cross validation을 하는 grid search .. grid를 돌아다니면서 좋은거 찾아다님.
# cross validation 일반화에 대한 성능 향상.

from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width' ,1000)

def simple_grid_search(x_train, x_test, y_train, y_test):
    best_score, best_parameter = 0, {}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            #gamma 6개 c 6개 6X6 형태(grid) 로 만들어서 search함
            # 시간이 많이 걸림
            # 한번에 할거 36번하기 때문
            # 그러나 다른 것도 시간 걸림
            clf = SVC(gamma=gamma, C=C)
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)

            if best_score < score:
                best_score = score
                best_parameter = {'gamma':gamma, 'C':C}

    print('best score :', best_score)
    print('best param :', best_parameter)
    #best score : 0.973684210526
    #best param : {'gamma': 0.001, 'C': 100}

def better_grid_search(x_total, x_test, y_total, y_test):
    x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, random_state=0)
    print(x_train.shape, x_valid.shape, x_test.shape)
    # (84, 4) (28, 4) (38, 4)

    best_score, best_parameter = 0, {}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            # gamma 6개 c 6개 6X6 형태(grid) 로 만들어서 search함
            # 시간이 많이 걸림
            # 한번에 할거 36번하기 때문
            # 그러나 다른 것도 시간 걸림
            clf = SVC(gamma=gamma, C=C)
            clf.fit(x_train, y_train)
            score = clf.score(x_valid, y_valid)

            if best_score < score:
                best_score = score
                best_parameter = {'gamma': gamma, 'C': C}

    #clf = SVC(gamma=best_parameter['gamma'], C=best_parameter['C'])
    clf = SVC(**best_parameter) # dictionary를 풀어줌

    clf.fit(x_total, y_total) # 학습
    score = clf.score(x_test, y_test) # 검증


    print('test score :', score)
    print('best score :', best_score)
    print('best param :', best_parameter)
    #test score : 0.973684210526
    #best score : 1.0
    #best param : {'gamma': 0.001, 'C': 100}

def cv_grid_search(x_train, x_test, y_train, y_test):
    best_score, best_parameter = 0, {}
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            # gamma 6개 c 6개 6X6 형태(grid) 로 만들어서 search함
            # 시간이 많이 걸림
            # 한번에 할거 36번하기 때문
            # 그러나 다른 것도 시간 걸림
            clf = SVC(gamma=gamma, C=C) #분류기

            scores = cross_val_score(clf, x_train, y_train, cv=5) #estimater # 얘는 cross_val은 cv가 처리하는 것임
            #이게 random을 가지고 있는게 아니라 cv= KFold(shuffle 이런식으로 되면서 거기서 random..
            # 이건 평균이 가장 좋은 것을 넣는 거라 앞의 얘들보다 신뢰할 수 있음

            if best_score < scores.mean():
                best_score = scores.mean()
                best_parameter = {'gamma': gamma, 'C': C}

    clf = SVC(**best_parameter)  # dictionary를 풀어줌

    clf.fit(x_train, y_train)  # 학습
    score = clf.score(x_test, y_test)  # 검증

    print('test score :', score)
    print('best score :', best_score)
    print('best param :', best_parameter)
    # test score : 0.973684210526
    # best score : 0.972689629211
    # best param : {'gamma': 0.01, 'C': 100}

def grid_search_cv(x_train, x_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100] }
    grid_search = GridSearchCV(SVC(), param_grid=param_grid)
    grid_search.fit(x_train, y_train)

    print('test score :', grid_search.score(x_test, y_test)) # test set 안나눠도 얘가 알아서 다 해줌.
    print('best score :', grid_search.best_score_)
    print('best param :', grid_search.best_params_)
    #test score : 0.973684210526
    #best score : 0.964285714286
    #best param : {'C': 100, 'gamma': 0.01}
    return grid_search, param_grid

def draw_bad_heatmap(scores, param_grid):
    plt.figure(2, figsize=(12, 5))
    param_linear = {'C': np.linspace(1, 2, 6), 'gamma': np.linspace(1, 2, 6)}
    param_onelog = {'C': np.linspace(1, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
    param_range = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-7, -2, 6)}
    # linearspace 1과 2의 구간을 6개로 나누겠다. 각각 결과 36개임

    for i, param_grid in enumerate([param_linear, param_onelog, param_range]):
        grid_search = GridSearchCV(SVC(), param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

        plt.subplot(1, 3, i+1)
        draw_heatmap(scores, param_grid)

def draw_heatmap(scores, param_grid):
    ax = plt.gca()
    img = ax.pcolor(scores, cmap = 'viridis')
    img.update_scalarmappable()
    ax.set_xlabel('gamma')
    ax.set_ylabel('C')
    ax.set_xticks(np.arange(len(param_grid['gamma']))+0.5)
    ax.set_yticks(np.arange(len(param_grid['C']))+0.5)
    ax.set_xticklabels(param_grid['gamma'])
    ax.set_yticklabels(param_grid['C'])
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        c = 'k' if np.mean(color[:3]) > 0.5 else 'w'
        ax.text(x, y, '{:.2f}'.format(value), color=c, ha='center', va='center')


def cv_pandas_heatmap(x_train, x_test, y_train, y_test):
    grid_search, param_grid = grid_search_cv(x_train, x_test, y_train, y_test)
    results = pd.DataFrame(grid_search.cv_results_)
    #print(results)
    print(results.head(5).T)

    scores = np.array(results.mean_test_score).reshape(6, 6)
    print(scores)

    plt.figure(1)
    draw_heatmap(scores, param_grid)
    plt.figure(2)
    draw_bad_heatmap(x_train, y_train) #일부러 만들어 낸 좋지 않은 상황
    plt.show()


#비대칭 매개 변수 그리드 서치
def different_params(x_train, x_test, y_train, y_test):
    param_grid = [{'kernel': ['rbf'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
                  {'kernel': ['linear'], # linear는 c 필요 없음. 서로 다를 때 비대칭 매개변수라고 한다.
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}]
    grid_search = GridSearchCV(SVC(), param_grid, cv = 5)
    grid_search.fit(x_train, y_train)

    print('test score :', grid_search.score(x_test, y_test))
    print('best score :', grid_search.best_score_)
    print('best param :', grid_search.best_params_)

    # 어떤 커널이 좋은지 판단하기 위해서 쓰면 좋다. rbf 커널이 좋다고 결과 나옴
    # best param 을 이렇게 찾는 건 좋지 않다.
    # 주변에 뭐 모여있을 수도 있다. 2등이나 3등이 베일에 가려질 수 있다.

iris = load_iris()
#print(iris)
data = train_test_split(iris.data, iris.target, random_state=0) #이게 난수 들어가서 계속 바뀜 안바뀌게 하려면 random_state=0 너어야 함.
#print(data)
x_train, x_test, y_train, y_test = data
#gamma = kernel의 크기, c = constraint의 크기

#simple_grid_search(*data)
#better_grid_search(*data) #tuple을 풀어줌
#cv_grid_search(*data)
#grid_search_cv(*data)
#cv_pandas_heatmap(*data)
#different_params(*data)

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# 구간을 전달
# 그리드 서치는 값을 넣어서 갑 매핑 시킨다면
# 이거는 구간을 넣어서 구간에 대한 매핑한단다.

print(x_test)