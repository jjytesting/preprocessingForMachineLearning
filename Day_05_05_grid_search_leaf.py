#문제
# leaf.csv 파일을 사용해서
# svm을 적용한 그리드 서치 + 교차검증을 구현해 보세요.
# 마지막에는 히트맵도 그려봅니다.

from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV) #CV cross validation을 하는 grid search .. grid를 돌아다니면서 좋은거 찾아다님.
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

def grid_search_cv(x_train, x_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100] }
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    print('test score :', grid_search.score(x_test, y_test)) # test set 안나눠도 얘가 알아서 다 해줌.
    print('best score :', grid_search.best_score_)
    print('best param :', grid_search.best_params_)
    #test score : 0.973684210526
    #best score : 0.964285714286
    #best param : {'C': 100, 'gamma': 0.01}
    return grid_search, param_grid

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
    #print(results.head(5).T)

    scores = np.array(results.mean_test_score).reshape(6, 6)
    #print(scores)

    draw_heatmap(scores, param_grid)
    plt.show()

leaf = pd.read_csv('Data/leaf.csv')

le = preprocessing.LabelEncoder().fit(leaf.species) #deep learning에서는 Label Binarizer를 쓰고 여기서는 Encoder사용
label = le.transform(leaf.species)
leaf = leaf.drop(['id', 'species'], axis=1)

#data = train_test_split(leaf, )
#x_train = StandardScaler().fit_transfor(x_train) #스케일링 하면 결과가 훨씬 잘나옴

sss = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=23) #데이터가 편중되지 않도록 자동으로 섞어줌

for train_index, test_index in sss.split(leaf, label):
    x_train, x_test = leaf.values[train_index], leaf.values[test_index]
    y_train, y_test = label[train_index], label[test_index]


cv_pandas_heatmap(x_train, x_test, y_train, y_test)
