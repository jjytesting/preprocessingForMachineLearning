#uci 머신러닝 검색 adult > data folder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import  LabelBinarizer, LabelEncoder
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)
# 직업이 많은데 학습할려면 숫자로 다 바꿔줘야 함. 그래야 모델 비교 가능
# one hot label? 로 만들 수 있음
# 현재 파일로는 변환해야 할 게 너무 많음

def usage_dummies():
    # one hot label 만들기
    print(pd.get_dummies(['a', 'b', 'c', 'a']))
    print(LabelBinarizer().fit_transform(['a','b','c','a']))

    df = pd.get_dummies(['a', 'b', 'c', 'a']) #반환 값 dataFrame
    print(df.index)
    print(df.values)
    print(df.a)
    print('-' * 50)

    print(pd.get_dummies(['a', 'b', np.nan]))
    #print(LabelBinarizer().fit_transform(['a', 'b', np.nan])) #결측치가 있으면 결과가 안나옴 error

    print(pd.get_dummies(['a', 'b', np.nan], dummy_na=True))
    print('-' * 50)

    df = pd.DataFrame({'A': ['a', 'b', 'a'],
                       'B': ['b', 'a', 'c'],
                       'C': [1, 2, 3]})

    print(df) #범주형 데이터가 숫자로 바뀜
    print(pd.get_dummies(df)) # get dummies의 장점
    print(pd.get_dummies(df, prefix = ['c1', 'c2'])) #prefix를 이용해서 column의 이름을 원하는 대로 수정 가능
'''
age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
'''

def logistic_regression1():
    #url = 'Data/adult.txt'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    adult = pd.read_csv(url, header=None, index_col=False, names=names)
    print(adult.shape) #(32561, 15)

    # 숫자형 : 'age', 'hours-per-week'
    # 범주형 : 'workclass', 'education', 'occupation', 'sex', 'income'
    adult = adult[['age', 'workclass', 'education', 'occupation', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']]
    print(adult.shape)
    print(adult.head(3))
    print()

    print('남녀 갯수 :')
    print(adult.sex.value_counts()) #unique 갯수 찾아줌, Male 갯수, female 갯수

    print(adult.groupby('sex').size())
    #print(adult.sex == 'Male') # 안됨.. 그이유는 데이터에 공백이 하나씩 들어가 있어서임
    #print(adult.sex == ' Male')

    print('-'*50)

    adult_dummies = pd.get_dummies(adult)
    print(adult_dummies.head(3))
    print()

    print(adult_dummies.columns)
    print(adult_dummies.columns.values)

    '''
    one hot label로 1, 0으로 들어감
    income_ <=50K  income_ >50K
                1             0
                1             0
    '''

    print('-' * 50)
    #x = adult_dummies.loc[:, 'age': 'sex_ Male']
    x = adult_dummies.loc[:, :'sex_ Male']

    #y = adult_dummies.loc[:, 'income_ <=50K':] # 2차원이라서 에러 (32561, 2)
    y = adult_dummies.loc[:, 'income_ <=50K']

    print(x.shape, y.shape) #(32561, 46)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    print(type(x))          #<class 'pandas.core.frame.DataFrame'>
    print(type(x_train))    #<class 'pandas.core.frame.DataFrame'> pandas 넣어도 machine learning algorithm 계산 잘해준다.

    logreg = LogisticRegression() #2가지 중에서 한개 찾는 건데, 이거 하려면 y값이 1차원 데이터여야 한다.
    logreg.fit(x_train, y_train)

    print('train :', logreg.score(x_train, y_train))
    print(' test :', logreg.score(x_test, y_test))
    #train : 0.832022932023
    # test : 0.829627809851

def logistic_regression2():
    # url = 'Data/adult.txt'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    adult = pd.read_csv(url, header=None, index_col=False, names=names)

    # 숫자형 : 'age', 'hours-per-week'
    # 범주형 : 'workclass', 'education', 'occupation', 'sex', 'income'
    #adult = adult[['age', 'workclass', 'education', 'occupation', 'sex', 'hours-per-week', 'income']]

    age = LabelEncoder().fit_transform(adult.age)
    workclass = LabelEncoder().fit_transform(adult.workclass)
    education = LabelEncoder().fit_transform(adult.education)
    occupation = LabelEncoder().fit_transform(adult.occupation)
    sex = LabelEncoder().fit_transform(adult.sex)
    hours_per_week = LabelEncoder().fit_transform(adult['hours-per-week'])
    income = LabelEncoder().fit_transform(adult.income)

    print('workclass', workclass)   #workclass : [7 6 4 ..., 4 4 5]
    new_adult = np.vstack([age,workclass, education, occupation, sex, hours_per_week]) #income은 그 자체로 y data라서 안 넣어도 됨
    print('shape :', new_adult.shape) #(6, 32561)

    #hstack을 사용하면 수평으로 연결. 결과는 (195366,).
    '''
    new_adult = np.hstack(
        [age, workclass, education, occupation, sex, hours_per_week])  # income은 그 자체로 y data라서 안 넣어도 됨
    print('shape :', new_adult.shape)  # (6, 32561)
    '''

    new_adult = new_adult.T
    print('shape :', new_adult.shape) # shape : (32561, 6)

    x_train, x_test, y_train, y_test = train_test_split(new_adult, income, random_state=0)


    logreg = LogisticRegression()  # 2가지 중에서 한개 찾는 건데, 이거 하려면 y값이 1차원 데이터여야 한다.
    logreg.fit(x_train, y_train)

    print('train :', logreg.score(x_train, y_train))
    print(' test :', logreg.score(x_test, y_test))

    #train : 0.76122031122
    # test : 0.757032305614  더 안좋아짐. 
    '''
    adult_dummies = pd.get_dummies(adult)
    x = adult_dummies.loc[:, :'sex_ Male']
    y = adult_dummies.loc[:, 'income_ <=50K']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    logreg = LogisticRegression()  # 2가지 중에서 한개 찾는 건데, 이거 하려면 y값이 1차원 데이터여야 한다.
    logreg.fit(x_train, y_train)

    print('train :', logreg.score(x_train, y_train))
    print(' test :', logreg.score(x_test, y_test))
    # train : 0.832022932023
    # test : 0.829627809851
    '''
logistic_regression2()
