from sklearn import svm, model_selection, datasets, preprocessing


def show_accuracy(seed):

    cancer = datasets.load_breast_cancer()
    data = model_selection.train_test_split(cancer.data, cancer.target,
                                            random_state=seed) # 75%, 25%로 자동 나눔
    x_train, x_test, y_train, y_test = data

    clf = svm.SVC(C=100)
    clf.fit(x_train, y_train)

    print('origin:', clf.score(x_test, y_test))

    #________________________________________
    #0~1 사이로 스케일링

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train) #transform이 학습하는거
    x_test_scaled = scaler.transform(x_test)

    clf.fit(x_train_scaled, y_train) #clf == classification fit은 확인

    print('minmax :', clf.score(x_test_scaled, y_test)) #

    #___________________________
    # 평균 0, 분산 1로 스케일링
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf.fit(x_train_scaled, y_train)

    print('standard :', clf.score(x_test_scaled, y_test))

import random
for  _ in range(10):
    show_accuracy(random.randrange(100))
    print('-'*50)