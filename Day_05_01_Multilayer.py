#딥러닝에서 성능을 높이는 방법 2가지
#1. layer 여러개 만들기
#2. layer 넓게 만들기
# 이건 layer 여러개 만드는 것에 대한 것

import  matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.keys()) # bunch라고 부르는 자료 구조 안에 있음
# DESCR 데이터 설명

#print(cancer)
print(len(cancer.values()))
print('max by feature')
print(cancer.data.max(axis=0)) #수직 방향. 열 우선
print(cancer.target) #양성인지 음성인지 1, 0

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

#레이어 : 기본 100개
mlp1 = MLPClassifier(random_state=42) #Multi-layer Perceptron classifier.
mlp1.fit(x_train, y_train) #가중치 저장

#가중치 활용해서 채점
print('train :', mlp1.score(x_train, y_train)) # 정확도 계산 score 함수, 원본 데이터가 x_train
print(' test :', mlp1.score(x_test, y_test)) # 정확도를 계산하는 거다 보니까 정답도 넣어주어야 함. 정답이 있는거라 지도학습이다.

#train : 0.906103286385
# test : 0.881118881119
# 이정도는 사용하기 어려운 정확도의 값이다.

#___________________________________________________

# standardization.
# 0하고 1은 굳이 재조정 할 필요 없으니 x값만 재조정함
mean_on_train = x_train.mean(axis=0)
std_on_train = x_train.std(axis=0)

x_train_scaled = (x_train - mean_on_train) / std_on_train
x_test_scaled = (x_test - mean_on_train) /std_on_train

# max_iter은 기본 200회
mlp2 = MLPClassifier(random_state=0)
mlp2.fit(x_train_scaled, y_train)

print(' iter :', mlp2.max_iter)
print('train :', mlp2.score(x_train_scaled, y_train))
print(' test :', mlp2.score(x_test_scaled, y_test))

#scaling을 하니까 결과가 높아진다....
#train : 0.990610328638
#test: 0.965034965035
mlp3 = MLPClassifier(random_state=0, max_iter=1000, alpha=1)
mlp3.fit(x_train_scaled, y_train)

print(' iter :', mlp3.max_iter)
print('alpha :', mlp3.alpha)
print('train :', mlp3.score(x_train_scaled, y_train))
print(' test :', mlp3.score(x_test_scaled, y_test))
# iter : 1000
#alpha : default(0.0001) #learning 비율 학습을 느슨하게 하겠다. 기본값. 값을 높이면 빠르게 learning하지만 발산이라고 튕길 수 있다.
#train : 0.992957746479
# test : 0.972027972028

#adam이 최적화된 optimizer , alpha를 바꿔도 크게 차이 안남.
#train : 0.988262910798
# test : 0.972027972028
mlp4 = MLPClassifier(random_state=0, max_iter=1000, alpha=1)
mlp4.fit(x_train_scaled, y_train)

print(' iter :', mlp4.max_iter)
print('alpha :', mlp4.alpha)
print('train :', mlp4.score(x_train_scaled, y_train))
print(' test :', mlp4.score(x_test_scaled, y_test))

print(len(mlp4.coefs_))
print(mlp4.coefs_[0].shape)
print(mlp4.coefs_[1].shape)

plt.figure(figsize=(20, 5))
ax = plt.gca()
im = ax.imshow(mlp4.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('hidden units')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2%', pad=0.1) #pad가 작으면 달라붙고, 아니면 커짐..
plt.colorbar(im, cax)
plt.show()
