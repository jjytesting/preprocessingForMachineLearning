import numpy as np

np.random.seed(1) # 값 고정하기
# print(np.random.random(2, 3))
print(np.random.random((2, 3))) #매개변수는 하나여야 함, 균등분포(균등하게) , 정규분포 (중앙 많이 배치), default는 0과 1사이 값
print(np.random.uniform(size=(2, 3)))
print(np.random.rand(2, 3))
# 모두 균등 분포임

print(np.random.randn(2,3))

a = np.random.choice(range(10), 12)
print(a)

b = a.reshape(-1, 4)
print(b)

#딥러닝에서 sum 이용해서 합계 계산
print(np.sum(b))
#전체 말고 어떤 축을 따라 더하는 게 중요해짐
print(np.sum(b, axis=0)) #수직 계산, 중요중요, 열(column)
print(np.sum(b, axis=1)) #수평 계산, 중요중요, 행(row)

print('-' * 50)

def func(y, x):
    return y*10 + x

a = np.fromfunction(func, shape=(5, 4), dtype = np.int32) #function을 이용해서도 채워 넣을 수 있음
a = np.fromfunction(lambda y, x: y*10+x, shape=(5, 4), dtype = np.int32) #python에서는 lambda 사용
print(a)
print(a[0])
print(a[0][0]) #좋지 않은 연산, a[0]를 불러온 다음에 또 [0]함
print(a[0, 0]) #fancy indexing 한번에 접근하기

print(a[0:2,0]) #slicing 가능
print(a[1:3,1:3])

#문제
#거꾸로 출력해보세요.
print(a[::-1,::-1])

#행과 열을 바꾸어서 출력해보세요.
print(a.T)
for i in range(a.shape[-1]):
    print(a[:,i])

# 전체 행렬의 특정 열 구하는 문제 많이 나옴

print('-'*50)

a = np.arange(6).reshape(-1, 3)
print(a)

a[0] = 99 #0번째 행 모두 바꿔버림
print(a)

a[:, 0] = 88  #0번째 열 모두 바꿔버림
print(a)

a[:,::2] = 77
print(a)

# fancy indexing은 list에서는 동작하지 않는다. list는 배열이 아니기 때문에

# 문제
# 대각선이 1로 채워진 5X5 행렬
a = np.arange(25).reshape(5,5)
print(a)

a[:,:] = 0
print(a)

for i in range(a.shape[-1]):
    a[i,i] = 1
print(a)

print(np.eye(5,5))

b = np.zeros((5,5))
b[range(5), range(5)] = 1
b[[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]] = 1
print(b)

# 문제
# 테두리가 1로 채워진 5X5 배열을 만드세요.
# 나머지는 0

a = np.zeros((5,5))
#a[0, :] = 1
#a[-1, :] = 1
a[0], a[-1] = 1, 1
a[:, 0],a[:, -1] = 1, 1
print(a)

b = np.ones((5, 5))
b[1:-1, 1:-1] = 0 #한번에 계산하기
print(b)

print('-'*50)

a = np.arange(10)
print(a)

print(a[[1, 4, 7]])
# print(a[1, 4, 7]) # error - fancy indexing 문법에 해당된다.

b = a.reshape(-1, 5)
print(b)
print(b[[0]])
print(b[[1, 0]])
print(b[[0, 1], [2, 3]]) #0행 2열, 1행 3열
b[[0, 1], [2, 3]] = 99 #numpy에서 읽을 수 있다는 것은 대입할 수 있다는 것임
print(b)

c = b > 5
print(c)
print(b[c]) #true인 것만 뽑아줌

print('-'*50)

a = np.array([3, 1, 2])
print(np.sort(a))
print(a)

b = np.argsort(a) #정렬되었을 때의 index를 리턴
print(b)
print(a[b])

x = np.array([4, 3, 1, 5, 2])
y = np.argsort(x) # 2 4 1 0 3
print(y)

print('-' * 50)

#onehot encoding
a = [1, 3, 0, 3]

# a를 아래처럼 만들어 보세요 (np.max)
# [[0 1 0 0] 1 아 index값으로 활용해서 1 넣으라는 거구나
# [0 0 0 1] 3
# [1 0 0 0] 0
# [0 0 0 1]] 3
b = np.zeros([len(a), np.max(a)+1])
b[range(len(a)), a] = 1
print(b)

n = np.max(a)+1
b = np.eye(n, n)[a] # 단위 행렬을 가져다가 원하는 걸 끄집어 내기... 와우.. numpy 답대
print(b)

