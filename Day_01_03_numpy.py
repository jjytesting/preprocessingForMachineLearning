import numpy as np

def not_use():
    print(np.arange(10)) #배열에 대한 range
    print(np.arange(10, 20))
    print(np.arange(10, 20, 2))
    print(np.arange(5, -5, -1))

    print(type(np.arange(10)))

    #다차원 배열이라는 의미 n dimensional array
    # 배열 : 동일한 데이터 타입, 데이터 연속
    # 파이썬은 배열이 아니라 리스트 왜냐하면 동일한 데이터 타입을 안쓰니까

    a= np.arange(6)
    print(a)
    print(a.shape, a.ndim, a.dtype) # 중요중요

    b = np.arange(6).reshape(2,3) # 중요중요 : 데이터의 갯수가 정확히 나누어 떨어져야 함 6/2 = 3
    print(b)
    print(b.shape, b.ndim, b.dtype) # 중요중요

    # 차원을 바꿔도 성능저하나 영향을 미치는 것 없음

    # 문제
    # 변수 c에 2페이지 3행 4열짜리 배열을 만들고 결과를 확인해 보세요.

    c = np.arange(24).reshape(2,3,4)
    print(c)
    print(c.shape, c.ndim, c.dtype) # 중요중요

    print(a.itemsize, b.itemsize, c.itemsize) # d type의 크기를 알려줌
    print(a.size, b.size, c.size) # 차원에 관계 없이 안의 데이터 갯수
    print('-' * 50)

    print(np.array([2, 3, 4]))
    print(np.array((2, 3, 4)))
    print(list(np.array((2, 3, 4))))

    # print(np.array(2,3,4)) 에러 왜냐하면 매개변수 1개 tuple을 줘야 하는데 여러개를 넣는 것이다.

    # 문제
    # np.array를 사용해서
    # 0~5까지의 정수를 2행 3열 배열에 넣어 보세요. (3가지)
    array = [0, 1, 2, 3, 4, 5]
    array = np.array(array).reshape(2,3)
    print(array)

    array = np.array([0, 1, 2, 3, 4, 5]).reshape(2,3) #강사님
    print(array)

    array = np.array([[0, 1, 2], [3, 4, 5]]) #강사님
    print(array)

    array = np.array((0, 1, 2, 3, 4, 5)).reshape(2,3)
    print(array)

    array = np.array([np.arange(3), np.arange(3,6)]) #강사님
    array = np.array(range(6)).reshape(2,3) #강사님

    array = np.arange(6).reshape(2,3) # 가장 편한것
    print(array)
    print(np.arange(6).reshape(2,3))
    print(np.arange(6).reshape(-1,3))
    print(np.arange(6).reshape(2,-1))  #비워 놓으면 파이썬이 알아서 계산해줌

    # 문제
    # 2차원 배열을 1차원 리스트로 변환해 보세요.
    a = np.arange(6).reshape(2, 3)
    print(list(a.reshape(a.size)))
    print(list(a.reshape(-1)))


    print('-' * 50)
    print(np.zeros((2, 5))) #0으로 초기화 하는 함수
    print(np.ones((2, 5)))
    print(np.empty((2,5))) #직접 값을 채워야 하는 경우, 안채우기 때문에 좀 더 빠르다.
    print(np.full((2,5),0.5))

    a = [[1, 2, 3], [4, 5, 6]]
    print(np.zeros_like(a))
    print(np.ones_like(a))
    print(np.empty_like(a)) #실제로 넣는게 아니라 믿지말 것
    print(np.full_like(a, 0.5))

    print(np.zeros_like(a, dtype=np.float))
    print(np.ones_like(a, dtype=np.float32))
    print(np.empty_like(a, dtype=np.float64))
    print(np.full_like(a, 0.5, dtype=np.float16))

    print(np.zeros_like(a, dtype=np.float).dtype)

    print(np.arange(0, 2, 0.25)) # 마지막 종료는 포함하지 않음, 원래 list는 정수만 된다. array만 되는 것
    print(np.linspace(0, 2, 9)) #linear space #구간을 9개로 나누겠다. 오른쪽 끝에 자기 자신을 포함한다.

a = np.arange(3)

print(a)
print(a + 1) # 덧셈 3번 일어남                    #broadcasting
print(a ** 2)
print(a > 1)
print(np.sin(a)) #배열의 모든 요소에 대해 계산해줌    #universal function


print()
b =  np.array([3, 4, 5])

print(a > b)
print(b > a) # 각각의 요소끼리 비교                  #vector operation
print(a + b)
print(a * b) # 그냥 곱셈

c = np.arange(6).reshape(-1, 3)
c += 1 # broadcasting
print(c)
print(c > 3)
print(np.logical_and(c > 0, c < 3))
print(c[np.logical_and(c > 0, c < 3)])
print(c[c>3]) #내가 원하는 데이터만 뽑아내기 위해 사용

# 문제
# c와 행렬 곱셈할 수 있는 배열을 만드세요.
# np.dot()
print(c)
d = np.arange(6).reshape(3,-1)
print(d)
print(np.dot(c,d))
print(np.dot(d,c))
print('-' * 50)

# 슬라이싱 (리스트, ndarray)
a = list(range(10))
print(a)
print(a[-1], a[-2])
print(a[3:7]) #range()와 사용 방법 동일, 종료는 포함하지 않음

# 문제
# 앞쪽 절반을 출력해보세요
print(a[:len(a)//2])
print(a[0:len(a)//2])

# 뒤쪽 절반을 출력해보세요
print(a[len(a)//2:])
print(a[len(a)//2:len(a)])

# 짝수 번째만 출력해보세요
for i in range(len(a)):
    if i % 2 == 0:
        print(a[i], end = ' ')
print()
print(a[::2])
print("ok")

# 홀수 번째만 출력해보세요
for i in range(len(a)):
    if i % 2 == 1:
        print(a[i], end = ' ')
print()
print(a[1::2])

# 거꾸로 출력해 보기
for i in reversed(range(len(a))):
    print(a[i], end = ' ')
print()
print(a[::-1])

print(a[3:3]) # 빈거임
print(a[len(a)-1:0:-1])
print(a[len(a)-1:-1:-1]) #왜 안될까, 마지막을 의미하는 것이 비우는 것 혹은 -1이다. 비우게 되면 된다.
print(a[-1:-1:-1])
print(a[-1::-1])
print(a[::-1])

print('-'*50)

# 차원이 다른 broadcasting

a = np.arange(3)
b = np.arange(6)
c = np.arange(3).reshape(-1, 3)
d = np.arange(6).reshape(-1, 3)
e = np.arange(3).reshape(3, -1)

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(e.shape)

# 검사
# 5개의 변수를 각각 2개씩 모두 더해 보세요
# 그래서 안되는 것은
# 왜 그런지 생각해 보세요
print('a', a)
print('b', b)
print('c', c)
print('d', d)
print('e', e)

# print("ab", a + b) #error 6개랑 3개랑 달라서
print("ac", a + c)
print("ad", a + d)
#print("bc", b + c) #error
#print("bd", b + d) #error
print("be", b + e)
print("cd", c + d)
print("ce", c + e)
#print("de", d + e) #error

