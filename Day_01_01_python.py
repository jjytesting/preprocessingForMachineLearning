


print(12, 3.14, True, 'hello')
print(type(12), type(3.14), type(True), type('hello'))

#다중 치환
a, b = 7, 3
print(a, b)

#산술 연산
print(a + b)
print(a - b)
print(a * b)
print(a / b)  # 실수 나눗셈
print(a ** b) # 지수
print(a // b) # 정수 나눗셈
print(a % b)

print('a' + 'b')
print('a' * 3)
print('-' * 50)

# 관계연산자
print(a < b)
print(a <= b)
print(a > b)
print(a >= b)
print(a == b)
print(a != b)

# 형변환
print(int(a < b))
print(int(a > b))

age = 15
print(10 <= age <= 19) #범위 연산. 파이썬에서만 동작함
print('-' * 50)

# 논리 연산
print('\n\n\n\n\n')
print(True and True)
print(True and False)
print(False and True)
print(False and False)

a = 3
if a % 2 == 1:
    print('홀수') # 반드시 indentation
else:
    print('짝수')

if a < 0:
    print('음수')
elif a > 0:
    print('양수')
else:
    print('제로')

# shift + enter 중간에서 밑 라인으로 내려감
# ctrl + c 한줄 자동 복사함

for i in range(5): # 종료
    print(i, end = ' ')
print()

for i in range(0, 5): # 시작, 종료
    print(i, end = ' ')
print()

for i in range(0, 5, 1): # 시작, 종료, 증감
    print(i, end = ' ')
print()


for i in range(4, -1, -1): # 시작, 종료, 증감. 그런데 이런것 파이썬 스럽지 않음!!
    print(i, end = ' ')
print()


for i in reversed(range(5)): # 거꾸로.. 아무거나 다 뒤집음
    print(i, end = ' ')
print()

print('-' * 50)

# collection : list, tuple, set, dictionary
#               []     ()    {}    {}

a = [1, 3, 5] #list
print(a)
print(a[0], a[1], a[2])

a.append(7)
#a += [9]
#a.extend([9])
#a.append([7]) [7]이라는 리스트가 들어감
# a += 9 에러남

for i in range(len(a)):
    print(i, a[i])

for i in a:  #iterable. for문 오른쪽에 쓸 수 있는 것. 여러개 들어있다는 의미
    print(i)

# a를 거꾸로 뒤집어 보세요
b =[]
for i in reversed(a):
    b.append(i)

a = b

for i in range(len(a)):
    print(i, a[i])


for i in range(len(a)//2):
    #print(i, len(a)-1-i)
    a[i], a[len(a)-1-i] = a[len(a)-1-i], a[i]


for i in range(len(a)):
    print(i, a[i])

print('-' * 50)

#tuple
a = (1, 3, 5)
print(a)
print(a[0], a[1], a[2])

for i in a:
    print(i)

# 튜플 : 상수 버전의 리스트, 변경할 수 있는 리스트
# a.append(7) # 에러
# a[0] = 99 # 에러

# 튜플은 파이썬이 내부적으로 많이 사용함
print('-' * 50)

def dummy_1():
    pass

a  =dummy_1()
print(a)

def dummy_2(a, b):
    if a < b:
        return a, b
    return b, a

m1, m2 = dummy_2(3, 7)
print(m1, m2)

m = dummy_2(3, 7) # packing 여러개를 하나로 묶기
print(m, m[0], m[1]) # tuple이 됨

def dummy_3():
    return [1, 3, 5]

a = dummy_3()
print(a)

a1, a2, a3 = dummy_3() # unpacking 하나를 여러개로 풀기 (파이썬이 알아서 해줌)
print(a1, a2, a3)
print('-' * 50)

print(1, 2, 3, sep='**', end='\n\n') #sep도 keyword parameter에 해당됨
print(1, 2, 3)

def f_1(a, b, c):
    print(a, b, c)

f_1(1, 2, 3)        # positional parameter # 성능은 이게 더 좋음
f_1(a=1, b=2, c=3)  # keyword parameter # 개발할 때는 이게 더 좋음
f_1(1, b=2, c=3)    # 섞어쓸 수 있음
f_1(1, c=2, b=3)    # keyword 순서는 중요하지 않음

def f_2(*args):     # 매개변수에 * 붙이는 것 : 가변인자 - 매개변수의 수가 정해져 있지 않음 packing
    print(args, *args) # unpacking (force 강제로)

f_2()
f_2(1)
f_2(1, 2)

# 딕셔너리
# 영한 사전 : 영어 단어를 찾으면 한글 설명 나옴
# 영어 단어 : key
# 한글 설명 : value

a = {'name': 'kim', 'age': 20, 30: 50}
a = dict(name='kim', age= 20) # 30=50은 사용 불가, 이렇게 선언하는거 더 많이 씀, 앞의 것 변수여야 함
print(a)
#print(a['name'], a['age'], a[30])
print(a['name'], a['age'])

print(a.keys())
print(a.values())
print(a.items())

for k in a.keys():
    print(k, end= ' ')
print()

for v in a.values():
    print(v, end= ' ')
print()

# 문제
# items 함수를 for문에 적용해 보세요.

for v in a.items():
    key, value = v
    print(key, '=', value, end= ',')

for k, v in a.items():
    print(k, v, end= ' ')
print()

print('-' * 50)
for k in a:
    print(k, a[k])

s = 'hello'
for c in s:
    print(c, end=' ')
print()

#range, reversed, enumerate 3총사
for i, c in enumerate(s): # 반복 횟수를 알려줌, 문제는 튜플로 리턴함
    print(i, c)

# 문제
# 딕셔너리의 items 함수와 enumerate를 연결해서 사용해 보세요.
a = dict(name='kim', age= 20)

for v in enumerate(a.items()):
    print(v)
print()

for i, (k, v) in enumerate(a.items()):
    print(i, k, v)
print()
