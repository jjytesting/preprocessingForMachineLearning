import pandas as pd
# xlrt, xlwt (엑셀을 읽고 쓰는 module)

import matplotlib.pyplot as plt
import numpy as np

def use_Series():
    s = pd.Series([5, 1, 2, 9])
    print(s)
    # 만든 것과 나타나는 모습이 좀 달라서 어렵게 느껴짐
    # 앞에 인덱스가 하나 더 붙음
    # Series = pandas의 1차원 data, 많이 안씀, dictionary like
    # pandas = data frame

    print(s.values)
    print(s.index)
    print('-' * 50)

    s2= pd.Series([5, 1, 2, 9], index=['a', 'b', 'c', 'd'])
    print(s2)

    print(s2.values)
    print(s2.index)

    print(s2[0], s2[1], s2[-2], s2[-1]) #마지막에서 두 번째, 마지막에서 첫 번째, 이렇게 잘 안씀
    print(s2['a'], s2['b']) #이렇게 더 잘 써서 dictionary 닮았다고 함. 그러나 dictionary는 아니고 사용법이 같을 뿐이다.
    print('-' * 50)

    s3 = pd.Series({'a': 5, 'b': 1, 'c': 2, 'd':9}) #dictionary를 이용해서 series생성 가능
    print(s3)

def comprehension(): #코드가 짧아지고 가독성이 좋아지므로 안쓰기 어려운 문법
    #for i in range(5):
    #    i
    #리스트 만들 때 [i for i in range(5)]
    # 튜플 만들 때 (i for i in range(5)]
    # 딕셔너리 만들 때 {i for i in range(5)}
    print([i for i in range(5)])
    print([0 for i in range(5)])
    print([[i] for i in range(5)])
    a = list(range(10))
    print([i for i in a]) #a를 그대로 복사
    print([i for i in a if i % 2 == 1]) #if에 대해서 참인 경우만 들어감
    print([i*j for i in a if i % 2 == 1 for j in range(3)])


def use_names():
    baby_names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']

    np.random.seed(1)
    names = [baby_names[np.random.randint(5)] for _ in range(1000)] #comprehension을 쓸거임
    births = [np.random.randint(1000) for _ in range(1000)]

    #baby_set = [baby_names[np.random.randint(5)], np.random.randint(1000)) for _ in range(1000)]

    baby_set = list(zip(names, births)) #zip은 data를 하나로 묶어주는 함수
    print(baby_set)

    df = pd.DataFrame(baby_set, columns=['Name', 'Births']) #왼쪽 index, 위쪽 column
    print(df) #전체 60개의 data만 보여줌
    print(df.index)
    print(df.values)
    print(df.columns)
    print('-'*50)

    df.info()
    print(df.head()) #기본 5개 출력
    print(df.head(3))

    print(df.tail()) #뒷 부분 5개 출력
    print(df['Name'].unique()) # 수직 데이터 unique는 중복되지 않은 것만 꺼내 주는 역할

    name_by = df.groupby('Name')
    print(name_by)
    print(name_by.sum())

    #name_by.sum().plot(kind='bar') #pandas는 좀 더 예쁘게 그려준다. 우리가 필요한 것을 알아서 넣어준다.
    #plt.show()

    print(name_by.size())
    name_by.size().plot()
    plt.ylim(0,300)
    plt.show()

df = pd.DataFrame({'state': ['ohio','ohio','ohio','nevada','nevada','nevada'],
                   'year':[2000, 2001, 2002, 2000, 2001, 2002],
                   'population': [1.5, 1.7, 3.6, 2.4, 2.9, 2.8]})

print(df)

print('-' * 50)

print(df.index)
df.index = ['one', 'two', 'three', 'four', 'five', 'six']
print(df.index)
print(df)
print('-' * 50)

print(df['population'])
print(type(df['population']))
print(df['population'][2])
print(df['population']['three'])

print(df.population) # dot expression
print(df.population[2])
print('-'*50)

print(df.iloc[2]) #행 데이터에 접근하는 방법
print(df.ix[2]) #위에꺼랑 같은 것

print('-'* 50)
print(df.loc['three']) # 명확함 왜냐하면 index 인지 아닌지 알 수 있음.
print(df.ix['three']) #ix는 정수 인덱스나 실제 인덱스나 아무거나 쓸 수 있다.

print('-'* 50)
print(df)
print(df.iloc[1:3])
print(df.loc['two':'three']) #print(df.loc['two':'four']) loc는 종료하는 것도 나옴..