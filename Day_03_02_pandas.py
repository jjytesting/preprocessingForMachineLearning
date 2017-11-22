import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/scores.csv')
print(df)
#print(df[0]) error 인덱스 쓸 수 없음
print(df.iloc[0])
print(df.ix[0])
print(df.loc[0])
print(df.iloc[:3])
print('-'*50)

#subjects = ['kor','eng', 'mat', 'biong']
subjects = ['kor','eng', 'mat', 'bio']
print(df['kor'])
print(df[subjects])
print(df[subjects[::-1]])
print(df.kor)
print('-'*50)

print(df[subjects].sum())
# 학생별 합계를 내보자
print(df[subjects].sum(axis=0))
print(df[subjects].sum(axis=1))
print(df[subjects].sum().sum())

# 문제
# 과목별 평균, 학생별 평균을 출력해보세요.
print(df[subjects].mean(axis=0)) # 과목 평균
print(df[subjects].mean(axis=1)) # 학생 평균
print('-'*50)
df['sum'] = df[subjects].sum(axis=1)
#df.sum = df[subjects].sum(axis=1) # column 추가는 안된다... 동작 안함
# df['avg'] = df.sum / len(subjects) # error
df['avg'] = df['sum'] / len(subjects)
print(df)

print(df.sort_values('avg'))
print(df.sort_values('avg', ascending=False))
print('-' * 50)

sorted_df = df.sort_values('avg', ascending=False)
sorted_df.index = sorted_df.name
print(sorted_df)
print(sorted_df.index.values)

#del sorted_df.name 안됨
del sorted_df['name']
print(sorted_df)

def not_used():
    #sorted_df['avg'].plot(kind='bar')
    sorted_df['avg'].plot(kind='bar', figsize=(8,4))
    # x축 레이블 숨김
    ax = plt.axes()
    ax.xaxis.label.set_visible(False)
    #plt.ylim(0,100)
    plt.show()

print(df['class'] == 1)
c1 = df[df['class'] == 1]
c2 = df[df['class'] == 2]

print(c1)

mean_c1 = c1['sum'].mean()/4
print(c1['avg'].mean())
print(mean_c1)

# t-test
# 건너뛰기?

#sorted_df[subjects].plot(kind='bar')
#sorted_df['kor'].plot(kind='bar')
plt.axes().xaxis.label.set_visible(False)
#df[subjects].boxplot()
#plt.show()

#문제
# 1반과 2반 데이터를 boxplot으로 그려보세요.
plt.figure(1)
c1[subjects].boxplot()
plt.title('class1')

plt.figure(2)
c2[subjects].boxplot()
plt.title('class2')
plt.show()
# 박스 크기에서 1.5배 이상 벗어난 경우 이상치, o로 표현됨. 3배 이상 벗어나면 아주 이상한 값.

