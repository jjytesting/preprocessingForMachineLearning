import pandas as pd
import matplotlib.pyplot as plt

#문제
#columns : Name, Gender, Births

#yob1880.txt 파일로 부터 아래 문제를 해결하세요.
# 1. 남자와 여자 이름 갯수
# 2. 남자와 여자 이름 갯수 합계
# 3. 남자 또는 여자 top5 막대 그래프

#names = pd.read_csv('Data/yob1880.txt',header=None)
#names.columns = ['Name', 'Gender', 'Birth']
#print(names)

'''
gender_by = names.groupby(['Gender', 'Name'])
print(gender_by)
print(gender_by['Name'].unique())
print(gender_by.sum().sort_values(by='Birth', ascending=False))


names_num = gender_by.pivot_table(values='Birth', columns='Gender', aggfunc='sum')
print(names_num)
'''

names = pd.read_csv('Data/yob1880.txt', header=None)

names.columns = ['Name', 'Gender', 'Birth']

men = (names.Gender == 'M')
women = (names.Gender == 'F')

print(men.count())
print(men.sum(), women.sum())

#print(len(names[names.Gender == 'M']))
print(names[names.Gender == 'M'].count())
print(len(names[names.Gender == 'M']))
print('-' * 50)

# 1번
by_gender = names.groupby('Gender').size()
print(by_gender)

# 2번
print(names.groupby('Gender').sum())

#3번
men_only = names[names.Gender == 'M']
print(men_only[:5])

top5 = men_only[:5]
print(top5)

top5.index = top5.Name
del top5['Name']
print(top5)

#top5.plot(kind='bar')
#plt.show()

# 문제
# 남자와 여자 이름이 같은 데이터를 찾아 보세요.

women_only = names[names.Gender == 'F']
#men_only.index = men_only.Name
#women_only.index = women_only.Name

both = []
for name in women_only.Name:
    if name in men_only.Name:
        both.append(name);
#for name in women_only.Name:

print(both)
