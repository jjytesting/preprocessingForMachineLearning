import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 이거는 선생님 코드랑 비교해서 버그 찾아야 함

# https://grouplens.org/datasets/movielens/
# http://files.grouplens.org/datasets/movielens/ml-1m.zip

#ratings ==> UserID::MovieID::Rating::Timestamp
#users ==> UserID::Gender::Age::Occupation::Zip-code
#movies ==> MovieID::Title::Genres (여러개의 장르에 속해있을 수 있음)

# http://www.hanbit.co.kr/store/books/look.php?p_code=B6540908288
# 파이썬 라이브러리를 활용한 데이터 분석(수정보완판)

pd.set_option('display.width', 1000)

def get_movies():
    users = pd.read_csv('ml-1m/users.dat',
                        header=None, sep='::', engine='python',
                        names=['UserId','Gender','Age','Occupation','Zip-code'])

    movies = pd.read_csv('ml-1m/movies.dat',
                        header=None, sep='::', engine='python',
                        names=['MovieID','Title','Genres'])

    ratings = pd.read_csv('ml-1m/ratings.dat',
                        header=None, sep='::', engine='python',
                        names=['UserId','MovieID','Rating','Timestamp'])

    #print(movies)

    data = pd.merge(pd.merge(ratings, users), movies)

    return data

def basic_usage():
    data = get_movies()

    t1 = data.pivot_table(values='Rating', columns='Gender') #어떤 특정한 기준으로 축을 달리해서 볼 수 있는 table, 여자들이 남자들보다 후한 평가를 하고 있음.
    print(t1)


    t2 = data.pivot_table(values='Rating', columns='Gender', index='Age') #어떤 특정한 기준으로 축을 달리해서 볼 수 있는 table, 여자들이 남자들보다 후한 평가를 하고 있음.
    print(t2)

    t2.index = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"] #alt key 누르고 복사하면 세로로 선택 가능
    print(t2)

    #문제
    # 연령대별 성별 평점 결과를 막대 그래프로 그려 보세요.
    def draw_graph():
        t2.plot(kind='bar')
        plt.show()

    # t3 = data.pivot_table(values='Rating', columns=['Gender','Occupation']) # 에러
    t3 = data.pivot_table(values='Rating', index =['Gender','Occupation']) # multi index
    print(t3)

    t4 = data.pivot_table(values='Rating', index = 'Age', columns = ['Gender','Occupation'])
    # NaN은 없는 데이터 (결측치)
    print(t4)


    t5 = data.pivot_table(values='Rating', index = 'Age', columns = ['Gender','Occupation'], fill_value=0)
    # 결측치 표현방식 변경 가능, 0으로 하면 평가를 0으로 했다고 잘못 인식 가능하므로 신중하게 해라
    print(t5)

    print(t5.head(3))


    t7 = data.pivot_table(values='Rating', index =['Gender','Age']) # multi index
    print(t7)
    print(t7.unstack()) #index에 있던 녀석이 column으로 이동함.
    print(t7.unstack().unstack()) #index에 있던 녀석이 column으로 이동함.
    print(t7.unstack().stack()) #column에서 index로 이동

    t8= data.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc='mean')
    t8= data.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc='sum') # pivot 만들 때 연산 지정 가능
    print(t8)

    t9= data.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=[np.mean, np.sum]) # 함수 이름 전달이 더 좋음, 두 개도 할 수 있음
    print(t9)

    # 상황이 여의치 않아서 따로 계산한 경우
    t9_1= data.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=np.mean)
    print(t9_1)

    t9_2= data.pivot_table(values='Rating', index='Age', columns='Gender', aggfunc=np.sum)
    print(t9_2)

    print(pd.concat([t9_1,t9_2])) #column이 같은 거 붙일 때는 merge는 부적절하다. 이럴 때 concat 쓴다.
    print(pd.concat([t9_1,t9_2], axis=1)) #수평으로 붙이기

def groupby_usage(data):

    count_by_title = data.groupby('Title').size() # bounded : 짧은데 자동완성 안뜬다. title로 묶기. size는 몇개나 나왔는지 중복된 갯수 검사해줌. 제목 별 사람들이 평가한 갯수
    print(count_by_title.head())
    print(count_by_title.sum()) # 전체 평점에 대한 갯수 1000209개

    #temp = (count_by_title >= 500 )
    #print(temp)

    count_by_title_500 = count_by_title[count_by_title >= 500]
    print(count_by_title_500.head())

    index_over_500 = count_by_title.index[count_by_title_500]
    print(index_over_500)
    print(index_over_500.values)

    return index_over_500

data = get_movies()
# class member를 호출하는 2가지 방법이 있음
by_gender = pd.DataFrame.pivot_table(data, values='Rating', index='Title', columns='Gender') # unbound 방식 : 긴데 자동완성이 뜰 확률이 높다.
#print(by_gender.head())
index_over_500 = groupby_usage(data)

#rating_500 = by_gender[index_over_500] # data frame에 [] 는 column을 넣어야 한다.
#rating_500 = by_gender.ix[index_over_500] # data frame에 [] 는 column을 넣어야 한다.
rating_500 = by_gender.loc[index_over_500]
# 평점이 500개 이상인 영화에 대해서만 남자와 여자 평균 점수를 가리킴
#print(rating_500.head())

top_female = rating_500.sort_values(by='F', ascending=False)
print(top_female)
print(top_female.iloc[:5])
print(top_female.index[:5])

#여성들이 남성들보다 좋아하는 영화, 평점 간의 차이로 알 수 있겠다.
rating_500['Diff'] = rating_500['F'] - rating_500['M']
print(rating_500)

female_better = rating_500.sort_values(by='Diff')
print(female_better.head())

rating_500['Dist'] = (rating_500['F'] - rating_500['M']).abs()
far_off = rating_500.sort_values(by='Dist', ascending=False)
print(far_off.head())
print('-'*50)
#표준 편차
# rating_std = data.groupby('Title').std() #이렇게하면 모든 column에 대해 표준 편차를 구해버림
rating_std = data.groupby('Title')['Rating'].std() #Rating만 표준 편차를 구함

print(rating_std.head())

rating_std_500 = rating_std.loc[index_over_500]
print(rating_std_500.head())
print(type(rating_std_500)) #값이 하나라서 Series

# std_500_sorted = pd.Series.sort_values(rating_std_500) #unbound method
std_500_sorted = rating_std_500.sort_values() #bound method
print(std_500_sorted.head())

