import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.options.display.width = 1000
filename = 'MLB World Series Champions_ 1903-2016.xlsx'
champs = pd.read_excel('world-series/'+filename, index_col=0)
print(champs.head())
# print(champs['Wins'])

total_teams = champs['Champion'].unique()
print(total_teams)
print(len(total_teams)) # 우승한 적이 있는 팀의 갯수

#문제
# 100승 이상 팀만 출력해 보세요.
'''
bigger_than_100wins = champs['Wins'] >= 100
print(len(bigger_than_100wins))
champs_100wins = champs[bigger_than_100wins]
print(champs_100wins)

print(len(champs_100wins['Champion'].unique()))
print(champs_100wins['Champion'].unique())
'''
over_100_index = champs.Wins >= 100
print(over_100_index.head())

over_100 = champs[over_100_index]
print(over_100)
print(over_100['Champion'].unique())

# 우승 팀 전체의 평균 승률은 얼마입니까?
print(champs.WinRatio.mean())

#문제 뉴욕 양키스의 평균 승률을 구해 보세요.
yankees_index = champs['Champion'] == 'New York Yankees'

print (yankees_index)
yankees = champs[yankees_index]
print(yankees)
print(yankees['WinRatio'].mean())

# 뉴욕 양키스가 우승한 최초 연도와 마지막 연도를 구해 보세요.
print(yankees.index)
print(yankees.index[0])
print(yankees.index[-1])
print(yankees.index.max())
print(yankees.index.min())

print(yankees.iloc[0])
print(yankees.iloc[-1])

print(yankees.loc[1923])

# 문제
# 가장 많이 우승한 5개 팀을 보여 주세요.
# 막대?

#names[names.Gender == 'M'].count()
'''unique_names = champs['Champion'].unique()

team_wins = {}
for team in unique_names:
    team_index = (champs['Champion'] == team)
    team_data = champs[team_index]
    team_wins[team] = len(team_data)

print(team_wins)
team_series= pd.Series(team_wins, index=team_wins.keys())
print(team_series.sort_values(ascending=False).head(5))
'''

by_teams = champs.groupby('Champion').size()
#print(by_teams)
sorted_teams = by_teams.sort_values(ascending=False)
print(sorted_teams.head(5)) # 이렇게 하면 공동 5등에 해당하는 2개 팀 덜보임

fifth = sorted_teams[4]
print(fifth)

top5_index = (sorted_teams >= fifth)
print(top5_index)
top5 = sorted_teams[top5_index]
print(top5)       # 횟수까지 표시
print(top5.index) # 이름만 표시