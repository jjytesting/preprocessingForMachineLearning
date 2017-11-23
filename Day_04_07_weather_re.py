import requests
import re

url ='http://www.kma.go.kr/weather/climate/past_cal.jsp?stn=108&yy=2017&mm=9&x=19&y=15&obs=1'
received = requests.get(url)
#print(received.text)

#codes = re.findall(r'[0-9]+',text)
'''

degree = re.findall(r'([0-9]+.[0-9]+[℃]+)', received.text)
avg = re.findall(r'평균기온:([0-9]+.[0-9]+[℃]+)', received.text)
max = re.findall(r'최고기온:([0-9]+.[0-9]+[℃]+)', received.text)
min = re.findall(r'최저기온:([0-9]+.[0-9]+[℃]+)', received.text)

print(degree)

print(avg)
print(max)
print(min)

degrees = re.findall(r'평균기온:([0-9]+.[0-9]+[℃]+).*최고기온:([0-9]+.[0-9]+[℃]+).*최저기온:([0-9]+.[0-9]+[℃]+)', received.text)
print(degrees)

'''

tbody = re.findall(r'<tbody>(.+?)</tbody>', received.text, re.DOTALL) #한개 밖에 없음
#print(tbody)
#print(tbody[0])
# 날짜 행과 데이터 행이 2개씩 배치됨
trs = re.findall(r'<tr>(.+?)</tr>', tbody[0], re.DOTALL)

trs = [tr for tr in trs[1::2]]
days = []
for tr in trs:
    tds = re.findall(r'<td class="align_left">(.+?)</td>', tr)

    tds = [td for td in tds if td != '&nbsp;']
    #print(tds)

    days += tds

for day in days:
    print(day)

    items = re.findall(r'[0-9]+.[0-9]', day)
    print(items[:3])
'''
items = re.findall(r'ass="align_left">(.+?)</td>',tbody[0])
for item in items:
    print(item)
'''