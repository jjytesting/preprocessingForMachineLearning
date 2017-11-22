import requests
import re
import csv

# file열기
f = open('Data/kma.csv', 'w', encoding='utf-8', newline='') #빈줄 없앨 때 newline='' 이거 찾기 어려우므로 중요중요!!

# 기상청 서비스 RSS 전국
url = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108'
received = requests.get(url)
#print(received)
#print(received.text)

# 문제
# province와 city, data를 출력해 보세요

'''
def my():
    locations = re.findall(r'<location wl_ver="3">(.+)</location>', received.text, re.DOTALL)
    print(locations)
    
    provinces = re.findall(r'<province>(.+)</province>', locations)
    print(provinces)
    cities = re.findall(r'<city>([가-힣]+)</city>', received.text)
    print(cities)
    datas = re.findall(r'<data>(.+?)</data>', received.text)
    print(datas)
'''

# .+ : 탐욕적 (제일 큰거 하나만 찾아버림)
# .+? : 비 탐욕적 (제일 작은것 찾음)
locations = re.findall(r'<location wl_ver="3">(.+?)</location>', received.text, re.DOTALL)
#print(len(locations))
# print(locations)

kma = []

for loc in locations:
    province = re.findall(r'<province>(.+?)</province>', loc)
    city = re.findall(r'<city>([가-힣]+)</city>', loc)
    # area = re.findall(r'<province>(.+)</province>.+?<city>(.+)</city>', loc, re.DOTALL) # ?붙이면 성능이 더 좋음, 찾기 싫은 것 건너 뛸 때도 사용
    # print(area[0][0])
    # print(area[0][1]) # 튜플로 묶어서 return해 줌

    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)

    #print(province, city)
    #print(len(data))
    for datum in data: # 공백이 알맞게 들어가는지 유의할 것
        mode = re.findall(r'<mode>(.+)</mode>', datum)
        tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
        wf = re.findall(r'<wf>(.+)</wf>', datum)
        tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
        tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
        reliability = re.findall(r'<reliability>(.+)</reliability>', datum)

        # print('{},{},{},{},{},{},{},{}'.format(province[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], reliability[0])) #형식 문자열, 여러종류의 데이터를 문자열로 변환할 때 사용하는 문법
        # csv 파일로 저장하기 위해 , 를 미리 넣어둔 것임
        # findall은 무조건 list를 반환하므로 한개를 리턴하더라도 list. [0]해야 원소 값만 가져올 수 있다.
        # ['제주도'] ['제주'] 보기에는 편하지만 컴퓨터가 어떤 데이터인지 파악하기 어려우므로 한출로 바꿔주자.
        # 아무줄이나 읽어오면 모든 데이터가 들어 있어서 찾을 필요가 없다.
        # 제주도, 제주, A02,2017-11-24 00:00,구름많고 비,9,13,보통
        kma.append([province[0], city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], reliability[0]]) #[]안에 넣으면 각 원소를 list형태로 넣겠다는 뜻
        #print('{},{},{},{},{},{},{},{}'.format(area[0][0], area[0][1], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], reliability[0]),file=f)
        # csv 쓰던지 print해서 file 지정하던지 편한걸로, 근데 그러면 귀찮은 형식 문자열 해야함.

    # print(*kma, sep='\n') #*를 붙이면 여러개로 묶인 것을 풀어주는 역할
    csv.writer(f).writerows(kma)

f.close()