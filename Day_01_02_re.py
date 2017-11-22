import re
import requests

def not_used_1():
    # 아이디 전화번호
    db = '''3412    [Bob] 123
    3834  Jonny 333
    1248   Kate 634
    1423   Tony 567
    2567  Peter 435
    3567  Alice 535
    1548  Kerry 534'''

    # print(db)
    # r: raw 가공하지 않았다
    temp = re.findall(r'', db) # r 기존 문자열과 비교하기 위한 용도로 붙임
    temp = re.findall(r'[0-9]', db)
    print(temp)

    numbers = re.findall(r'[0-9]+', db)
    print(numbers)

    # 문제
    # 아이디만 찾아보세요
    # ids = re.findall(r'^[0-9]{3}',db) 이런거 안됨
    ids = re.findall(r'^[0-9]+',db, re.MULTILINE)
    print(ids)
    # 전화번호만 찾아보세요
    phoneNumbers = re.findall(r'[0-9]+$',db, re.MULTILINE)
    print(phoneNumbers)

    # 문제
    # 이름만 찾아보세요
    names = re.findall(r'[A-Z][a-z]+',db)
    print(names)

    # T로 시작하는 이름만 찾아보세요.
    names = re.findall(r'T[a-z]+',db)
    print(names)

    # T로 시작하지 않는 이름만 찾아보세요.
    #names = re.findall(r'[^T][a-z]+',db) #ony가 들어가게 됨
    names = re.findall(r'[A-SU-Z][a-z]+',db)
    print(names)

def not_used_2():
    url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
    received = requests.get(url)
    print(received)
    print(received.text)

    text = received.content.decode('utf-8')
    print(text)
    print(type(text))

    #문제
    #수신 데이터로부터
    #코드와 도시 이름만 찾아 보세요

    codes = re.findall(r'[0-9]+',text)
    print(codes)

    #야매루
    #cities = re.findall(r'[^0-9",a-z:[{}\]]+',text)
    #print(cities)

    cities = re.findall(r'[가-힣]+',text)
    print(cities)

    items = zip(codes, cities) #묶어버리기
    print(list(items))

    # [{"code":"11","value":"서울특별시"},{"code":"26","value":"부산광역시"},{"code":"27","value":"대구광역시"},{"code":"28","value":"인천광역시"},{"code":"29","value":"광주광역시"},{"code":"30","value":"대전광역시"},{"code":"31","value":"울산광역시"},{"code":"41","value":"경기도"},{"code":"42","value":"강원도"},{"code":"43","value":"충청북도"},{"code":"44","value":"충청남도"},{"code":"45","value":"전라북도"},{"code":"46","value":"전라남도"},{"code":"47","value":"경상북도"},{"code":"48","value":"경상남도"},{"code":"50","value":"제주특별자치도"}]

    print(re.findall(r'"[0-9]+"',text)) #큰 따옴표로 감싼 숫자들
    print(re.findall(r'"([0-9]+)"',text)) #괄호 안에것만 뽑아줌

    print(re.findall(r'"([가-힣]+)"',text))
    items = re.findall(r'"value":"([가-힣]+)"',text)
    items = re.findall(r'{"code":"([0-9]+)","value":"(.+?)"',text)
    print(items)
    print(len(items))

#용인 수지도서관 빈자리 몇개?
url = 'http://211.251.214.169:8080/SeatMate_sj/SeatMate.php?classInfo=1'
received = requests.get(url)
print(received)
#print(received.text)

text = received.content.decode('euc-kr')
#print(text)

# 문제
# 빈 자리가 몇 개인지 알려 주세요.
# 번호까지 알려주면 더 좋습니다.
#margin-left:0px; margin-right:0px; margin-bottom:0px; margin-top:0px; padding-left:0px; padding-right:0px;  padding-bottom:0px; padding-top:0px; ">423</div></TD>

seats=re.findall(r'">([0-9]+)</div></TD>',text)
print(len(seats))
print(seats)

