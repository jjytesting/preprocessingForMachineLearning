import json
import requests
# http://www.jsontest.com/

def func_1():
    j1 = '{"ip": "8.8.8.8"}' #dictionary를 가지고 있는 문자열

    print(j1)
    print(type(j1))

    j2 = json.loads(j1)
    # dump file 객체를 문자로 변환
    # dumps string
    # load file 문자를 객체로 변환
    # loads string
    print(j2)
    print(type(j2))

    j3 = json.dumps(j2)
    print(j3)
    print(type(j3))

#--------------------------------------------#
def func_2():
    dt = '''{
       "time": "05:11:25 AM",
       "milliseconds_since_epoch": 1511241085391,
       "date": "11-21-2017"
    }'''

    dt_dict = json.loads(dt) #load string

    for v in dt_dict.values():
        print(v, end=' ')
    print()

    for k in dt_dict:
        print(k, dt_dict[k])

    print(list(dt_dict.values()))


#--------------------------------------------#
# 문제
# 코드 번호와 도시 이름을 json으로 파싱해서 출력해 보세요.
def func_3():
    url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
    received = requests.get(url)


    text = received.content.decode('utf-8')
    #print(text)
    #print(type(text))

    data = json.loads(text)
    for datum in data:
         #wt = json.dumps(datum)
         #city = json.loads(wt)
         #print(list(city.values()))
         print(datum['code'],datum['value'])

    # [로 시작하는 것은 배열
    # {로 시작하는 것은 dictionary

#-------------------------------------------#
# 양재역 지하철 정보
url = 'http://place.map.daum.net/main/v/SES3402?_=1511243424471'
received = requests.get(url)


text = received.content.decode('utf-8')
print(text)
# {"isLogin":false,"isExist":true,"basicInfo":{"cid":14659150,"placenamefull":"양재역 신분당선","phonenum":"031-8018-7720","reltelList":[{"phonenum":"031-8018-7777","phonetype":"lost"}],"address":{"newaddr":{"newaddrfull":"남부순환로 2585","bsizonno":"06735"},"region":{"name3":"양재동","fullname":"서울 서초구 양재동","newaddrfullname":"서울 서초구"}},"wpointx":507773,"wpointy":1106674,"roadview":{"panoid":1077649884,"tilt":4,"pan":42.1376,"wphotox":507754,"wphotoy":1106653,"rvlevel":0},"cateid":"3304","catename":"수도권지하철신분당선","feedback":{"allphotocnt":46,"blogrvwcnt":10,"comntcnt":0,"scoresum":0,"scorecnt":0},"facilityInfo":{"doorCode":1,"toiletCode":3,"desc":"민원실, 엘리베이터, 기차표예매소","door":"왼쪽","crossingYN":"Y","platform":"양쪽","toilet":"개찰구 안/밖"},"stationid":"SES3402","subwayId":"SES34","subwayName":"신분당선","stationMainNameEn":"Yangjae(Seocho-gu Office)","preStations":[{"id":"SES3401","stationMainName":"강남"}],"nextStations":[{"id":"SES3403","stationMainName":"양재시민의숲"}],"subwayTransferList":[{"id":"SES0332","subwayId":"SES3","subwayName":"서울지하철 3호선","subwaySimpleName":"3호선","simpleName":"양재역"}],"stationMainName":"양재","stationSubName":"(서초구청)","subwayRegionid":"01","subwayDailyType":"WEEKDAY","isStation":true},"timetable":{"dayType":1,"timetableList":[{"hour":5,"up":[{"highlight":false,"directionInfo":"정자 > 강남","time":"05:45"},{"highlight":false,"directionInfo":"정자 > 강남","time":"05:55"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"05:32"},{"highlight":false,"directionInfo":"강남 > 광교","time":"05:45"},{"highlight":false,"directionInfo":"강남 > 광교","time":"05:56"}]},{"hour":6,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"06:05"},{"highlight":false,"directionInfo":"광교 > 강남","time":"06:17"},{"highlight":false,"directionInfo":"광교 > 강남","time":"06:28"},{"highlight":false,"directionInfo":"광교 > 강남","time":"06:39"},{"highlight":false,"directionInfo":"광교 > 강남","time":"06:50"},{"highlight":false,"directionInfo":"정자 > 강남","time":"06:55"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"06:07"},{"highlight":false,"directionInfo":"강남 > 광교","time":"06:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"06:29"},{"highlight":false,"directionInfo":"강남 > 광교","time":"06:40"},{"highlight":false,"directionInfo":"강남 > 광교","time":"06:50"},{"highlight":false,"directionInfo":"강남 > 광교","time":"06:58"}]},{"hour":7,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"07:01"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:14"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:20"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:26"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:32"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:38"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:49"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:54"},{"highlight":false,"directionInfo":"광교 > 강남","time":"07:58"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"07:04"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:10"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:16"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:27"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:33"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:39"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:45"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:51"},{"highlight":false,"directionInfo":"강남 > 광교","time":"07:57"}]},{"hour":8,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"08:03"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:07"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:12"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:16"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:21"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:25"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:30"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:35"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:39"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:49"},{"highlight":false,"directionInfo":"광교 > 강남","time":"08:54"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"08:01"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:06"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:10"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:15"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:19"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:24"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:28"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:33"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:37"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:47"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:52"},{"highlight":false,"directionInfo":"강남 > 광교","time":"08:58"}]},{"hour":9,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"09:00"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:06"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:12"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:19"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:27"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:36"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"09:52"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"09:05"},{"highlight":false,"directionInfo":"강남 > 광교","time":"09:11"},{"highlight":false,"directionInfo":"강남 > 광교","time":"09:18"},{"highlight":false,"directionInfo":"강남 > 정자","time":"09:23"},{"highlight":false,"directionInfo":"강남 > 광교","time":"09:29"},{"highlight":false,"directionInfo":"강남 > 광교","time":"09:38"},{"highlight":false,"directionInfo":"강남 > 광교","time":"09:46"},{"highlight":false,"directionInfo":"강남 > 광교","time":"09:54"}]},{"hour":10,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"10:00"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:16"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:24"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:32"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:40"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:48"},{"highlight":false,"directionInfo":"광교 > 강남","time":"10:56"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"10:02"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:10"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:26"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:34"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:50"},{"highlight":false,"directionInfo":"강남 > 광교","time":"10:58"}]},{"hour":11,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"11:04"},{"highlight":false,"directionInfo":"광교 > 강남","time":"11:12"},{"highlight":false,"directionInfo":"광교 > 강남","time":"11:20"},{"highlight":false,"directionInfo":"광교 > 강남","time":"11:28"},{"highlight":false,"directionInfo":"광교 > 강남","time":"11:36"},{"highlight":false,"directionInfo":"광교 > 강남","time":"11:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"11:52"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"11:06"},{"highlight":false,"directionInfo":"강남 > 광교","time":"11:14"},{"highlight":false,"directionInfo":"강남 > 광교","time":"11:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"11:30"},{"highlight":false,"directionInfo":"강남 > 광교","time":"11:38"},{"highlight":false,"directionInfo":"강남 > 광교","time":"11:46"},{"highlight":false,"directionInfo":"강남 > 광교","time":"11:54"}]},{"hour":12,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"12:00"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:16"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:24"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:32"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:40"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:48"},{"highlight":false,"directionInfo":"광교 > 강남","time":"12:56"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"12:02"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:10"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:26"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:34"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:50"},{"highlight":false,"directionInfo":"강남 > 광교","time":"12:58"}]},{"hour":13,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"13:04"},{"highlight":false,"directionInfo":"광교 > 강남","time":"13:12"},{"highlight":false,"directionInfo":"광교 > 강남","time":"13:20"},{"highlight":false,"directionInfo":"광교 > 강남","time":"13:28"},{"highlight":false,"directionInfo":"광교 > 강남","time":"13:36"},{"highlight":false,"directionInfo":"광교 > 강남","time":"13:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"13:52"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"13:06"},{"highlight":false,"directionInfo":"강남 > 광교","time":"13:14"},{"highlight":false,"directionInfo":"강남 > 광교","time":"13:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"13:30"},{"highlight":false,"directionInfo":"강남 > 광교","time":"13:38"},{"highlight":false,"directionInfo":"강남 > 광교","time":"13:46"},{"highlight":false,"directionInfo":"강남 > 광교","time":"13:54"}]},{"hour":14,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"14:00"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:16"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:24"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:32"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:40"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:48"},{"highlight":false,"directionInfo":"광교 > 강남","time":"14:56"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"14:02"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:10"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:26"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:34"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:50"},{"highlight":false,"directionInfo":"강남 > 광교","time":"14:58"}]},{"hour":15,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"15:04"},{"highlight":false,"directionInfo":"광교 > 강남","time":"15:12"},{"highlight":false,"directionInfo":"광교 > 강남","time":"15:20"},{"highlight":false,"directionInfo":"광교 > 강남","time":"15:28"},{"highlight":false,"directionInfo":"광교 > 강남","time":"15:36"},{"highlight":false,"directionInfo":"광교 > 강남","time":"15:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"15:52"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"15:06"},{"highlight":false,"directionInfo":"강남 > 광교","time":"15:14"},{"highlight":false,"directionInfo":"강남 > 광교","time":"15:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"15:30"},{"highlight":false,"directionInfo":"강남 > 광교","time":"15:38"},{"highlight":false,"directionInfo":"강남 > 광교","time":"15:46"},{"highlight":false,"directionInfo":"강남 > 광교","time":"15:54"}]},{"hour":16,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"16:00"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:16"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:24"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:32"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:40"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:48"},{"highlight":false,"directionInfo":"광교 > 강남","time":"16:56"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"16:02"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:10"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:26"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:34"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:50"},{"highlight":false,"directionInfo":"강남 > 광교","time":"16:58"}]},{"hour":17,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"17:04"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:12"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:20"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:28"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:36"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:44"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:51"},{"highlight":false,"directionInfo":"광교 > 강남","time":"17:58"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"17:06"},{"highlight":false,"directionInfo":"강남 > 광교","time":"17:14"},{"highlight":false,"directionInfo":"강남 > 광교","time":"17:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"17:30"},{"highlight":false,"directionInfo":"강남 > 광교","time":"17:37"},{"highlight":false,"directionInfo":"강남 > 광교","time":"17:45"},{"highlight":false,"directionInfo":"강남 > 광교","time":"17:53"}]},{"hour":18,"up":[{"highlight":false,"directionInfo":"정자 > 강남","time":"18:01"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:06"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:11"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:17"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:23"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:29"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:34"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:38"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:43"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:48"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:53"},{"highlight":false,"directionInfo":"광교 > 강남","time":"18:58"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"18:01"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:07"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:12"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:17"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:27"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:32"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:37"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:47"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:52"},{"highlight":false,"directionInfo":"강남 > 광교","time":"18:57"}]},{"hour":19,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"19:03"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:13"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:18"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:23"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:28"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:33"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:38"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:43"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:48"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:53"},{"highlight":false,"directionInfo":"광교 > 강남","time":"19:58"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"19:02"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:07"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:12"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:17"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:22"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:27"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:32"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:37"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:42"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:47"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:52"},{"highlight":false,"directionInfo":"강남 > 광교","time":"19:57"}]},{"hour":20,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"20:03"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:08"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:13"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:20"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:27"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:35"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:43"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:51"},{"highlight":false,"directionInfo":"광교 > 강남","time":"20:58"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"20:02"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:07"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:12"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:24"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:32"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:40"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:48"},{"highlight":false,"directionInfo":"강남 > 광교","time":"20:55"}]},{"hour":21,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"21:05"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:11"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:17"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:24"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:30"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:37"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:43"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:50"},{"highlight":false,"directionInfo":"광교 > 강남","time":"21:56"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"21:01"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:08"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:14"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:21"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:27"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:34"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:40"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:47"},{"highlight":false,"directionInfo":"강남 > 광교","time":"21:53"}]},{"hour":22,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"22:03"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:09"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:16"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:22"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:29"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:37"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:45"},{"highlight":false,"directionInfo":"광교 > 강남","time":"22:53"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"22:00"},{"highlight":false,"directionInfo":"강남 > 광교","time":"22:06"},{"highlight":false,"directionInfo":"강남 > 정자","time":"22:13"},{"highlight":false,"directionInfo":"강남 > 광교","time":"22:18"},{"highlight":false,"directionInfo":"강남 > 광교","time":"22:27"},{"highlight":false,"directionInfo":"강남 > 정자","time":"22:32"},{"highlight":false,"directionInfo":"강남 > 광교","time":"22:39"},{"highlight":false,"directionInfo":"강남 > 광교","time":"22:47"},{"highlight":false,"directionInfo":"강남 > 광교","time":"22:55"}]},{"hour":23,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"23:01"},{"highlight":false,"directionInfo":"광교 > 강남","time":"23:09"},{"highlight":false,"directionInfo":"광교 > 강남","time":"23:17"},{"highlight":false,"directionInfo":"광교 > 강남","time":"23:25"},{"highlight":false,"directionInfo":"광교 > 강남","time":"23:35"},{"highlight":false,"directionInfo":"광교 > 강남","time":"23:45"},{"highlight":false,"directionInfo":"광교 > 강남","time":"23:55"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"23:03"},{"highlight":false,"directionInfo":"강남 > 광교","time":"23:11"},{"highlight":false,"directionInfo":"강남 > 광교","time":"23:19"},{"highlight":false,"directionInfo":"강남 > 광교","time":"23:27"},{"highlight":false,"directionInfo":"강남 > 광교","time":"23:35"},{"highlight":false,"directionInfo":"강남 > 광교","time":"23:45"},{"highlight":false,"directionInfo":"강남 > 광교","time":"23:55"}]},{"hour":24,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"24:05"},{"highlight":false,"directionInfo":"광교 > 강남","time":"24:17"},{"highlight":false,"directionInfo":"광교 > 강남","time":"24:29"},{"highlight":false,"directionInfo":"광교 > 강남","time":"24:41"},{"highlight":false,"directionInfo":"광교 > 강남","time":"24:53"}],"down":[{"highlight":false,"directionInfo":"강남 > 광교","time":"24:05"},{"highlight":false,"directionInfo":"강남 > 광교","time":"24:17"},{"highlight":false,"directionInfo":"강남 > 광교","time":"24:29"},{"highlight":false,"directionInfo":"강남 > 정자","time":"24:41"},{"highlight":false,"directionInfo":"강남 > 정자","time":"24:53"}]},{"hour":25,"up":[{"highlight":false,"directionInfo":"광교 > 강남","time":"25:05"}]}],"fastCheck":false,"upDirectionName":"강남 방향","downDirectionName":"양재시민의숲 방향"},"exitAroundInfo":{"x":507773,"y":1106674,"exitList":[{"exitNum":"1","roadview":{"panoid":1077972143,"tilt":2,"pan":-57.0,"wphotox":507315,"wphotoy":1106957,"rvlevel":3},"busstop":[{"busStopId":"11220521007","busStopName":"서초구청맞은편","busStopDisplayId":"22129","x":506766,"y":1106895,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"406, 405"}]},{"busStopId":"BS106668","busStopName":"서초구청맞은편","busStopDisplayId":"90067","x":506773,"y":1106895,"busInfo":[{"busType":"일반","busTypeCode":"GENERAL","busList":"500-5, 19"},{"busType":"직행","busTypeCode":"DIRECT","busList":"1241"}]},{"busStopId":"BS114722","busStopName":"서초구청맞은편","busStopDisplayId":"22980","x":506779,"y":1106896,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초21, 서초17"}]}],"arroundInfo":"서초우성아파트, 영동중학교, 한전 강남지점, 수협은행, 서초동, 남부순환도로"},{"exitNum":"2","roadview":{"panoid":1077649745,"tilt":-1,"pan":-101.0,"wphotox":507444,"wphotoy":1107098,"rvlevel":1},"arroundInfo":"영동중학교, SC은행, 국민은행, 강남대로, KDB산업은행 남서초지점"},{"exitNum":"3","roadview":{"panoid":1077643596,"tilt":-1,"pan":-29.0,"wphotox":507542,"wphotoy":1107065,"rvlevel":0},"arroundInfo":"은광여자고등학교, 은성중학교, 농협, 서울언주초등학교, 강남대로, 한국야구위원회"},{"exitNum":"4","roadview":{"panoid":1078032304,"tilt":-2,"pan":39.0,"wphotox":507629,"wphotoy":1107023,"rvlevel":0},"busstop":[{"busStopId":"11230661017","busStopName":"양재역말죽거리.강남베드로병원","busStopDisplayId":"23318","x":508395,"y":1107180,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"402, 406, N37"},{"busType":"지선","busTypeCode":"GREEN","busList":"3012, 4319, 4433, 4435, 4412"}]},{"busStopId":"BS110845","busStopName":"양재역말죽거리.강남베드로병원","busStopDisplayId":"91034","x":508402,"y":1107182,"busInfo":[{"busType":"일반","busTypeCode":"GENERAL","busList":"917, 11-3, 6"},{"busType":"직행","busTypeCode":"DIRECT","busList":"3300, 3200, 3400"}]},{"busStopId":"BS115967","busStopName":"양재역4번출구","busStopDisplayId":"23918","x":508144,"y":1107138,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초18, 서초08, 서초18-1, 강남02, 강남10, 서초21, 서초20, 강남07"}]},{"busStopId":"BS68158","busStopName":"양재역","busStopDisplayId":"23841","x":508574,"y":1107212,"busInfo":[{"busType":"공항","busTypeCode":"AIRPORT","busList":"6009"}]}],"arroundInfo":"대림아파트, 럭키아파트, 서울시농업기술센터, 언주초등학교, 은광여중,고교, 한신아파트, 강남여성인력개발센터, 도곡동, 은성중학교"},{"exitNum":"5","roadview":{"panoid":1077972101,"tilt":-10,"pan":153.0,"wphotox":507721,"wphotoy":1106985,"rvlevel":0},"busstop":[{"busStopId":"11220661030","busStopName":"양재역.양재1동민원분소","busStopDisplayId":"22270","x":508248,"y":1107053,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"N37, 402, 406, 641"},{"busType":"지선","busTypeCode":"GREEN","busList":"3012, 4319, 4435, 4433, 4412"}]},{"busStopId":"BS110802","busStopName":"양재역.양재1동민원분소","busStopDisplayId":"90156","x":508242,"y":1107051,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"M7426"},{"busType":"일반","busTypeCode":"GENERAL","busList":"6, 917, 11-3"}]},{"busStopId":"BS117695","busStopName":"양재역.종합복지관","busStopDisplayId":"23920","x":508019,"y":1107012,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초18-1, 강남02, 서초18, 서초08"}]},{"busStopId":"BS117697","busStopName":"양재역5번출구","busStopDisplayId":"23921","x":508096,"y":1107025,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초21, 강남10"}]},{"busStopId":"BS68170","busStopName":"양재역","busStopDisplayId":"23827","x":508508,"y":1107098,"busInfo":[{"busType":"공항","busTypeCode":"AIRPORT","busList":"6009"}]}],"arroundInfo":"양재동우체국, 양재종합사회복지관, 양재1동주민센터, 양재파출소"},{"exitNum":"6","roadview":{"panoid":1077643539,"tilt":1,"pan":20.0,"wphotox":507656,"wphotoy":1106852,"rvlevel":0},"arroundInfo":"양재1동주민센터, 양재종합사회복지관, 양재동"},{"exitNum":"7","roadview":{"panoid":1077643537,"tilt":1,"pan":-21.0,"wphotox":507704,"wphotoy":1106794,"rvlevel":0},"busstop":[{"busStopId":"11220661033","busStopName":"양재역","busStopDisplayId":"22290","x":507824,"y":1106708,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"405"},{"busType":"지선","busTypeCode":"GREEN","busList":"3412, 4435, 4432"}]},{"busStopId":"BS110807","busStopName":"양재역","busStopDisplayId":"90087","x":507829,"y":1106703,"busInfo":[{"busType":"일반","busTypeCode":"GENERAL","busList":"19, 917, 6, 11-3"}]},{"busStopId":"BS114401","busStopName":"양재역7번출구.프라자약국","busStopDisplayId":"22440","x":507772,"y":1106771,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초09, 서초18, 서초20, 서초18-1"}]}],"arroundInfo":"양재1동주민센터, 서초구민회관, 양재동 꽃 시장, 양재종합사회복지관"},{"exitNum":"8","roadview":{"panoid":1077643450,"tilt":-3,"pan":104.0,"wphotox":507954,"wphotoy":1106496,"rvlevel":1},"busstop":[{"busStopId":"11220661053","busStopName":"양재역.서초문화예술회관(중)","busStopDisplayId":"22004","x":508124,"y":1106274,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"471, 542, 140, 440, 470, 641, 400, 421, 462, 407, 441"},{"busType":"광역","busTypeCode":"RED","busList":"9711A, 9408, 9404"}]},{"busStopId":"BS100582","busStopName":"양재역.서초문화예술회관(중)","busStopDisplayId":"31008","x":508135,"y":1106261,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"9802, 9500, 9501"}]},{"busStopId":"BS69914","busStopName":"양재역.서초문화예술회관(중)","busStopDisplayId":"90085","x":508130,"y":1106268,"busInfo":[{"busType":"일반","busTypeCode":"GENERAL","busList":"33, 500-5"},{"busType":"직행","busTypeCode":"DIRECT","busList":"1570, 3007, 1550, 3100, 9700, 3002, 1005-1, 3030"}]}],"arroundInfo":"양재천방면 , 현대테라하임아파트, 서초구청, 양재고등학교, 외교안보연구원, 참포도나무병원"},{"exitNum":"9","roadview":{"panoid":1077649937,"tilt":-5,"pan":-132.0,"wphotox":507857,"wphotoy":1106525,"rvlevel":0},"busstop":[{"busStopId":"11220661052","busStopName":"양재역.서초문화예술회관(중)","busStopDisplayId":"22003","x":507939,"y":1106460,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"471, 140, 440, 470, 421, 542, 462, 407, 441, 541"},{"busType":"광역","busTypeCode":"RED","busList":"9711A, 9408, 9404"}]},{"busStopId":"BS118850","busStopName":"양재역.서초문화예술회관(중)","busStopDisplayId":"31010","x":507943,"y":1106455,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"9501, 9802, 9500"}]},{"busStopId":"BS217451","busStopName":"양재역신한은행앞","busStopDisplayId":"90257","x":507803,"y":1106536,"busInfo":[{"busType":"일반","busTypeCode":"GENERAL","busList":"6, 19, 917, 11-3"}]},{"busStopId":"BS217452","busStopName":"양재역9번출구/양재역엘타워빌딩","busStopDisplayId":"90254","x":507889,"y":1106436,"busInfo":[{"busType":"직행","busTypeCode":"DIRECT","busList":"5300-1, 5300, 8201, 1311"}]},{"busStopId":"BS217456","busStopName":"양재역커피빈앞","busStopDisplayId":"90256","x":507785,"y":1106556,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"M5422"},{"busType":"직행","busTypeCode":"DIRECT","busList":"3900전, 3007, 3003, 3000, 3002"}]},{"busStopId":"BS217457","busStopName":"양재역9번출구/양재역엘타워빌딩","busStopDisplayId":"90255","x":507883,"y":1106443,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"M6410"},{"busType":"직행","busTypeCode":"DIRECT","busList":"3102, 3100, 3101"},{"busType":"시외","busTypeCode":"INTERCITY","busList":"700"}]},{"busStopId":"BS69913","busStopName":"양재역.서초문화예술회관(중)","busStopDisplayId":"90084","x":507933,"y":1106469,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"M4434, M4403"},{"busType":"일반","busTypeCode":"GENERAL","busList":"33, 500-5"},{"busType":"직행","busTypeCode":"DIRECT","busList":"6002, 1005, 8501, 3100, 5006, 1251, 4403, 1151, 9700, 1005-1, G5100, 8101, 6002-1, 6001, 5003, 1551B, 6501, 1551, 1550-1, 5100, 5002, 5001-1, 1560, 9004, 5001, 3030"}]}],"arroundInfo":"서초구민회관, 신한은행, 서울가정법원, 행정법원"},{"exitNum":"10","roadview":{"panoid":1077649836,"tilt":-3,"pan":-101.0,"wphotox":507653,"wphotoy":1106769,"rvlevel":0},"busstop":[{"busStopId":"11220661049","busStopName":"양재역","busStopDisplayId":"22289","x":507660,"y":1106709,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"405, 400"},{"busType":"지선","busTypeCode":"GREEN","busList":"4435, 3412, 4432"}]},{"busStopId":"BS114520","busStopName":"양재역.환승주차장","busStopDisplayId":"22514","x":507607,"y":1106778,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초18-1, 서초20, 서초18, 서초09, 서초08"}]},{"busStopId":"BS163662","busStopName":"양재역","busStopDisplayId":"31073","x":507697,"y":1106664,"busInfo":[{"busType":"광역","busTypeCode":"RED","busList":"9100, 9201, 9200, M6405, 9300"}]}],"arroundInfo":"양재고등학교, 서초구청, 서초보건소"},{"exitNum":"11","roadview":{"panoid":1077649802,"tilt":-10,"pan":-86.0,"wphotox":507529,"wphotoy":1106902,"rvlevel":0},"arroundInfo":"서초보건소, 서초구청, 양재고등학교, 외교센터, 성남,분당방면"},{"exitNum":"12","roadview":{"panoid":1077972073,"tilt":-4,"pan":-172.0,"wphotox":507423,"wphotoy":1106920,"rvlevel":-1},"busstop":[{"busStopId":"11220521015","busStopName":"외교안보연구원.서초구청","busStopDisplayId":"22130","x":506768,"y":1106793,"busInfo":[{"busType":"간선","busTypeCode":"BLUE","busList":"405, 406"}]},{"busStopId":"BS115060","busStopName":"서초구청","busStopDisplayId":"22863","x":507162,"y":1106859,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초17"}]},{"busStopId":"BS115244","busStopName":"외교안보연구원.서초구청","busStopDisplayId":"22854","x":506761,"y":1106792,"busInfo":[{"busType":"마을","busTypeCode":"MAUL","busList":"서초21, 서초17"}]},{"busStopId":"BS250526","busStopName":"외교안보연구원.서초구청","busStopDisplayId":"90144","x":506777,"y":1106795,"busInfo":[{"busType":"일반","busTypeCode":"GENERAL","busList":"19"}]}],"arroundInfo":"서초구청, 서초보건소, 양재고교, 외교센터"}]},"fastTransInfo":{"subwayid":"SES34","subwayName":"신분당선","list":[{"direction":"강남방향","tranferInfoList":[{"subwayid":"SES3","subwayName":"3호선","direction":"상하행","door":"1-1번문"}]},{"direction":"정자방향","tranferInfoList":[{"subwayid":"SES3","subwayName":"3호선","direction":"상하행","door":"5-4번문"}]}]}}


