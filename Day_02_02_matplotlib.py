import matplotlib.pyplot as plt
from matplotlib import font_manager, rc, colors, cm
import numpy as np
import csv


# 가장 기본적이고 가장 대표적인 시각화 library

def test_1():
    plt.plot([10, 20, 30, 40, 50]) # y축 data만 들어감
    plt.show()


def test_2():
    # (1,1), (2,2) 이게 더 편할 것 같지만 이런 데이터 만들기 매우 어렵다.
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro') # x data와 y data 연결하지 말고 점으로 표시하자 ro
    #plt.axis([0, 6, 0, 20]) # 그래프에 범위 설정해 줌
    plt.xlim(0, 6)
    plt.ylim(0, 20)
    plt.show()


def test_3():
    #문제
    #x의 범위가 (-10, 10)일 때의 x^2 그래프를 그려 보세요.
    '''
    x = [int(x) for x in range(-10,11)]
    y = [x ** 2 for x in x]
    plt.plot(x,y)
    plt.show()
    '''
    #plt.plot(range(-10, 10), range(-10, 10) ** 2) # 안됨. 1단위로만 할 수 있음, broadcasting 안됨
    plt.plot(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1) ** 2) # broadcasting 연산 numpy에만 가능
    plt.show()


def test_4():
    x = np.linspace(-10, 10, 100)
    y = np.sin(x)

    plt.plot(x, y)
    # plt.plot(x, y, marker='x')
    plt.plot(x, y, 'rx')
    plt.show()

def test_5():
    fig, ax = plt.subplots() #ax이용하면 특정 그래프 접근 가능
    ax.grid(True)

    x1 = np.arange(0.01, 2, 0.01)
    plt.plot(x1, np.log(x1), 'r')
    plt.plot(x1, -np.log(x1), 'b')
    x2 = np.arange(0.01-2, 0, 0.01)
    plt.plot(x2, np.log(-x2), 'y')
    plt.plot(x2, -np.log(-x2), 'g') #k는 black

    plt.xlim(-2,2)
    plt.show()

def test_6():
    def func(t):
        return np.exp(-t) * np.cos(2*np.pi*t) #matplotlib의 tutorial

    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)

    plt.figure(1)
    plt.subplot(221) #2행 2열의 1번째에 그려라

    plt.plot(t1, func(t1), 'bo')
    plt.plot(t2, func(t2), 'k')

    #plt.figure(2) #새로운 그래프를 만듦
    #plt.subplot(224) # figure 안에 영역을 나눔
    plt.subplot(2, 1, 2)
    #plt.subplot(212)
    plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

    plt.show()


def test_7():
    # 문제
    # test_5()에서 사용했던 Log 그래프를
    # 첫 번째 Figure에 각각 2개, 두 번째 Figure에 각각 2개씩 그립니다.

    fig, ax = plt.subplots()  # ax이용하면 특정 그래프 접근 가능
    ax.grid(True)
    fig = plt.figure(1)
    x1 = np.arange(0.01, 2, 0.01)

    plt.subplot(2, 1, 1)
    fig.gca().grid(True)
    plt.plot(x1, np.log(x1), 'r')
    plt.xlim(-2, 2)

    plt.subplot(2, 1, 2)
    fig.gca().grid(True)
    plt.plot(x1, -np.log(x1), 'b')
    plt.xlim(-2, 2)

    fig = plt.figure(2)

    x2 = np.arange(0.01 - 2, 0, 0.01)

    plt.subplot(2, 1, 1)
    fig.gca().grid(True)
    plt.plot(x2, np.log(-x2), 'y')
    plt.xlim(-2, 2)

    plt.subplot(2, 1, 2)
    fig.gca().grid(True)
    plt.plot(x2, -np.log(-x2), 'g')  # k는 black
    plt.xlim(-2, 2)

    plt.show()


def test_8():
    means_men = ([20, 35, 30, 35, 27])
    means_women = ([25, 32, 34, 20, 25])

    n_group = len(means_men)
    index = np.arange(n_group)
    bar_width = 0.45
    opacity = 0.4

    plt.bar(index , means_men, bar_width, alpha=opacity, color = 'b', label= 'Men')
    plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color = 'r', label= 'Women')

    plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
    plt.tight_layout() #주변의 여백을 없애줘서 꽉 차게 만든다. 사람들이 보통 이게 더 좋다고 얘기한단다.
    plt.show()




#test_8()

def matplotlib_ex():
    # http://matplotlib.org/gallery/pyplots/compound_path_demo.html#sphx-glr-gallery-pyplots-compound-path-demo-py
    import numpy as np

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.path as path

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # histogram our data with numpy
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)

    nverts = nrects*(1+3+1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5,0] = left
    verts[0::5,1] = bottom
    verts[1::5,0] = left
    verts[1::5,1] = top
    verts[2::5,0] = right
    verts[2::5,1] = top
    verts[3::5,0] = right
    verts[3::5,1] = bottom

    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.show()

def test_9():
    f = open('Data/2016_GDP.txt', 'r', encoding='utf-8')

    '''
    for row in f:
        print(row)
    '''
    f.readline() #필요 없는 첫 번째 라인 건너뜀

    names, money = [], []
    for row in csv.reader(f, delimiter=':'):
        #print(row)
        names.append(row[1])
        money.append(int(row[-1].replace(',',''))) #필요 없는 쉼표 없애기
    # print(money)

    f.close()
    path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    print(font_name)
    rc('font', family =font_name)

    top10_names = names[:10]
    top10_money = money[:10]

    index = np.arange(10)

    # plt.bar(index, top10_money)
    # plt.bar(index, top10_money, color=colors.BASE_COLORS)
    plt.bar(index, top10_money, color=colors.TABLEAU_COLORS) #은은한게 왠지 내가 더 일을 한듯한.. 잘하기 어려운 것
    # plt.bar(index, top10_money, color='rgb')
    # plt.bar(index, top10_money, color=['red', 'green', 'black'])
    # plt.bar(index, top10_money, color=colors.CSS4_COLORS)
    # color map 도 있는데 나중에 다시

    plt.xticks(index, top10_names)
    plt.xticks(rotation = -90) # 270하고 같음
    plt.xticks(rotation = 60)
    plt.title('top10')
    plt.tight_layout(pad = 1)
    # plt.subplots_adjust(left = 0.5, right= 0.3) # error
    # plt.subplots_adjust(left=0.5, right=0.8) #left top right bottom
    plt.subplots_adjust(bottom = 0.2, top=0.9, left = 0.1, right = 0.95)
    #plt.xticks(index, top10_names, rotation = 45) # 한글은 폰트 설정 해야됨
    #plt.xticks(index, top10_names, rotation= 'vertical')

    plt.show()

test_9()