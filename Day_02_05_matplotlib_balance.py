import xlrd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager, rc, colors
import numpy as np

path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = path).get_name()
rc('font',family = font_name)

def read_import_export():
    wb = xlrd.open_workbook('Data\국가별수출입 실적_201711305.xls')

    sheets = wb.sheets()
    print(sheets)

    sheet = sheets[0]
    print(sheet.nrows)

    result = []
    for row in range(6, sheet.nrows):
        #print(sheet.row_values(row))

        values = sheet.row_values(row)

        country = values[1]
        outcome = int(values[3].replace(',',''))
        income = int(values[5].replace(',', ''))
        balance = int(values[6].replace(',', ''))

        result.append([country, outcome, income, balance])

    return result


def sorted_top10(result):
    result_sorted = sorted(result,
                           key=lambda t: t[-1],
                           reverse=True) #key는 무엇으로 정렬할지 알려줌
    #print(*result_sorted, sep='\n')
    # 문제
    # 1. 흑자 상위 10개국에 대해 막대 그래프를 그리세요
    # 2. 적자 상위 10개국에 대해 막대 그래프를 그리세요.
    # 3. 흑자, 적자를 하나의 그래프에 표현해주세요. (왼쪽 10개, 오른쪽 10개)
    #      적자 10개, 적자 금액,      흑자 10개,    흑자 금액
    '''
    red_names = [result_sorted[x][0] for x in range(-1,-11,-1)]
    red_balances = [result_sorted[x][-1] for x in range(-1,-11,-1)]
    black_names = [result_sorted[x][0] for x in range(10)]
    black_balances = [result_sorted[x][-1] for x in range(10)]

    plt.figure(1)

    n_group = len(red_balances)
    index1 = np.arange(n_group)
    bar_width = 0.45
    opacity = 0.4

    plt.bar(index1, red_balances, bar_width, alpha=opacity, color='r', label='red balaces')
    #plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')
    plt.xticks(index1 + bar_width / 2, (red_names))
    plt.xticks(rotation=45)
    #plt.xlim()
    plt.tight_layout()  # 주변의 여백을 없애줘서 꽉 차게 만든다. 사람들이 보통 이게 더 좋다고 얘기한단다.
    #plt.show()

    plt.figure(2)
    n_group = len(black_balances)
    index2 = np.arange(n_group)
    bar_width = 0.45
    opacity = 0.4

    plt.bar(index2, black_balances, bar_width, alpha=opacity, color='k', label='black balaces')
    # plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')
    plt.xticks(index2 + bar_width / 2, (black_names))
    plt.xticks(rotation=45)
    # plt.xlim()
    plt.tight_layout()  # 주변의 여백을 없애줘서 꽉 차게 만든다. 사람들이 보통 이게 더 좋다고 얘기한단다.
    #plt.show()

    plt.figure(3)

    plt.bar(index1, black_balances, bar_width, alpha=opacity, color='k', label='black balaces')
    plt.bar(index1 + bar_width, red_balances, bar_width, alpha=opacity, color='r', label='red balances')
    plt.xticks(index1 + bar_width / 2, (black_names))
    plt.xticks(rotation=45)
    # plt.xlim()
    plt.tight_layout()  # 주변의 여백을 없애줘서 꽉 차게 만든다. 사람들이 보통 이게 더 좋다고 얘기한단다.
    #plt.show()
    '''

    ##################################################
    #for row in result_sorted[:10]:
    #    print(row)
    red_names = []
    black_names = []
    red_balances = []
    black_balances = []

    for country, _, _, balance in result_sorted[:-11:-1]: # placeholder 변수로 사용하지 않겠다. 나한테는 지금 필요 없다.
        #print(country, balance)
        red_names.append(country)
        red_balances.append(balance)

    for country, _, _, balance in result_sorted[:10]: # placeholder 변수로 사용하지 않겠다. 나한테는 지금 필요 없다.
        #print(country, balance)
        black_names.append(country)
        black_balances.append(balance)


    return red_names, red_balances, black_names, black_balances

def draw_balance(names,balances):
    formatter = FuncFormatter(lambda x, pos:int(x //10000))
    _, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)

    plt.bar(range(len(names)), balances, color = colors.TABLEAU_COLORS)
    plt.xticks(range(len(names)), names, rotation='vertical')
    plt.show()

def draw_balance_in_onefigure(red_names, red_balances, black_names, black_balances):
    formatter = FuncFormatter(lambda x, pos:int(x //10000))
    _, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)

    names = black_names + red_names[::-1]
    balances = black_balances +red_balances[::-1]

    plt.bar(range(len(names)), balances, color = ['black'] * 10 + ['red'] * 10)
    plt.xticks(range(len(names)), names, rotation='vertical')
    plt.show()

result = read_import_export()
#print(*result, sep='\n')
red_names, red_balances, black_names, black_balances = sorted_top10(result)
#draw_balance(black_names, black_balances)
#draw_balance(red_names, red_balances)
draw_balance_in_onefigure(red_names, red_balances, black_names, black_balances)