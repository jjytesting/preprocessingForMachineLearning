import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as color

def thing():
    print('red'*3)
    print(['red']*3)
    print([['red']*3, ['black']*3])
    print(['red']*3,['black']*3)
    print(['red']*3 + ['black']*3)

    #redredred
    #['red', 'red', 'red']
    #[['red', 'red', 'red'], ['black', 'black', 'black']]
    #['red', 'red', 'red'] ['black', 'black', 'black']
    #['red', 'red', 'red', 'black', 'black', 'black']

def color_1():
    x = np.random.rand(100)
    y = np.random.rand(100)
    t = np.arange(100) # 0부터 99까지의 값인데 색상 값으로 넣어버리면..

    plt.scatter(x, y, c=t)
    plt.show()

def color_2():
    x = np.arange(100)
    y = x
    t = x # 0부터 99까지의 값인데 색상 값으로 넣어버리면..

    plt.scatter(x, y, c=t)
    #plt.scatter(x, -y, c=t, cmap='viridis')
    #plt.scatter(x, -y, c=[cm.viridis(0), cm.viridis(255)])
    plt.scatter(x, -y, c=[cm.viridis(0)] * 50 + [cm.viridis(255)] * 50)
    #plt.scatter(x, -y, c=[cm.viridis(0), cm.viridis(2550000)]) #255 넘으면 255 들어감. 총 256색
    plt.show()

def color_3():
    print(plt.colormaps())
    print(len(plt.colormaps()))
    #['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
    #  'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
    # 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
    # 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
    # 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', ...]

color_3()
