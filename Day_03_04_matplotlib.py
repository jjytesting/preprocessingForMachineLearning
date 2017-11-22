import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def random_walker():
    np.random.seed(1)

    walks, heights, pos = [], [], 0
    for _ in range(100):
        state = np.random.randint(3) - 1
        pos += state

        print(pos)

        walks.append(state)
        heights.append(pos)
    plt.figure(1)
    plt.plot(walks, 'ro')
    plt.figure(2)
    plt.plot(heights)
    plt.show()

def style_ggplot():
    x = np.linspace(0, 10) #50개 기본

    print(plt.style.available)
    print(len(plt.style.available)) #23개의 style이 있음
    # ['bmh', 'classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark',
    # 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
    # 'seaborn-whitegrid', 'seaborn', '_classic_test']

    with plt.style.context('ggplot'):
        plt.plot(x, np.sin(x) + 0.5 * x + np.random.randn(50))
        plt.plot(x, np.sin(x) + 1.0 * x + np.random.randn(50))
        plt.plot(x, np.sin(x) + 1.5 * x + np.random.randn(50))

    plt.show()

def all_styles():
    x= np.linspace(0, 10)
    n = np.random.randn(50)
    y1 = np.sin(x) + 0.5 * x + n
    y2 = np.sin(x) + 1.0 * x + n
    y3 = np.sin(x) + 1.5 * x + n

    for i, style in enumerate(plt.style.available):
        plt.figure(i+1)

        with plt.style.context(style):
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.plot(x, y3)
            plt.title(style)

    plt.show()

def all_in_one():
    x = np.linspace(0, 10)
    n = np.random.randn(50)
    y1 = np.sin(x) + 0.5 * x + n
    y2 = np.sin(x) + 1.0 * x + n
    y3 = np.sin(x) + 1.5 * x + n

    plt.figure(figsize=(20,12))
    for i, style in enumerate(plt.style.available):

        with plt.style.context(style):
            plt.subplot(4, 6, i + 1)
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.plot(x, y3)
            plt.title(style)

    plt.tight_layout()
    #plt.show()
    plt.savefig('Data/style.png')

# http://avalon.law.yale.edu/20th_century/mlk01.asp
f = open('Data/i_have_a_dream.txt', 'r', encoding='utf-8')
text = f.read()
f.close()


#wc1 = WordCloud().generate(text)
#plt.imshow(wc1, interpolation='bilinear')
#plt.axis('off')

#plt.figure()
wc2 = WordCloud(max_font_size = 40).generate(text)
plt.axis('off')
plt.show()

#https://github.com/amueller/word_cloud
#http://amueller.github.io/word_cloud/auto_examples/colored.html

#http://stencilry.org/main/
#마스킹 사이트