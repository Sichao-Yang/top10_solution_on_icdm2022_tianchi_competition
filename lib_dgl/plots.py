import matplotlib.pyplot as plt
import random, json, glob
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')

def lineplot(xs,
            ys=[],
            title='no title',
            legend=[],
            xlabel='x', 
            ylabel='y', 
            save_path='tmp.png',
            figsiz=(8,6)
    ):
    """oneliner to draw & save line plot

    Args:
        xs (list of 1d np array): list of x for lines
        ys (list): list of y
        title (str): name show on plot
        legend (str): 
        xlabel (str): 
        ylabel (str): 
        save_path (_type_): path should end with '.png'
        figsiz (tuple): (width, height)
    """
    plt.figure(figsize=figsiz, dpi=100)
    if len(xs)!=len(ys) or len(xs)!=len(legend):
        raise ValueError('lengh of xs should be equal to ys and legend!')
    for x, y in zip(xs, ys):
        plt.plot(x,y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)
    plt.savefig(save_path)
    plt.close()

def scatterplot(x,
                y,
                title='no title',
                xlabel='x', 
                ylabel='y', 
                save_path='tmp.png',
                figsiz=(8,6)
    ):
    """oneliner for 2d scatterplot used in classification result presentation

    Args:
        x (2d numpy array): for the xy dimension
        y (1d numpy array): for classes index
        title (str, optional): _description_. Defaults to 'no title'.
        xlabel (str, optional): _description_. Defaults to 'x'.
        ylabel (str, optional): _description_. Defaults to 'y'.
        save_path (str, optional): _description_. Defaults to 'tmp.png'.
        figsiz (tuple, optional): figure (width, height). Defaults to (8,6).
    """
    plt.figure(figsize=figsiz, dpi=100)
    labels = list(np.unique(y))
    for label in labels:
        idx = np.where(y==label)[0]
        plt.scatter(x[idx,0], x[idx,1], label=str(label))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def barplot(xs,
            ys=[],
            title=None,
            legend=[],
            xlabel='x', 
            ylabel='y', 
            save_path='tmp.png',
            figsiz=(8,6),
            width=0.5
            ):
    """oneliner to draw & save bar plot

    Args:
        xs (list of 1d np array): list of x for lines
        ys (list): list of y
        title (str): name show on plot
        legend (str): 
        xlabel (str): 
        ylabel (str): 
        save_path (_type_): path should end with '.png'
        figsiz (tuple): (width, height)
    """
    plt.figure(figsize=figsiz, dpi=100)
    for x, y in zip(xs, ys):
        plt.bar(x, height=y, width=width)
    if title:
        plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)
    plt.savefig(save_path)
    plt.close()
    

def save_subplot(dims, x, y1, y2, skip, title, xlabel, ylabel, save_path, idx_list=None):
    m, n = dims
    _, axs = plt.subplots(m,n)
    idx = 1
    for i in range(0,m):
        if idx_list != None:
            idx = idx_list[i]
        else:
            idx = i
        axs[i].plot(x,y1[:,skip*idx])
        axs[i].plot(x,y2[:,skip*idx],linestyle='dashed')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.legend()
    plt.savefig(save_path)
    plt.close()

def drawplot(true, pred, resroot):
    m, targets = true.shape
    x = range(m)
    skip = 24
    ylabel = "(kW)"
    xlabel = "(per 15mins)"
    save_path = resroot+'/type1.png'
    title = f"1D+24hrs Power Forecasting graphed every {skip//(60/15)} hours"
    dims = [targets//skip, 1]
    save_subplot(dims, x, true, pred, skip, title, xlabel, ylabel, save_path)

    sample_days = 5
    day_list = random.sample(range(m//96), k=sample_days)
    x = range(96)
    save_path = resroot+'/type2.png'
    title = f"24+(1~{targets//4})hrs Power Forecasting sampled on day: {day_list}"
    dims = [len(day_list), 1]
    y1 = np.transpose(true)
    y2 = np.transpose(pred)
    skip = 96
    save_subplot(dims, x, y1, y2, skip, title, xlabel, ylabel, save_path, day_list)

    x = range(96)
    for i in range(26):
        y1 = true[i*96,:]
        y2 = pred[i*96,:]
        save_path = resroot+f'/day{i}.png'
        title = f"Power Forecasting on day{i} started from 20210220"
        lineplot(x, y1, y2, title, xlabel, ylabel, save_path)


class Expplots:
    def __init__(self, dir='./log'):
        self.dir = dir
        self.scores = {'train':[], 'valid':[]}
        self.losses = {'train':[], 'valid':[]}
        self.lr = []
        self.epochs = []
    
    def new_epoch_draw(self, scores, losses, lr, epoch, tag=''):
        for key in self.scores:
            self.scores[key].append(scores[key])
            self.losses[key].append(losses[key])
        self.epochs.append(epoch)
        self.lr.append(lr)
        self._draw(tag)
    
    def _draw(self, tag):
        ks = self.scores.keys()
        lineplot(xs=[self.epochs, self.epochs], ys=[self.scores[k] for k in ks], title='scores', legend=['train', 'valid'],
                xlabel='epochs', ylabel='scores', save_path=self.dir+f'/{tag}_scores.png', figsiz=(8,6))
        lineplot(xs=[self.epochs, self.epochs], ys=[self.losses[k] for k in ks], title='losses', legend=['train', 'valid'],
                xlabel='epochs', ylabel='losses', save_path=self.dir+f'/{tag}_losses.png', figsiz=(8,6))
        lineplot(xs=[self.epochs], ys=[self.lr], title='lr', legend=[''],
                xlabel='epochs', ylabel='lr', save_path=self.dir+f'/{tag}_lr.png', figsiz=(8,6))


if __name__ == "__main__":
    count = 20
    x = np.arange(count)
    a = np.arange(-1,1.5,0.5)
    y = np.random.choice(a, count)
    y2 = np.random.choice(a, count)
    width = 0.2
    legend = ['1', '2']
    barplot([x,x+width], [y,y2], width=width, legend=legend)
        # for name in glob.glob(resroot+'/*'):
        #     if name.split('.')[-1] == 'csv':
        #         path2 = name

        # drawplot(true, pred.values, resroot)