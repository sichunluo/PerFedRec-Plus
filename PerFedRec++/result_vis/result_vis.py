from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import torch
import pickle

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_angle(data):
    X = data
    t_SNE = TSNE(n_components=2, perplexity=30, random_state=42)
    t_SNE_2d = t_SNE.fit_transform(X)
    x = t_SNE_2d[:, 0]
    y = t_SNE_2d[:, 1]
    arctan_yx = np.arctan2(y, x)
    x_range = np.linspace(-np.pi, np.pi, 1000)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(arctan_yx.reshape(-1, 1))
    log_density = kde.score_samples(x_range.reshape(-1, 1))
    density = np.exp(log_density)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_range, density)
    ax.set_xlabel('Angles')
    ax.set_ylabel('Density')
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 18
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.show()
    plt.savefig('./fig/'+'tsne_ang'+'.png')


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = ( (data - x_min) / (x_max - x_min) ) *0.98

    fig = plt.figure()
    ax = plt.subplot(111)
    mid = int(data.shape[0] / 2)
    print(data)
    for i in range(mid):
        plt.text(data[i, 0], data[i, 1], str("."),
                color=plt.cm.Set1(2 / 10),
                fontdict={'weight': 'bold', 'size': 9})
    for i in range(mid):
        plt.text(data[mid + i, 0], data[mid + i, 1], str("."),
                color=plt.cm.Set1(2 / 10),
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    plt.savefig('./fig/'+'tsne'+'.png')


def mian(data, label):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    plot_angle(data)


a = torch.load('./perfedrec+.pt')
item_emb = a.detach().cpu().numpy()
list_ = [0] * item_emb.shape[0]
mian(item_emb, list_)



