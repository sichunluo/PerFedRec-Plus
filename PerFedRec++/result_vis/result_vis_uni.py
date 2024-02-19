from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import random
import os

filepath = './yelp_test/train_ml100k.txt_PerFedRec_plus_item.pt'
post_fix = '+'

def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
seed_torch(seed=42)

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
    plt.savefig('./'+'tsne_ang'+post_fix+'.png')


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = ( (data - x_min) / (x_max - x_min) ) *0.98

    fig = plt.figure()
    ax = plt.subplot(111)
    mid = int(data.shape[0] / 2)
    print(data)
    for i in range(mid):
        plt.text(data[i, 0], data[i, 1], str("."),  # 此处的含义是用什么形状绘制当前的数据点
                color=plt.cm.Set1(2 / 10),  # 表示颜色
                fontdict={'weight': 'bold', 'size': 9})  # 字体和大小
    for i in range(mid):
        plt.text(data[mid + i, 0], data[mid + i, 1], str("."),
                color=plt.cm.Set1(2 / 10),
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])

    plt.title(title)
    plt.show()
    plt.savefig('./'+'tsne'+post_fix+'.png')


def mian(data, label):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label,'')
    plot_angle(data)



import torch
import pickle
a = torch.load(filepath)
print(a, a.shape)
item_emb = a.detach().cpu().numpy()

keep_cols = np.random.choice(item_emb.shape[0], size=2000, replace=False)

item_emb = item_emb[ keep_cols, :]

print(item_emb.shape)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

plt.rcParams.update({'font.size': 30})

fs = 30

print(plt.rcParams['font.family'])
from matplotlib.font_manager import FontProperties
font = FontProperties(family='sans-serif', size=30)

tsne = TSNE(n_components=2, random_state=0)
print(item_emb.shape)
item_emb_tsne = tsne.fit_transform(item_emb)

angles = np.arctan2(item_emb_tsne[:, 1], item_emb_tsne[:, 0])
angles = np.mod(angles, 2 * np.pi)

x = np.cos(angles)
y = np.sin(angles)

xy = np.vstack([x, y])
kde_xy = gaussian_kde(xy)(xy)

angles_adjusted = (angles + np.pi) % (2 * np.pi) - np.pi  # 将范围调整到 -π 到 π

sns.set(style="white")

fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

scatter = axes[0].scatter(x, y, c=kde_xy, cmap='Greens')
axes[0].set_xlabel('Features', fontproperties=font)
axes[0].axis('equal')
axes[0].set_yticks([-1,-0.5,0,0.5,1])

sns.kdeplot(angles_adjusted, shade=True, color="green", ax=axes[1])
axes[1].set_xlabel('Angle', fontsize=fs)
axes[1].set_ylabel('Density', fontsize=fs)
axes[1].set_xlim(-np.pi, np.pi)
axes[1].set_ylim(0, 0.3)
axes[1].set_yticks([0,0.3])

axes[0].tick_params(axis='both', which='major', labelsize=fs)
axes[1].tick_params(axis='both', which='major', labelsize=fs)
plt.tight_layout()

plt.savefig(''+'yelp_item_uni+'+'.png')
plt.show()



