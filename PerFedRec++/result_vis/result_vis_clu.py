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
# import torch

filepath = './yelp_test/train_PerFedRec_user.pt'
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

    # plt.xticks([-0.1, 1.1])  # 坐标轴设置
    # xticks(locs, [labels], **kwargs)locs，是一个数组或者列表，表示范围和刻度，label表示每个刻度的标签
    # plt.yticks([0, 1.1])
    plt.xticks([])
    plt.yticks([])

    plt.title(title)
    plt.show()
    plt.savefig('./'+'tsne'+post_fix+'.png')


def mian(data, label):
    # , n_samples, n_features = get_data()
    # print(data, label)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label,'')
    plot_angle(data)


    
# 
import torch
import pickle
a = torch.load(filepath)
print(a, a.shape)
item_emb = a.detach().cpu().numpy()



print(item_emb.shape)



embeddings = item_emb

kmeans = KMeans(n_clusters=10, random_state=41)
clusters = kmeans.fit_predict(embeddings)

# Step 3: Randomly select 100 users from each cluster
selected_users = []
sss = random.sample([_ for _ in range(10)],5)
print(sss)
for i in sss:
    cluster_indices = np.where(clusters == i)[0]
    selected_indices = random.sample(list(cluster_indices), 100)
    selected_users.extend(embeddings[selected_indices])

# selected_users = torch.stack(selected_users)

# Step 4: Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
reduced_embeddings = tsne.fit_transform(selected_users)

# Step 5: Plot the reduced embeddings
plt.figure(figsize=(10, 8))
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue')

colors = plt.cm.rainbow(np.linspace(0, 1, 5))
for i in range(5):
    plt.scatter(reduced_embeddings[i*100:(i+1)*100, 0], reduced_embeddings[i*100:(i+1)*100, 1], color=colors[i], label=f'Cluster {i+1}')


# plt.title("t-SNE Visualization of Selected Users")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
plt.xticks([])
plt.yticks([])

plt.savefig(''+'test'+'.png')
plt.show()



