import pickle
import numpy as np
import umap
import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# import np_utils

def vis_clean_cluster():
    data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/xs_cifar10.npy')
    data = data.reshape(data.shape[0], 32*32)
    sample = np.random.randint(data.shape[0], size=3000)
    data = data[sample, :]

    y_s = np.load('/remote-home/cs_igps_yangjin/cifar10_data/ys_cifar10.npy')
    y_s = y_s[sample]

    fit = umap.UMAP(random_state=42, n_components=3)
    u = fit.fit_transform(data)

    plt.scatter(u[:, 0], u[:, 1], c=y_s, cmap='Spectral', s=14)
    plt.gca().set_aspect('equal', 'datalim')
    clb = plt.colorbar(boundaries=np.arange(11) - 0.5)
    clb.set_ticks(np.arange(10))
    clb.ax.tick_params(labelsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'MNIST(Clean)', fontsize=16);
    if not os.path.exists('./img'):
        os.makedirs('./img')
    plt.savefig(f'img/MNIST_clean2.png', dpi=300)
    plt.clf()

if __name__ == '__main__':
    vis_clean_cluster()

# data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/xs_cifar10.npy')
# data = data.reshape(data.shape[0], 32*32*3)
# sample = np.random.randint(data.shape[0], size=3000)
# data = data[sample, :]
#
# pca = PCA(n_components=9)# 总的类别
# pca_result = pca.fit_transform(data)
#
# #Run T-SNE on the PCA features.
# tsne = TSNE(n_components=2, verbose = 1)
# tsne_results = tsne.fit_transform(pca_result[:5000])
#
# #-------------------------------可视化--------------------------------
# y_s = np.load('/remote-home/cs_igps_yangjin/cifar10_data/ys_cifar10.npy')
# y_s = y_s[sample]
#
# y_test_cat = np_utils.to_categorical(y_s, num_classes = 10)# 总的类别
# color_map = np.argmax(y_test_cat, axis=1)
# plt.figure(figsize=(10,10))
# for cl in range(10):# 总的类别
#     indices = np.where(color_map==cl)
#     indices = indices[0]
#     plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
# plt.legend()
# plt.show()