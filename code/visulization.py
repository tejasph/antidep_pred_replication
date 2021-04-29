# -*- coding: utf-8 -*-
"""
Quick script to do some visualization, used for an old class project
but not the current paper
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

pathData = '../data/canbind-clean-aggregated-data.with-id.csv'
pathLabel = '../data/targets.csv'
# read data and chop the header
X = np.genfromtxt(pathData, delimiter=',')
y = np.genfromtxt(pathLabel, delimiter=',')[:,1]
X = X[1:,]
X = X[:,2:]
X = np.append(y.reshape(-1,1),X,axis=1)

Z = TSNE(n_components=2).fit_transform(X)
plt.ylabel('z2')
plt.xlabel('z1')
plt.title('t-SNE')
one = Z[np.where(y==1)[0],:]
zero = Z[np.where(y==0)[0],:]
res = plt.scatter(one[:,0], one[:,1], c='blue')
notres = plt.scatter(zero[:,0], zero[:,1], c='red')
plt.legend((res,notres),("Respond","Not Respond"))
plt.savefig("../figs/t-SNE")
plt.gcf().clear()

Z = Isomap(n_components=2,n_neighbors=5).fit_transform(X)
plt.ylabel('z2')
plt.xlabel('z1')
plt.title('Isomap')
one = Z[np.where(y==1)[0],:]
zero = Z[np.where(y==0)[0],:]
res = plt.scatter(one[:,0], one[:,1], c='blue')
notres = plt.scatter(zero[:,0], zero[:,1], c='red')
plt.legend((res,notres),("Respond","Not Respond"))
plt.savefig("../figs/Isomap")
plt.gcf().clear()

Z = PCA(n_components=2).fit_transform(X)
plt.ylabel('z2')
plt.xlabel('z1')
plt.title('PCA')
one = Z[np.where(y==1)[0],:]
zero = Z[np.where(y==0)[0],:]
res = plt.scatter(one[:,0], one[:,1], c='blue')
notres = plt.scatter(zero[:,0], zero[:,1], c='red')
plt.legend((res,notres),("Respond","Not Respond"))
plt.savefig("../figs/PCA")
