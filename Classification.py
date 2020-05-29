# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:22:28 2020

@author: Florian
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

fig = plt.figure()
coords_list = np.loadtxt('PCA_list.txt')

Z = linkage(coords_list,method='ward',metric='euclidean')

plt.title("CAH")
dendrogram(Z ,orientation='left',color_threshold=0)
plt.show()
fig.savefig('CAH.pdf', format='pdf')