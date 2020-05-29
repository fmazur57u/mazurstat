# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:10:59 2020

@author: Florian
"""

import numpy as np
import matplotlib.pyplot as plt
from  do_pca import my_pca
import pandas

fig = plt.figure()
coords_list = np.loadtxt('PCA_list.txt')
columns= range(0, 372)
index = range(0, 285)

X = pandas.DataFrame(data=coords_list,index=index,columns=columns)
#Number of observations
n = X.shape[0]


p = X.shape[1]

eigvals, eigvecs = my_pca(coords_list)

y = (eigvals/np.sum(eigvals))*100
#scree plot

plt.plot(np.arange(1, p + 1), y, marker='o')
plt.title("Scree plot")
plt.ylabel("Eigen values (%)")
plt.xlabel("Principal component number")
plt.show()
fig.savefig('scree_plot.pdf', format='pdf')