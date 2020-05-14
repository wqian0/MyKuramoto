import numpy as np
from numpy import random as nrd
import random as rd
import copy
from random import choice
from array import *
import time
import os
import small_world as sw
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.ndimage as sim
from mpl_toolkits.mplot3d import Axes3D

def read_heatmap(file, n, a_start, a_inc, M_start, M_inc):
    result = np.zeros((n, n))
    for line in file:
        if not line.strip():
            continue
        line = [float(j) for j in line.strip().split('\t')]
        result[n-1-int(round((line[1]-a_start)/a_inc))][int(round((line[2]-M_start)/M_inc))] = line[3]
    return result

coupling_vals = np.linspace(.2, .5, 6)
M_vals = np.linspace(1, 2.5, 6)

ss_diffs_0_int = open("ss_diffs_0_intermediate.txt")
ss_diffs_1_int = open("ss_diffs_1_intermediate.txt")
ss_diffs_2_int = open("ss_diffs_2_intermediate.txt")
ss_diffs_3_int = open("ss_diffs_3_intermediate.txt")
ss_diffs_4_int = open("ss_diffs_4_intermediate.txt")

ss_diffs_acc = open("ss_diffs_accurate.txt")


ss_diffs_0 = open("ss_diffs_full.txt", "r")
ss_diffs_0_100k = open("ss_diffs_0_100k.txt", "r")
ss_diffs_1_to_4 = open("ss_diffs_1_to_4.txt", "r")
ss_diffs_1 = open("ss_diffs_1.txt", "r")
ss_diffs_2 = open("ss_diffs_2.txt", "r")
ss_diffs_3 = open("ss_diffs_3.txt", "r")
ss_diffs_4 = open("ss_diffs_4.txt", "r")

heatmap_0_100k = read_heatmap(ss_diffs_0_100k, 16, .2, .02, 1, .1)
heatmap_1_to_4 = read_heatmap(ss_diffs_1_to_4, 16, .2, .02, 1, .2)

heatmap_0 = read_heatmap(ss_diffs_0, 16, .2, .02, 1, .1)
heatmap_1 = read_heatmap(ss_diffs_1, 16, .2, .02, 1, .1)
heatmap_2 = read_heatmap(ss_diffs_2, 16, .2, .02, 1, .1)
heatmap_3 = read_heatmap(ss_diffs_3, 16, .2, .02, 1, .1)
heatmap_4 = read_heatmap(ss_diffs_4, 16, .2, .02, 1, .1)

heatmap_0_int = read_heatmap(ss_diffs_0_int, 15, .21, .02, 1.05, .1)
heatmap_1_int = read_heatmap(ss_diffs_1_int, 15, .21, .02, 1.05, .1)
heatmap_2_int = read_heatmap(ss_diffs_2_int, 15, .21, .02, 1.05, .1)
heatmap_3_int = read_heatmap(ss_diffs_3_int, 15, .21, .02, 1.05, .1)
heatmap_4_int = read_heatmap(ss_diffs_4_int, 15, .21, .02, 1.05, .1)

heatmap_int_tot = .2*(heatmap_0_int+heatmap_1_int+heatmap_2_int+heatmap_3_int+heatmap_4_int)

heatmap_tot = .2*(heatmap_0+heatmap_1+heatmap_2+heatmap_3+heatmap_4)

heatmap_acc = read_heatmap(ss_diffs_acc, 15, .21, .02, 1.05, .1)

heatmap_fig = plt.figure(figsize = (70,70))
plt.title(r"$R_{f}^{ss} - R_{0}^{ss}$", size = 22)
plt.xlabel(r"$m$", size = 22)
plt.ylabel(r"$\alpha$", size = 22)
plt.xticks(M_vals, fontsize = 20)
plt.yticks(coupling_vals, fontsize = 20)
plt.imshow(heatmap_tot, cmap = 'hot', interpolation = 'gaussian', extent=[1, 2.5, .2, .5], aspect=5)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)



ax = Axes3D(plt.figure(11))
coupling_vals = np.linspace(.5, .2, 16)
M_vals = np.linspace(1, 2.5, 16)
X, Y = np.meshgrid(coupling_vals, M_vals)
ax.plot_surface(X, Y, heatmap_tot, rstride = 1, cstride = 1, cmap = "hot")

plt.show()