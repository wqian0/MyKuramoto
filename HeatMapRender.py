import numpy as np
import GraphGenerator as gg
from numpy import random as nrd
import random as rd
import copy
from copy import deepcopy
from random import choice
from array import *
import time
import os, glob
import small_world as sw
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.ndimage as sim
from mpl_toolkits.mplot3d import Axes3D

trials_dir = "C:/Users/billy/PycharmProjects/Kuramoto2020/6400 trials/"
def read_heatmap(file, n, a_start, a_inc, M_start, M_inc):
    result = np.zeros((n, n))
    for line in file:
        if not line.strip():
            continue
        line = [float(j) for j in line.strip().split('\t')]
        result[n-1-int(round((line[1]-a_start)/a_inc))][int(round((line[2]-M_start)/M_inc))] = line[3]
    return result

def read_many_maps(file, trials, n, a_start, a_inc, M_start, M_inc):
    results = []
    counts = np.zeros((n,n))
    for i in range(trials):
        results.append(np.zeros((n,n)))
    for line in file:
        if not line.strip():
            continue
        line = [float(j) for j in line.strip().split('\t')]
        row = n-1-int(round((line[1]-a_start)/a_inc))
        col = int(round((line[2]-M_start)/M_inc))
        results[int(line[0])][row][col] = line[3]
        counts[row][col] += 1
    return results, counts

def compute_area(TA_OPs, s_delay, e_delay):
    first_half = deepcopy(TA_OPs[s_delay + 1: 67])
    second_half = deepcopy((TA_OPs[::-1])[e_delay + 1: 67])
    return np.trapz(second_half - first_half)
def compute_area_all(dir,trials, n, a_start, a_inc, M_start, M_inc):
    results = []
    counts = np.zeros((n, n))
    for i in range(trials):
        results.append(np.zeros((n, n)))
    for filename in glob.glob(os.path.join(dir, '*.txt')):
        with open(filename, 'r') as f:
            line = [float(j) for j in (filename[62:])[:-5].strip().split(' ')]
            row = n - 1 - int(round((line[2] - a_start) / a_inc))
            col = int(round((line[3] - M_start) / M_inc))
            TA_data = gg.readMatrixFromFile(f)[0]
            results[int(line[1])][row][col] += compute_area(TA_data, 16, 16)
            counts[row][col] += 1
    return results, counts

coupling_vals = np.linspace(.2, .5, 6)
M_vals = np.linspace(1, 2.5, 6)


ss_diffs_0 = open("ss_diffs_full.txt", "r")
ss_diffs_1 = open("ss_diffs_1.txt", "r")
ss_diffs_2 = open("ss_diffs_2.txt", "r")
ss_diffs_3 = open("ss_diffs_3.txt", "r")
ss_diffs_4 = open("ss_diffs_4.txt", "r")

heatmap_0 = read_heatmap(ss_diffs_0, 16, .2, .02, 1, .1)
heatmap_1 = read_heatmap(ss_diffs_1, 16, .2, .02, 1, .1)
heatmap_2 = read_heatmap(ss_diffs_2, 16, .2, .02, 1, .1)
heatmap_3 = read_heatmap(ss_diffs_3, 16, .2, .02, 1, .1)
heatmap_4 = read_heatmap(ss_diffs_4, 16, .2, .02, 1, .1)

ss_diffs_2020 = open("ss_diffs_2020.txt", "r")
all_maps, counts = read_many_maps(ss_diffs_2020, 25, 16, .2, .02, 1, .1)

avg = np.zeros((16, 16))
for i in range(len(all_maps)):
    avg += all_maps[i]

for r in range(16):
    for c in range(16):
        avg[r][c] /= counts[r][c]


results, counts = compute_area_all(trials_dir,25, 16, .2, .02, 1, .1)
avg_area = np.zeros((16,16))
for i in range(len(all_maps)):
    avg_area += results[i]
for r in range(16):
    for c in range(16):
        avg_area[r][c] /= counts[r][c]

heatmap_tot = .2*(heatmap_0+heatmap_1+heatmap_2+heatmap_3+heatmap_4)


heatmap_fig = plt.figure(figsize = (70,70))
plt.title(r"$R_{f}^{ss} - R_{0}^{ss}$", size = 22)
plt.xlabel(r"$m$", size = 22)
plt.ylabel(r"$\alpha$", size = 22)
plt.xticks(M_vals, fontsize = 20)
plt.yticks(coupling_vals, fontsize = 20)
plt.imshow(avg, cmap = 'hot', extent=[1, 2.5, .2, .5], aspect=5)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)



ax = Axes3D(plt.figure(11))
coupling_vals = np.linspace(.5, .2, 16)
M_vals = np.linspace(1, 2.5, 16)
X, Y = np.meshgrid(coupling_vals, M_vals)
ax.plot_surface(X, Y, avg, rstride = 1, cstride = 1, cmap = "hot")

plt.show()