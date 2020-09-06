import numpy as np
import GraphGenerator as gg
from colour import Color
from numpy import random as nrd
import random as rd
import copy
from copy import deepcopy
from random import choice
from array import *
import time
from matplotlib import colors
import os, glob
import small_world as sw
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import scipy.ndimage as sim
from mpl_toolkits.mplot3d import Axes3D
import MidpointNormalize as mn

trials_dir = "C:/Users/billy/PycharmProjects/Kuramoto2020/6400 trials/"
head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"

path_ER_sparse = head_dir + '/ER Graphs 500 edges/'
path_ER = head_dir + '/ER Graphs 1000 edges/'
path_ER_dense = head_dir + '/ER Graphs 2000 edges/'
path_SA = head_dir + '/Laplace-Optimized 1000 edges/'
path_SA_sparse = head_dir + '/Laplace-Optimized-500-edges/'
path_TA_OPs = head_dir + '/Time-Averaged OPs/ER to SA and back/100 Transition Graphs/'
path_final_states = head_dir + '/ICs/ER to SA to ER (1000, .3) final states/'
path_misaligned = head_dir + '/Synchrony Misaligned 1000 edges/'
path_modular = head_dir + '/Modular Graphs 500 edges (.9)/'
path_dense_mod = head_dir + '/Modular Graphs 1000 edges/'
path_freq_mod = head_dir + '/Frequency Modular 1000/'
path_sparsefreq_mod = head_dir + "/Frequency Modular 500/"
path_random_ICs = head_dir + "/ICs/random/"
path_random_nat_freqs = head_dir + "/ICs/rand_freqs/"

misaligned_files = []
IC_Files = []
TA_OP_Files = []
ER_Files = []
ER_Sparse_Files = []
ER_Dense_Files = []
SA_Files = []
SA_Sparse_Files = []
mod_Files = []
freq_mod_Files = []
sparsefreq_mod_Files = []
dense_mod_Files = []
ICs = []
ER_Graphs = []
ER_Dense_Graphs = []
ER_Sparse_Graphs = []
SA_Graphs = []
SA_Sparse_Graphs = []
MA_Graphs = []
mod_Graphs = []
freq_mod_Graphs = []
sparsefreq_mod_Graphs = []
dense_mod_Graphs = []

TA_OPs_Files_density = []
TA_OPs_density = []

TA_OPs_Files_SA = []
TA_OPs_SA = []

TA_OPs_Files_density_half = []
TA_OPs_density_half = []

for i in range(25):
    TA_OPs_Files_density.append(open("TA_OPs_randomized_density " + str(i) + ".txt", "r"))
    TA_OPs_density.append(gg.readMatrixFromFile(TA_OPs_Files_density[i])[0])
    TA_OPs_Files_density[i].close()

    TA_OPs_Files_SA.append(open("TA_OPs_ER_SA_ER " + str(i) + ".txt", "r"))
    TA_OPs_SA.append(gg.readMatrixFromFile(TA_OPs_Files_SA[i])[0])
    TA_OPs_Files_SA[i].close()

    TA_OPs_Files_density_half.append(open("TA OP Density half coupling "+str(i)+".txt", "r"))
    TA_OPs_density_half.append(gg.readMatrixFromFile(TA_OPs_Files_density_half[i])[0])
    TA_OPs_Files_density_half[i].close()

    ER_Files.append(open(os.path.join(path_ER + str(i) + ".txt"), "r"))
    ER_Dense_Files.append(open(os.path.join(path_ER_dense + str(i) + ".txt"), "r"))
    ER_Sparse_Files.append(open(os.path.join(path_ER_sparse + str(i) + ".txt"), "r"))
    misaligned_files.append(open(os.path.join(path_misaligned) + str(i) + ".txt", "r"))
    # mod_Files.append(open(os.path.join(path_modular) + str(i) + ".txt", "r"))
    dense_mod_Files.append(open(os.path.join(path_dense_mod) + str(i) + ".txt", "r"))
    SA_Files.append(open(os.path.join(path_SA + str(i) + ".txt"), "r"))
    SA_Sparse_Files.append(open(os.path.join(path_SA_sparse+str(i)+".txt"),"r"))
    freq_mod_Files.append(open(os.path.join(path_freq_mod + str(i) + ".txt"), "r"))

for i in range(25):
    ER_Sparse_Graphs.append(gg.readMatrixFromFile(ER_Sparse_Files[i]))
    SA_Sparse_Graphs.append(gg.readMatrixFromFile(SA_Sparse_Files[i]))
    ER_Graphs.append(gg.readMatrixFromFile(ER_Files[i]))
    SA_Graphs.append(gg.readMatrixFromFile((SA_Files[i])))
    ER_Dense_Graphs.append(gg.readMatrixFromFile(ER_Dense_Files[i]))
    dense_mod_Graphs.append(gg.readMatrixFromFile(dense_mod_Files[i]))

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
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
    first_half = deepcopy(TA_OPs[s_delay: 50 + s_delay + 1])
    second_half = deepcopy((TA_OPs[::-1])[e_delay: 50 + e_delay + 1])
    return np.trapz(second_half - first_half)

def get_partial_areas(TA_OPs, s_delay, e_delay):
    first_half = deepcopy(TA_OPs[s_delay: 50 + s_delay + 1])
    second_half = deepcopy((TA_OPs[::-1])[e_delay: 50 + e_delay + 1])
    difference = second_half - first_half
    return difference
def get_AList(start=None, end=None, final=None, dens_const=True, transitions=gg.numTransitionGraphs):
    if start is not None:
        gg.main(transitions, start=start, end=end, final=final, dens_const=dens_const)
    else:
        gg.main(transitions)
    f = open("adjacency matrices.txt", "r")
    AList = []
    ATemp = []
    for line in f:
        if not line.strip() and len(ATemp) == len(ATemp[0]):
            AList.append(ATemp)
            ATemp = []
            continue
        line = [float(i) for i in line.strip().split('\t')]
        ATemp.append(line)
    f.close()
    if final is None:
        # last_element = AList[len(AList)-1]
        # AList = AList[::5]
        # AList.append(last_element)
        AList = AList + list(reversed(AList))
    return np.array(AList), np.array(gg.freqs)

def get_laplacian(A):
    result = -deepcopy(A)
    for i in range(len(result[0])):
        result[i][i] += np.sum(A[i])
    return result

def compute_alg_connectivities(start, end, dens_constant = True):
    AList, freqs = get_AList(start = start, end = end, dens_const= dens_constant)
    alg_connectivities = np.zeros(len(AList), dtype = np.float64)
    for i in range(len(alg_connectivities)):
        evals= sp.linalg.eigh(np.array(get_laplacian(AList[i]), dtype = np.float64), eigvals=(1, len(AList[i]) - 1), eigvals_only = True)
        print(evals[-1])
        print("LAST EVAL")
        alg_connectivities[i] = evals[0]
    return alg_connectivities

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
            if all(abs(x - TA_data[16]) < .01 for x in TA_data[8:16]):
                if all(abs(x - TA_data[133]) < .01 for x in TA_data[125:]):
                    results[int(line[1])][row][col] += compute_area(TA_data, 16, 16)
                    counts[row][col] += 1
    return results, counts

def compute_ss_all(dir,trials, n, a_start, a_inc, M_start, M_inc):
    results_i = []
    results_f = []
    for i in range(trials):
        results_i.append(np.zeros((n, n)))
        results_f.append(np.zeros((n,n)))
    for filename in glob.glob(os.path.join(dir, '*.txt')):
        with open(filename, 'r') as f:
            line = [float(j) for j in (filename[62:])[:-5].strip().split(' ')]
            row = n - 1 - int(round((line[2] - a_start) / a_inc))
            col = int(round((line[3] - M_start) / M_inc))
            TA_data = gg.readMatrixFromFile(f)[0]
            results_i[int(line[1])][row][col] += TA_data[16]
            results_f[int(line[1])][row][col] += TA_data[133]
    return np.mean(results_i, axis = 0), np.mean(results_f, axis = 0)

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


ss_i, ss_f = compute_ss_all(trials_dir,25, 16, .2, .02, 1, .1)

results, counts = compute_area_all(trials_dir,25, 16, .2, .02, 1, .1)
avg_area = np.zeros((16,16))
for i in range(len(all_maps)):
    avg_area += results[i]
for r in range(16):
    for c in range(16):
        avg_area[r][c] /= counts[r][c]

heatmap_tot = .2*(heatmap_0+heatmap_1+heatmap_2+heatmap_3+heatmap_4)


heatmap_fig = plt.figure(figsize = (70, 70))
#plt.title("Synchrony Gap, "+r"$R_{f}^{ss} - R_{0}^{ss}$", size = 34)
plt.title(r"$R_{f}^{ss}$", size = 34)
#plt.title("Area", size = 34)
plt.xlabel("inertia, "+r"$m$", size = 34)
plt.ylabel("coupling, "+r"$\alpha$", size = 34)
plt.xticks(M_vals, fontsize = 30)
plt.yticks(coupling_vals, fontsize = 30)
norm = mpl.colors.Normalize(vmin=-1.,vmax=1.)
#plt.imshow(avg_area, cmap = 'RdBu_r', extent=[1, 2.5, .2, .5], norm = mn.MidpointNormalize(midpoint=0), interpolation = 'None', aspect=5)
plt.imshow(ss_f, cmap = 'hot', extent=[1, 2.5, .2, .5], interpolation = 'none', aspect=5)
cbar = plt.colorbar()
plt.gcf().subplots_adjust(bottom=0.13)
cbar.ax.tick_params(labelsize=30)
#plt.tight_layout()

coupling_vals = np.linspace(.2, .5, 16)
M_vals = np.linspace(1, 2.5, 16)
plt.figure()
plt.xlabel("coupling, "+r"$\alpha$", size = 30)
plt.ylabel(r"$R_{0}^{ss}$", size = 30)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
for i in range(0,16):
    plt.plot(coupling_vals, (ss_i[:,i])[::-1], label = r'$m = $' + str(M_vals[i])[0:3], lw = 0.5, color = colorFader('red', 'green', np.power(i / len(coupling_vals), .75)))
plt.legend()
plt.tight_layout()

plt.figure()
for i in range(0,15, 2):
    plt.plot(M_vals, (ss_i[i]), label = str(coupling_vals[i])[0:4])
plt.legend()
# reg_map_example = np.loadtxt("reg_pairwise_heatmap 6.txt")
# freq_map_example = np.loadtxt("freq_pairwise_heatmap 0.txt")
# plt.figure(9, figsize = (70,70))
# plt.title("Pairwise Synchrony, "+ r"$\langle R_{i,j} \rangle$",size = 30)
# plt.imshow(reg_map_example, vmin = 0, cmap = 'viridis')
# plt.xticks([], [])
# plt.yticks([], [])
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=30)
# plt.figure(10, figsize = (70,70))
# plt.title("Pairwise Synchrony, "+ r"$\langle R_{i,j} \rangle$",size = 30)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.imshow(freq_map_example, vmin = 0, cmap = 'viridis')
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=30)

plt.figure()
ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
diffs_avg = np.zeros(51)
alg_connectivities_avg = np.zeros(51)
all_diffs = []
all_alg_connectivites = []
for i in range(25):
    diffs = get_partial_areas(TA_OPs_SA[i],16,16)
    all_diffs.extend(list(diffs))
    if i % 10 == 0:
        plt.plot(TA_OPs_density_half[i][16:67], color = str(.05 * i))
        plt.plot((TA_OPs_density_half[i][::-1])[16:67], color = str(.05 * i))
    alg_connectivities = compute_alg_connectivities(ER_Graphs[i], SA_Graphs[i], dens_constant= True)
    diffs_avg += diffs
    alg_connectivities_avg += alg_connectivities[:51]
    print(alg_connectivities[:51])
    all_alg_connectivites.extend(list(alg_connectivities[:51]))
diffs_avg /= 25
alg_connectivities_avg /= 25
plt.figure()
plt.scatter(alg_connectivities_avg, diffs_avg, color = 'gray')
plt.plot(alg_connectivities_avg, diffs_avg, color = 'gray')
plt.xlabel("Algebraic Connectivity")
plt.ylabel("Order Parameter Gap")

plt.figure()
plt.scatter(all_alg_connectivites, all_diffs, color = 'gray', s= 15)
plt.xlabel("Algebraic Connectivity")
plt.ylabel("Order Parameter Gap")

plt.show()