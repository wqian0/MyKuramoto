import numpy as np
from numpy import random as nrd
import random as rd
import copy
from random import choice
from array import *
import time
import os, glob
import small_world as sw
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import GraphGenerator as gg
import MyKuramoto as mK

trials_dir = "C:/Users/billy/PycharmProjects/Kuramoto2020/6400 trials/"
TA_OPs_Files_density = []
TA_OPs_Files_density_half = []
TA_OPs_Files_SA = []
TA_OPs_Files_MA = []
TA_OPs_Files_SA_100k = []
TA_OPs_Files_OtherER = []
TA_OPs_Files_Mod = []
TA_OPs_Files_freqmod = []
TA_OPs_Files_density_massless = []
TA_OPs_Files_SA_massless = []

TA_OPs_Files_500 = []
TA_OPs_Files_250 = []
TA_OPs_Files_50 = []
TA_OPs_Files_100 = []

TA_OPs_Files_100_density = []
TA_OPs_Files_500_density = []

TA_OPs_density = []
TA_OPs_density_half = []
TA_OPs_SA = []
TA_OPs_MA = []
TA_OPs_SA_100k = []
TA_OPs_OtherER = []
TA_OPs_Mod = []
TA_OPs_freqmod = []

TA_OPs_500 = []
TA_OPs_250 = []
TA_OPs_100 = []
TA_OPs_50 = []

TA_OPs_100_density = []
TA_OPs_500_density = []

TA_OPs_density_massless = []
TA_OPs_SA_massless = []


for i in range(25):
    TA_OPs_Files_density.append(open("TA_OPs_randomized_density "+str(i)+".txt", "r"))
    TA_OPs_Files_density_half.append(open("TA OP Density half coupling "+str(i)+".txt", "r"))
    TA_OPs_Files_SA.append(open("TA_OPs_ER_SA_ER " + str(i) + ".txt", "r"))
    TA_OPs_Files_OtherER.append(open("TA_OPs_Strong_ICs_ER "+str(i)+".txt", "r"))
    TA_OPs_Files_Mod.append(open("reg_cluster_OP " + str(i) + ".txt", "r"))
    TA_OPs_Files_SA_100k.append(open("TA_OPs_ER_SA_ER_100k "+str(i)+".txt", "r"))
    #TA_OPs_Files_MA.append(open("TA_OPs_Strong_ICs_MA "+str(i)+".txt", "r"))
    TA_OPs_Files_MA.append(open("TA_OPs_randomized_MA " + str(i) + ".txt", "r"))
    # TA_OPs_Files_freqmod.append(open("TA_OPs_freq_mod "+str(i)+".txt", "r"))
    TA_OPs_Files_freqmod.append(open("freq_cluster_OP " + str(i) + ".txt", "r"))
    TA_OPs_Files_500.append(open("Graph_resolution_500"+str(i)+".txt", "r"))
    TA_OPs_Files_250.append(open("Graph_resolution_250" + str(i) + ".txt", "r"))
    TA_OPs_Files_50.append(open("Graph_resolution_50" + str(i) + ".txt", "r"))
    TA_OPs_Files_100.append(open("Graph_resolution_100" + str(i) + ".txt", "r"))

    TA_OPs_Files_100_density.append(open("100_transitions_density "+str(i)+".txt", "r"))
    TA_OPs_Files_500_density.append(open("500_transitions_density "+str(i)+".txt", "r"))

    TA_OPs_Files_density_massless.append(open("density_massless_2020_35 "+str(i)+".txt", "r"))
    TA_OPs_Files_SA_massless.append(open("ER_SA_ER_massless_2020_169 "+str(i)+".txt", "r"))



for i in range(25):
    TA_OPs_density.append(gg.readMatrixFromFile(TA_OPs_Files_density[i])[0])
    TA_OPs_density_half.append(gg.readMatrixFromFile(TA_OPs_Files_density_half[i])[0])
    TA_OPs_SA.append(gg.readMatrixFromFile(TA_OPs_Files_SA[i])[0])
    TA_OPs_OtherER.append(gg.readMatrixFromFile(TA_OPs_Files_OtherER[i])[0])
    TA_OPs_Mod.append((gg.readMatrixFromFile(TA_OPs_Files_Mod[i]))[0])
    TA_OPs_SA_100k.append(gg.readMatrixFromFile(TA_OPs_Files_SA_100k[i])[0])
    TA_OPs_MA.append(gg.readMatrixFromFile(TA_OPs_Files_MA[i])[0])
    if i is 5:
        TA_OPs_freqmod.append(gg.readMatrixFromFile(TA_OPs_Files_freqmod[i]))
    else:
        TA_OPs_freqmod.append(gg.readMatrixFromFile(TA_OPs_Files_freqmod[i])[0])
    TA_OPs_500.append(gg.readMatrixFromFile(TA_OPs_Files_500[i])[0])
    TA_OPs_250.append(gg.readMatrixFromFile(TA_OPs_Files_250[i])[0])
    TA_OPs_50.append(gg.readMatrixFromFile(TA_OPs_Files_50[i])[0])
    TA_OPs_100.append(gg.readMatrixFromFile(TA_OPs_Files_100[i])[0])

    TA_OPs_100_density.append(gg.readMatrixFromFile(TA_OPs_Files_100_density[i])[0])
    TA_OPs_500_density.append(gg.readMatrixFromFile(TA_OPs_Files_500_density[i])[0])

    TA_OPs_density_massless.append(gg.readMatrixFromFile(TA_OPs_Files_density_massless[i])[0])
    TA_OPs_SA_massless.append(gg.readMatrixFromFile(TA_OPs_Files_SA_massless[i])[0])

for i in range(25):
    TA_OPs_Files_density[i].close()
    TA_OPs_Files_density_half[i].close()
    TA_OPs_Files_SA[i].close()
    TA_OPs_Files_OtherER[i].close()
    TA_OPs_Files_Mod[i].close()
    TA_OPs_Files_MA[i].close()
    TA_OPs_Files_freqmod[i].close()
    TA_OPs_Files_500[i].close()
    TA_OPs_Files_250[i].close()
    TA_OPs_Files_50[i].close()
    TA_OPs_Files_100[i].close()

    TA_OPs_Files_100_density[i].close()
    TA_OPs_Files_500_density[i].close()

    TA_OPs_Files_density_massless[i].close()
    TA_OPs_Files_SA_massless[i].close()

TA_OPs_density_avg = np.zeros(len(TA_OPs_density[0]))
TA_OPs_density_half_avg = np.zeros(len(TA_OPs_density_half[0]))
TA_OPs_SA_avg = np.zeros(len(TA_OPs_SA[0]))
TA_OPs_OtherER_avg = np.zeros(len(TA_OPs_OtherER[0]))
TA_OPs_Mod_avg = np.zeros(len(TA_OPs_Mod[0]))
TA_OPs_SA_100k_avg = np.zeros(len(TA_OPs_SA_100k[0]))
TA_OPs_MA_avg = np.zeros(len(TA_OPs_MA[0]))
TA_OPs_freqmod_avg = np.zeros(len(TA_OPs_freqmod[0]))

TA_OPs_500_avg = np.zeros(len(TA_OPs_500[0]))
TA_OPs_250_avg = np.zeros(len(TA_OPs_250[0]))
TA_OPs_50_avg = np.zeros(len(TA_OPs_50[0]))
TA_OPs_100_avg = np.zeros(len(TA_OPs_100[0]))

TA_OPs_100_density_avg = np.zeros(len(TA_OPs_100_density[0]))
TA_OPs_500_density_avg = np.zeros(len(TA_OPs_500_density[0]))


TA_OPs_density_massless_avg = np.zeros(len(TA_OPs_density_massless[0]))
TA_OPs_SA_massless_avg = np.zeros(len(TA_OPs_SA_massless[0]))

density_count = 0
freq_count = 0
for i in range(25):
    TA_OPs_density_avg += np.array(TA_OPs_density[i])
    if all(abs(x - TA_OPs_density_half[i][16]) < .01 for x in TA_OPs_density_half[i][12:16]):
        TA_OPs_density_half_avg += np.array(TA_OPs_density_half[i])
        density_count += 1
    TA_OPs_SA_avg += np.array(TA_OPs_SA[i])
    TA_OPs_OtherER_avg += np.array(TA_OPs_OtherER[i])
    TA_OPs_Mod_avg += np.array(TA_OPs_Mod[i])
    TA_OPs_SA_100k_avg += np.array(TA_OPs_SA_100k[i])
    TA_OPs_MA_avg += np.array(TA_OPs_MA[i])
    if i is not 5:
        print(np.array(TA_OPs_freqmod[i]))
        print(TA_OPs_freqmod_avg)
        if all(abs(x - TA_OPs_freqmod[i][133]) < .005 for x in TA_OPs_freqmod[i][125:]):
            TA_OPs_freqmod_avg += np.array(TA_OPs_freqmod[i])
            freq_count += 1

    TA_OPs_500_avg += np.array(TA_OPs_500[i])
    TA_OPs_250_avg += np.array(TA_OPs_250[i])
    TA_OPs_50_avg += np.array(TA_OPs_50[i])
    TA_OPs_100_avg += np.array(TA_OPs_100[i])

    TA_OPs_100_density_avg += np.array(TA_OPs_100_density[i])
    TA_OPs_500_density_avg += np.array(TA_OPs_500_density[i])

    TA_OPs_density_massless_avg += np.array(TA_OPs_density_massless[i])
    TA_OPs_SA_massless_avg += np.array(TA_OPs_SA_massless[i])
print(density_count)
TA_OPs_density_avg/= 25
TA_OPs_density_half_avg /= density_count
TA_OPs_SA_avg/= 25
TA_OPs_Mod_avg /= 25
TA_OPs_OtherER_avg /= 25
TA_OPs_SA_100k_avg /= 25
TA_OPs_MA_avg /= 25
TA_OPs_freqmod_avg /= freq_count
print(freq_count)
print("monkaS")

TA_OPs_500_avg /= 25
TA_OPs_250_avg /= 25
TA_OPs_50_avg /= 25
TA_OPs_100_avg /= 25

TA_OPs_500_density_avg /= 25
TA_OPs_100_density_avg /= 25

TA_OPs_density_massless_avg /= 25
TA_OPs_SA_massless_avg /= 25

#correcting first slot
TA_OPs_Mod_avg[0] = TA_OPs_Mod_avg[1]
TA_OPs_freqmod_avg[0] = TA_OPs_freqmod_avg[1]



def read_all_data(dir, trials, n, a_start, a_inc, M_start, M_inc):
    counts = np.zeros((n, n))
    TA_OPs = np.zeros((16, 16, 134))
    for filename in glob.glob(os.path.join(dir, '*.txt')):
        with open(filename, 'r') as f:
            line = [float(j) for j in (filename[62:])[:-5].strip().split(' ')]
            row = n - 1 - int(round((line[2] - a_start) / a_inc))
            col = int(round((line[3] - M_start) / M_inc))
            TA_data = gg.readMatrixFromFile(f)[0]
            if all(abs(x - TA_data[16]) < .01 for x in TA_data[8:16]):
                if all(abs(x - TA_data[133]) < .01 for x in TA_data[125:]):
                    TA_OPs[row, col, :] += TA_data
                    counts[row][col] += 1
    return TA_OPs, counts

def fancy_plot(fignum, size, delay, data, markersize, linewidth = .5, alpha = 1, hline = False, hline_loc = 0.5, filename = None, color = 'black', m =0, a = 0):
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(fignum, dpi = 3000)
    ##CHANGE DPI TO 3000 WHEN SAVING PLOT
    plt.xlabel('Graph Index', size=20)
    #plt.ylabel('Module Order Parameter, ' + r'$\langle R_M   \rangle$', size=16)
    plt.ylabel('Order Parameter, ' + r'$\langle R \rangle$', size=20)
    plt.tight_layout()
    # plt.title(r'$m = $'+str(m)+', '+r'$\alpha = $'+str(a))
    axes = plt.gca()
    axes.set_xlim([-1, size + delay + 1])
    axes.set_ylim([0, 1])
    plt.rcParams.update({'font.size': 16})
    plt.axvline(x=delay, color='red', linewidth=1)

    tick_locs = np.arange(delay, delay + size + size // 5, size // 5)
    labels = np.arange(0, size + size // 5, size // 5)
    plt.xticks(tick_locs, labels=labels)

    plt.scatter(list(range(len(data) // 2)), data[0: (len(data) // 2)], color=color,
                marker='>', facecolors='none', s=markersize, lw=linewidth, edgecolors=color, alpha = alpha)
    plt.scatter(list(range(len(data) // 2)), list(reversed(data[len(data) // 2:])),
                color=color, marker='<', s=markersize, lw=linewidth, edgecolors='black', alpha = 1)
    if filename is not None:
        plt.savefig(filename +'.pdf')
        #plt.savefig(filename + '.png')
    if hline:
        plt.axhline(y=hline_loc, color='green', linestyle='dashed')
    plt.close()

def fancy_NoOffset(fignum, inputSize, data, markerSize, linewidth = .5, alpha = 1, hline = False, hline_loc = 0.5, filename = None):
    plt.rcParams.update({'font.size': 16})
    f4 = plt.figure(fignum, dpi = 3000)
    #CHANGE DPI to 3000
    plt.xlabel('Graph Index', size=20)
    plt.ylabel('Order Parameter, ' + r'$\langle R \rangle$', size=20)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    # plt.title("Averaged Data changing Density")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.set_xlim([-1, inputSize + 1])
    plt.rcParams.update({'font.size': 16})
    plt.scatter(list(range(len(data) // 2)), data[0: (len(data) // 2)],
                color='black',
                marker='>', facecolors='none', s=markerSize, lw=linewidth, alpha = alpha)
    plt.scatter(list(range(len(data) // 2)),
                list(reversed(data[len(data) // 2:])),
                color='black', marker='<', s=markerSize, lw=linewidth, edgecolors='black')
    if filename is not None:
        plt.savefig(filename +'.pdf')
        #plt.savefig(filename + '.png')

'''
density_std = np.zeros(len(TA_OPs_density[0]))
for i in range(len(density_std)):
    density_std[i] = np.std([TA_OPs_density[j][i] for j in range(len(TA_OPs_density))])

SA_std = np.zeros(len(TA_OPs_SA[0]))
for i in range(len(density_std)):
    SA_std[i] = np.std([TA_OPs_SA[j][i] for j in range(len(TA_OPs_SA))])

f4 = plt.figure(4)
plt.xlabel('Graph Index')
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$')
plt.tight_layout()
#plt.title("Averaged Data changing Density")
axes = plt.gca()
axes.set_ylim([0, 1])
axes.set_xlim([-1, 51])
for i in range(0,25,25):
    plt.scatter(list(range(len(TA_OPs_density_avg) // 2)), TA_OPs_density[15][0: (len(TA_OPs_density_avg) // 2)], color='black',
                marker='>', facecolors='none', s=25, lw=.5)
# plt.fill_between(list(range(len(TA_OPs_density_avg) // 2)), TA_OPs_density_avg[0: (len(TA_OPs_density_avg) // 2)]- density_std[0: (len(TA_OPs_density_avg) // 2)],
#                  TA_OPs_density_avg[0: (len(TA_OPs_density_avg) // 2)] + density_std[0: (len(TA_OPs_density_avg) // 2)], color = "blue", alpha =.1)
    plt.scatter(list(range(len(TA_OPs_density_avg) // 2)), list(reversed(TA_OPs_density[7][len(TA_OPs_density_avg) // 2:])),
                    color='black', marker='<', s=25, lw=.5, edgecolors='black')
# plt.fill_between(list(range(len(TA_OPs_density_avg) // 2)), np.array(list(reversed(TA_OPs_density_avg[len(TA_OPs_density_avg) // 2:])))- np.array(list(reversed(density_std[(len(TA_OPs_density_avg) // 2):]))),
#                  np.array(list(reversed(TA_OPs_density_avg[len(TA_OPs_density_avg) // 2:]))) + np.array(list(
#                      reversed(density_std[(len(TA_OPs_density_avg) // 2):]))), color = "red", alpha =.1)

f5 = plt.figure(5)
plt.xlabel('Graph Index')
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$')
plt.tight_layout()
#plt.title("Averaged Data Constant Density")
axes = plt.gca()
axes.set_xlim([-1,67])
axes.set_ylim([0, 1])


plt.axvline(x=gg.numTransitionGraphs // 3, color='red', linewidth=1)

tick_locs = np.arange(16,76, 10)
labels = np.arange(0, 60, 10)
plt.xticks(tick_locs, labels = labels)

for i in range(0,25,25):
    plt.scatter(list(range(len(TA_OPs_SA_avg) // 2)), TA_OPs_SA[17][0: (len(TA_OPs_SA_avg) // 2)], color='black',
            marker='>', facecolors='none', s=25, lw=.5)
# plt.fill_between(list(range(len(TA_OPs_SA_avg) // 2)), TA_OPs_SA_avg[0: (len(TA_OPs_SA_avg) // 2)]- SA_std[0: (len(TA_OPs_SA_avg) // 2)],
#                 TA_OPs_SA_avg[0: (len(TA_OPs_SA_avg) // 2)] + SA_std[0: (len(TA_OPs_SA_avg) // 2)], color = "blue", alpha =.1)
    plt.scatter(list(range(len(TA_OPs_SA_avg) // 2)), list(reversed(TA_OPs_SA[4][len(TA_OPs_SA_avg) // 2:])),
            color='black', marker='<', s=25, lw=.5, edgecolors='black')
# plt.fill_between(list(range(len(TA_OPs_SA_avg) // 2)), np.array(list(reversed(TA_OPs_SA_avg[len(TA_OPs_SA_avg) // 2:])))- np.array(list(reversed(SA_std[(len(TA_OPs_SA_avg) // 2):]))),
#                 np.array(list(reversed(TA_OPs_SA_avg[len(TA_OPs_SA_avg) // 2:]))) + np.array(list(
#                    reversed(SA_std[(len(TA_OPs_SA_avg) // 2):]))), color = "red", alpha =.1)
'''
'''
plt.rcParams.update({'font.size': 16})
f6 = plt.figure(6, dpi = 3000)
plt.xlabel('Graph Index', size = 20)
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$', size = 20)
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
#plt.title("Averaged Data Other ER")
axes = plt.gca()
axes.set_xlim([-1,67])
axes.set_ylim([0, 1])
plt.rcParams.update({'font.size': 16})
tick_locs = np.arange(16,76, 10)
labels = np.arange(0, 60, 10)
plt.xticks(tick_locs, labels = labels)
plt.rcParams.update({'font.size': 16})
plt.scatter(list(range(len(TA_OPs_OtherER_avg) // 2)), TA_OPs_OtherER_avg[0: (len(TA_OPs_OtherER_avg) // 2)], color='orange',
                marker='>', facecolors='none', s=25, lw=.5, edgecolors='orange')
plt.scatter(list(range(len(TA_OPs_OtherER_avg) // 2)), list(reversed(TA_OPs_OtherER_avg[len(TA_OPs_OtherER_avg) // 2:])),
                color='orange', marker='<', s=25, lw=.5, edgecolors='black')
plt.axvline(x=gg.numTransitionGraphs // 3, color='red', linewidth=1)
plt.rcParams.update({'font.size': 16})
#MA
plt.scatter(list(range(len(TA_OPs_MA_avg) // 2)), TA_OPs_MA_avg[0: (len(TA_OPs_MA_avg) // 2)], color='blue',
                marker='>', facecolors='none', s=25, lw=.5, edgecolors='blue')
plt.scatter(list(range(len(TA_OPs_MA_avg) // 2)), list(reversed(TA_OPs_MA_avg[len(TA_OPs_MA_avg) // 2:])),
                color='blue', marker='<', s=25, lw=.5, edgecolors='black')
plt.rcParams.update({'font.size': 16})


legend_elements = [Patch(facecolor='orange', edgecolor= 'black', label = 'Other ER', lw = .5), Patch(facecolor='blue', edgecolor='black', label = 'Synchrony-Misaligned', lw = .5)]
plt.legend(handles=  legend_elements, frameon = False, loc = 'lower right')
plt.axhline(y=TA_OPs_SA_avg[15], color='green', linestyle="--", dashes = (5, 5), linewidth = 1)
plt.savefig("MA2.pdf")
plt.savefig("MA.png")


f7 = plt.figure(7)
plt.xlabel('Graph')
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$', size = 20)
plt.rcParams.update({'font.size': 16})
plt.title("Averaged Data Modular", size = 20)
axes = plt.gca()
axes.set_ylim([0, 1])

plt.scatter(list(range(len(TA_OPs_Mod_avg) // 2)), TA_OPs_Mod_avg[0: (len(TA_OPs_Mod_avg) // 2)], color='blue',
                marker='>', facecolors='none', s=25, lw=.5)
plt.scatter(list(range(len(TA_OPs_Mod_avg) // 2)), list(reversed(TA_OPs_Mod_avg[len(TA_OPs_Mod_avg) // 2:])),
                color='red', marker='<', s=25, lw=.5, edgecolors='black')
plt.axvline(x=mK.startDelay, color='red', linewidth=1)


f9 = plt.figure(9)
plt.xlabel('Graph Index')
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$')
plt.title("Averaged Data MA")
axes = plt.gca()
axes.set_ylim([0, 1])

plt.scatter(list(range(len(TA_OPs_MA_avg) // 2)), TA_OPs_MA_avg[0: (len(TA_OPs_MA_avg) // 2)], color='blue',
                marker='>', facecolors='none', s=25, lw=.5)
plt.scatter(list(range(len(TA_OPs_MA_avg) // 2)), list(reversed(TA_OPs_MA_avg[len(TA_OPs_MA_avg) // 2:])),
                color='red', marker='<', s=25, lw=.5, edgecolors='black')
plt.axvline(x=mK.startDelay, color='red', linewidth=1)

f10 = plt.figure(10, dpi = 3000)
plt.xlabel('Graph Index', size = 20)
plt.rcParams.update({'font.size': 16})
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$', size = 20)
plt.tight_layout()
#plt.title("Modular")
axes = plt.gca()
axes.set_ylim([0, 1])
axes.set_xlim([-1,67])
tick_locs = np.arange(16,76, 10)
labels = np.arange(0, 60, 10)
plt.xticks(tick_locs, labels = labels)

plt.scatter(list(range(len(TA_OPs_freqmod_avg) // 2)), TA_OPs_freqmod_avg[0: (len(TA_OPs_freqmod_avg) // 2)], color='blue',
                marker='>', facecolors='none', s=25, lw=.5)
plt.scatter(list(range(len(TA_OPs_freqmod_avg) // 2)), list(reversed(TA_OPs_freqmod_avg[len(TA_OPs_freqmod_avg) // 2:])),
                color='blue', marker='<', s=25, lw=.5, edgecolors='black')
plt.axvline(x=gg.numTransitionGraphs // 3, color='red', linewidth=1)

#regmod
plt.scatter(list(range(len(TA_OPs_Mod_avg) // 2)), TA_OPs_Mod_avg[0: (len(TA_OPs_Mod_avg) // 2)], color='orange',
                marker='>', facecolors='none', s=25, lw=.5)
plt.scatter(list(range(len(TA_OPs_Mod_avg) // 2)), list(reversed(TA_OPs_Mod_avg[len(TA_OPs_Mod_avg) // 2:])),
                color='orange', marker='<', s=25, lw=.5, edgecolors='black')
legend_elements = [Patch(facecolor='orange', edgecolor= 'black', label = 'Modular', lw = .5), Patch(facecolor='blue', edgecolor='black', label = 'Frequency Modular', lw= .5)]
plt.legend(handles=  legend_elements, frameon = False, loc = 'lower right')

plt.axhline(y=TA_OPs_SA_avg[15], color='green', linestyle="--", dashes = (5, 5), linewidth = 1)
plt.savefig('modular2.pdf')
plt.savefig('modular.png')
#plt.show()

f11 = plt.figure(11)
plt.xlabel('Graph Index')
plt.ylabel('Order Parameter, '+r'$\langle R \rangle$')
plt.title("regmod")
axes = plt.gca()
axes.set_ylim([0, 1])
plt.scatter(list(range(len(TA_OPs_Mod_avg) // 2)), TA_OPs_Mod_avg[0: (len(TA_OPs_Mod_avg) // 2)], color='blue',
                marker='>', facecolors='none', s=25, lw=.5)
plt.scatter(list(range(len(TA_OPs_Mod_avg) // 2)), list(reversed(TA_OPs_Mod_avg[len(TA_OPs_Mod_avg) // 2:])),
                color='red', marker='<', s=25, lw=.5, edgecolors='black')
plt.axvline(x=mK.startDelay, color='red', linewidth=1)
#plt.show()

'''

#fancy_plot(17, 250, 83, TA_OPs_250_avg, 15)
# fancy_plot(18, 500, 166, TA_OPs_500_avg, 10, linewidth = .005, alpha = .2)
# fancy_plot(19, 100, 33, TA_OPs_100_avg, 20, linewidth = .4)
#fancy_NoOffset(20, 50, TA_OPs_density[15], filename = "TA_OPs_density_individual")
#fancy_plot(21, 50, 16, TA_OPs_density_half_avg, 25, filename = "density_half_coupling")

# TA_OPs, counts = read_all_data(trials_dir, 25, 16, .2, .02, 1, .1)
# print(TA_OPs.shape)
# print(counts)
# for r in range(16):
#     for c in range(16):
#         TA_OPs[r][c] /= counts[r][c]
# for r in range(16):
#     for c in range(16):
#         fancy_plot(16 * r + c, 50, 16, TA_OPs[r][c],25, filename = str(r)+"-"+str(c)+"-with_title", m = round(c * .1 +1, 1), a = round(r * .02 + .2, 1) )
#         print(counts[r][c])
# print(counts[13][8])


# fancy_NoOffset(20, 101, TA_OPs_100_density_avg, 20, linewidth = .4, filename = "100_density_averaged")
# fancy_NoOffset(21, 500, TA_OPs_500_density_avg, 10, linewidth = .005, alpha = .2, filename = "500_density_averaged")

TA_OPs, counts = read_all_data(trials_dir, 25, 16, .2, .02, 1, .1)
print(TA_OPs[10][10])
#fancy_plot(21, 50, 16, TA_OPs[10][10]/ counts[10][10], 25, filename= "ER_SA_ER_reduced_range.pdf", color = "black")

fancy_plot(21, 50, 16, TA_OPs_SA_massless_avg, 25, filename= "massless_SA_169.pdf", color = "black")
plt.show()



