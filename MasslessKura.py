import numpy as np
from scipy.integrate import ode, odeint
import numpy as np
import os
import cmath
import pylab as plt
import GraphGenerator as gg
import matplotlib.pyplot as plt
import random as rd
import sys

head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"
#head_dir = "/data/jux/bqqian/Kuramoto"

N, alpha, dt, frequencyBound, steps = gg.size, .05, .02, gg.freqBound, 50000
startDelay, endDelay = gg.numTransitionGraphs // 3,gg.numTransitionGraphs // 3

oParameterData, standardOPData, averagedOPData, inst_freqs, inst_phases = [], [], [], [], []


def getRandomDistribution(N, lowerBound, upperBound, distribution):
    list = []
    for r in range(N):
        list.append(distribution(lowerBound,upperBound))
    return list

def Kura(init, t, A, w_nat, a):
    theta = np.array(init)
    delta_theta = np.subtract.outer(theta, theta)
    dot_theta = w_nat + a * np.einsum('ij,ji->i', A, np.sin(delta_theta))
    return dot_theta


def runRK(A, phases0, w_nat, time):
    result = odeint(Kura, phases0, time, args=(A, w_nat, alpha))
    for t in range(len(result) - 1):
        standardOPData.append(abs(complex_OP2(result[t])))
        #inst_phases.append(result[t])
        #inst_freqs.append((result[t+1]-result[t])/dt)
    return result


def complex_OP2(theta):
    return (1.0 / N) * sum(cmath.exp(complex(0, a)) for a in theta)

def orderParameter(theta, t):
    return (1.0/N)*abs(sum(np.e ** complex(0, a) for a in theta[:, t]))


def averagedOP(start, end, OPData):
    return np.sum(OPData[start:end]) / (end - start)

def runSim(AMList, phases, frequencies):
    global oParameterData
    global standardOPData
    global localOPData
    oParameterData = []
    standardOPData = []
    localOPData = []
    time = np.linspace(0, dt * steps, steps)
    theta_0 = runRK(AMList[0], phases, frequencies, time)
    averagedOPData.append(averagedOP(len(standardOPData)-steps + 1, len(standardOPData), standardOPData))
    endStateTheta = theta_0[len(theta_0) - 1, :]
    for i in range(startDelay):
        theta = runRK(AMList[0], endStateTheta, frequencies, time)
        endStateTheta = theta[len(theta) - 1, :]
        print(i)
        averagedOPData.append(averagedOP(len(standardOPData) - steps + 1, len(standardOPData), standardOPData))
    for i in range(1, len(AMList)):
        theta = runRK(AMList[i], endStateTheta, frequencies, time)
        endStateTheta = theta[len(theta) - 1, :]
        print(i)
        averagedOPData.append(averagedOP(len(standardOPData) -steps + 1, len(standardOPData), standardOPData))
    for i in range(endDelay):
        theta = runRK(AMList[len(AMList)-1], endStateTheta, frequencies, time)
        endStateTheta = theta[len(theta) - 1, :]
        print(i)
        averagedOPData.append(averagedOP(len(standardOPData) - steps + 1, len(standardOPData), standardOPData))

def get_AList(start=None, end=None, final=None, dens_const=True, transitions = gg.numTransitionGraphs):
    if start is not None:
        gg.main(transitions,start=start, end=end, final=final, dens_const=dens_const)
    else:
        gg.main(gg.numTransitionGraphs)
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
        AList = AList + list(reversed(AList))
    print(str(len(AList)))
    return np.array(AList), gg.freqs

def get_ICs(f=None, removeMean=False):
    if f is not None:
        ICs = gg.readMatrixFromFile(f)
        if removeMean:
            return ICs[0] - np.mean(ICs[0])
        else:
            return ICs[0]
    else:
        init_phases = np.array(getRandomDistribution(N, -np.pi, np.pi, rd.uniform))
        init_freqs = np.array(gg.getRandomDistribution(N, -gg.freqBound, gg.freqBound))
        if removeMean:
            init_phases -= np.mean(init_phases)
            init_freqs -= np.mean(init_freqs)
        return init_phases

OPs, ta_OPs, ss_diffs, end_states = [], [], [], []

path_ER_sparse = head_dir+'/ER Graphs 500 edges/'
path_ER = head_dir+'/ER Graphs 1000 edges/'
path_ER_dense = head_dir+'/ER Graphs 2000 edges/'
path_SA = head_dir+'/Laplace-Optimized 1000 edges/'
path_SA_sparse = head_dir+'/Laplace-Optimized-500-edges/'
path_TA_OPs = head_dir+'/Time-Averaged OPs/ER to SA and back/100 Transition Graphs/'
path_final_states = head_dir+'/ICs/ER to SA to ER (1000, .3) final states/'
path_misaligned = head_dir+'/Synchrony Misaligned 1000 edges/'
path_modular = head_dir+'/Modular Graphs 500 edges (.9)/'
path_dense_mod = head_dir+'/Modular Graphs 1000 edges/'
path_freq_mod = head_dir+'/Frequency Modular 1000/'
path_sparsefreq_mod = head_dir+"/Frequency Modular 500/"
path_random_ICs = head_dir+"/ICs/random/"
path_random_nat_freqs = head_dir+"/ICs/rand_freqs/"

misaligned_files, IC_Files, TA_OP_Files, ER_Files, ER_Sparse_Files, ER_Dense_Files = [], [], [], [], [], []

SA_Files, SA_Sparse_Files, mod_Files, freq_mod_Files, sparsefreq_mod_Files, dense_mod_Files = [], [], [], [], [], []

ICs, ER_Graphs, ER_Dense_Graphs, ER_Sparse_Graphs, SA_Graphs, SA_Sparse_Graphs, = [], [], [], [] ,[], []

MA_Graphs, mod_Graphs, freq_mod_Graphs, sparsefreq_mod_Graphs, dense_mod_Graphs = [], [], [], [], []

for i in range(25):
    ER_Files.append(open(os.path.join(path_ER + str(i) + ".txt"), "r"))
    ER_Dense_Files.append(open(os.path.join(path_ER_dense + str(i) + ".txt"), "r"))
    ER_Sparse_Files.append(open(os.path.join(path_ER_sparse+str(i)+".txt"),"r"))
    SA_Files.append(open(os.path.join(path_SA + str(i) + ".txt"), "r"))
    SA_Sparse_Files.append(open(os.path.join(path_SA_sparse+str(i)+".txt"),"r"))

arg_1 = 0

ER_Sparse_Graphs.append(gg.readMatrixFromFile(ER_Sparse_Files[arg_1]))
SA_Sparse_Graphs.append(gg.readMatrixFromFile(SA_Sparse_Files[arg_1]))
ER_Graphs.append(gg.readMatrixFromFile(ER_Files[arg_1]))
SA_Graphs.append(gg.readMatrixFromFile((SA_Files[arg_1])))
ER_Dense_Graphs.append(gg.readMatrixFromFile(ER_Dense_Files[arg_1]))
AList, freqs = get_AList(ER_Sparse_Graphs[0], ER_Dense_Graphs[0], dens_const=False)
ICs = get_ICs(open(path_random_ICs + str(arg_1) + ".txt", "r"))

freqs = gg.readMatrixFromFile(open(path_random_nat_freqs+str(arg_1)+".txt", "r"))[0]
runSim(AList, ICs, freqs)

timeArray = [dt* i for i in range(len(standardOPData))]

ta_OP = averagedOPData
f4 = plt.figure(4)
plt.xlabel('Graph')
plt.ylabel('Time-averaged Order Parameter')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.scatter(list(range(len(ta_OP) // 2)), ta_OP[0: (len(ta_OP) // 2)], color='blue',
                marker='>', facecolors='none', s=25, lw=.5, alpha=1)
plt.scatter(list(range(len(ta_OP) // 2)), list(reversed(ta_OP[len(ta_OP) // 2:])),
                color='red', marker='<', s=25, lw=.5, edgecolors='black', alpha=1)
plt.axvline(x=(len(AList) / 2.0 + startDelay), color='orange', linewidth=0.2)


plt.show()