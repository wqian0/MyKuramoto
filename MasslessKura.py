import numpy as np
from scipy.integrate import ode, odeint
from scipy.integrate import ode
import numpy as np
import os
import functools
import math
import cmath
#import pylab as plt
import GraphGenerator as gg
import matplotlib.pyplot as plt
import random as rd
import sys
from queue import *
#from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
N, alpha, dt, frequencyBound, steps, startDelay, endDelay= gg.size, .15, .02, gg.freqBound, 250, 0, 0
m = 0
oParameterData=[]
standardOPData =[]
localOPData = []
inst_kinetic = []
inst_potential = []

pos_OPData = []
neg_OPData = []
averagedOPData = []
averagedLocalOPData = []
inst_freqs = []
inst_phases = []

head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"
#head_dir = "/data/jux/bqqian/Kuramoto"
def getRandomDistribution(N, lowerBound, upperBound, distribution):
    list = []
    for r in range(N):
        list.append(distribution(lowerBound,upperBound))
    return list

def stepRungeKutta(A, w, theta, a, dt):
    l1,l2,l3,l4 = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

    for i in range(N):
        l1[i]= dt * (w[i] + a*sum(A[i]*np.sin(theta-theta[i])))
    theta_rk = [x+y / 2 for x,y in zip(theta,l1)]

    for i in range(N):
        l2[i] = dt * (w[i] + a * sum(A[i] * np.sin(theta_rk - theta_rk[i])))
    theta_rk = [x + y / 2 for x, y in zip(theta, l2)]

    for i in range(N):
        l3[i] = dt * (w[i] + a*sum(A[i] * np.sin(theta_rk - theta_rk[i])))
    theta_rk = [x + y  for x, y in zip(theta, l3)]

    for i in range(N):
        l4[i] = dt * (w[i] + a* sum(A[i] * np.sin(theta_rk - theta_rk[i])))
    l = (1/6)*(l1+2*l2+2*l3+l4)
    return [x+y for x,y in zip(theta ,l)], l/dt


def runRK(A, phases0, w, dt, num_steps):
    theta, w_inst =np.zeros((N, num_steps + 1)), np.zeros((N, num_steps))
    theta[:,0] = [x for x in phases0]
    #oParameterData.append(structuralOrderParameter(A, theta, 0))
    standardOPData.append(orderParameter(theta, 0))
    #localOPData.append(localOrderParameter(A, theta, 0))
    for t in range(num_steps-1):
        theta[:,t+1], w_inst[:, t] = stepRungeKutta(A, w, theta[:, t], alpha, dt)
        inst_phases.append(theta[:, t+1])
        #inst_freqs.append(w_inst[:, t])
        #oParameterData.append(structuralOrderParameter(A, theta, t + 1))
        standardOPData.append(orderParameter(theta, t+1))
        #localOPData.append(localOrderParameter(A, theta, t+1))
    theta[:, num_steps], w_inst[:, num_steps-1] = stepRungeKutta(A, w, theta[:, num_steps-1], alpha, dt)
    inst_phases.append(theta[:, num_steps])
    inst_freqs.append(w_inst[:, num_steps-1])
    return theta

def Kura(init, t, A, w_nat, a):
    theta = np.array(init)
    dot_theta = np.zeros(N)
    for i in range(N):
        dot_theta[i] = w_nat[i] + a * np.dot(A[i], np.sin(theta - theta[i]))
    return dot_theta


def runRK2(A, phases0, w_nat, time):
    result = odeint(Kura, phases0, time, args=(A, w_nat, alpha), atol = 1e-12, rtol = 1e-12)
    for t in range(len(result) - 1):
        standardOPData.append(abs(complex_OP2(result[t])))
        # complex_OP.append(complex_r)
        inst_phases.append(result[t])
        inst_freqs.append((result[t+1]-result[t])/dt)
           # inst_kinetic.append(get_kinetic(inst_freqs[len(inst_freqs)-1]))
           # inst_potential.append(get_potential(A, result[t]))

        # localOPData.append(localOrderParameter(A, theta[t]))
        # localOPData.append(LOP2(A, theta[t], nz))
        # clusterOPData.append(clusterOrderParameter(gg.clusters, theta[t]))
    return result


def complex_OP2(theta):
    return (1.0 / N) * sum(cmath.exp(complex(0, a)) for a in theta)

def orderParameter(theta, t):
    return (1.0/N)*abs(sum(np.e ** complex(0, a) for a in theta[:, t]))

# def structuralOrderParameter(A, theta, t):
#     count =0.0
#     sum =0.0
#     for r in range(N):
#         for c in range(r, N):
#             if A[r][c] > 0:
#                 count += A[r][c]
#                 sum += A[r][c] * np.abs(np.cos((theta[r][t]-theta[c][t])/2))
#             elif A[r][c] <0:
#                 count += - A[r][c]
#                 sum += -A[r][c] * (1.0 - np.abs(np.cos((theta[r][t]-theta[c][t])/2.0)))
#     return sum/count

def structuralOrderParameter(A, theta, t):
    count =0.0
    sum =0.0
    for r in range(N):
        for c in range(r, N):
            if A[r][c] > 0:
                count += A[r][c]
                sum += .5*A[r][c] * abs(np.e ** complex(0, theta[r][t]) + np.e ** complex(0, theta[c][t]))
    return sum/count

def localOrderParameter(A, theta, t):
    degSum = 0.0
    total = 0.0
    currentSum = 0.0
    for r in range(N):
        for c in range(N):
            if A[r][c]>0:
                currentSum += np.e ** complex(0, theta[c][t])
                degSum+=A[r][c]
        total += abs(currentSum)
        currentSum = 0
    return total/degSum

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
    theta_0 = runRK2(AMList[0], phases, frequencies, time)
    averagedOPData.append(averagedOP(len(standardOPData)-steps + 1, len(standardOPData), standardOPData))
    endStateTheta = theta_0[len(theta_0) - 1, :]
    for i in range(startDelay):
        theta = runRK2(AMList[0], endStateTheta, frequencies, time)
        endStateTheta = theta[len(theta) - 1, :]
        print(i)
        averagedOPData.append(averagedOP(len(standardOPData) - steps + 1, len(standardOPData), standardOPData))
    for i in range(1, len(AMList)):
        theta = runRK2(AMList[i], endStateTheta, frequencies, time)
        endStateTheta = theta[len(theta) - 1, :]
        print(i)
        averagedOPData.append(averagedOP(len(standardOPData) -steps + 1, len(standardOPData), standardOPData))
    for i in range(endDelay):
        theta = runRK2(AMList[len(AMList)-1], endStateTheta, frequencies, time)
        endStateTheta = theta[len(theta) - 1, :]
        print(i)
        averagedOPData.append(averagedOP(len(standardOPData) - steps + 1, len(standardOPData), standardOPData))

def get_AList(start=None, end=None, final=None, dens_const=True, transitions = gg.numTransitionGraphs):
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

timeArray = [dt* i for i in range(len(standardOPData))]
# timeArray = [dt* i for i in range(len(oParameterData))]
'''
for i in range(N):2
    plt.subplot(N+1, 1, 1 + i)
    plt.plot(timeArray,theta[i])
'''
'''
titleText = str(alpha)+" "+str(frequencyBound) +" " + str(dt)+" "+str(steps)+" "+str(gg.size)+" "+str(gg.numTransitionGraphs)+"x"+str(gg.dE)+" "+str(gg.modules)+" "+str(gg.pModular)+" "+str(gg.finalEdges)+" "+str(gg.rSeed)

# plt.axvline(x=startDelay*dt*steps)
# plt.axvline(x=(startDelay+len(AList))*dt*steps)
# plt.axvline(x=dt*steps*200)



plt.tight_layout()
plt.rcParams.update({'font.size': 40})
plt.figure(figsize=(15,9))
f2 = plt.figure(2, dpi = 3000)
axes = plt.gca()
axes.set_ylim([0,1])
plt.xlim([0, 2032])
plt.xlabel('time, '+r'$t$', size = 44)
plt.rcParams.update({'font.size': 40})
plt.ylabel('Order Parameter, '+r'$R$', size = 44)
plt.tight_layout()
#plt.title(titleText)
# plt.axvline(x=(delay)*dt)
#plt.axvline(x=len(AList)*steps*dt/2.0, color = 'green')
#plt.axvline(x=(startDelay*steps)*dt, color='red')
#plt.axvline(x=((startDelay+len(AList))*steps)*dt, color='red')
# plt.plot(timeArray, standardOPData, color='black', linewidth = 1)
time_array = timeArray[0:len(timeArray)//2]
first_half = standardOPData[0 : (len(standardOPData)//2)]
second_half = list(reversed(standardOPData[(len(standardOPData)//2) : len(standardOPData)]))
plt.plot(time_array, first_half, color='blue', linewidth = 1)
#plt.plot(time_array, second_half, color='red', linewidth = .4)
plt.plot(timeArray[len(timeArray)//2:], list(reversed(second_half)), color='red', linewidth = .5)
plt.savefig('OP_massless.pdf')
plt.savefig('OP_massless.png')


ta_OP = averagedOPData
f4 = plt.figure(4)
plt.xlabel('Graph')
plt.ylabel('Time-averaged Order Parameter')
plt.title(titleText)
axes = plt.gca()
axes.set_ylim([0, 1])
plt.scatter(list(range(len(ta_OP) // 2)), ta_OP[0: (len(ta_OP) // 2)], color='blue',
            marker='>', facecolors='none', s=25, lw=.5, alpha=1)
plt.scatter(list(range(len(ta_OP) // 2)), list(reversed(ta_OP[len(ta_OP) // 2:])),
           color='red', marker='<', s=25, lw=.5, edgecolors='black', alpha=1)
#plt.scatter(list(range(len(ta_OP) // 2, len(ta_OP))), ta_OP[len(ta_OP) // 2:],
#           color='red', marker='<', s=25, lw=.5, edgecolors='black', alpha=1)
    # plt.scatter(list(range(len(ta_OP))), ta_OP, color='black', s = 15)
    # plt.axvline(x=(len(AList) + startDelay), color='red', linewidth=0.2)
    # plt.axvline(x=(startDelay), color='red', linewidth=0.2)
    # plt.axvline(x=(len(AList) / 2.0 + startDelay), color='orange', linewidth=0.2)

plt.rcParams.update({'font.size': 40})
plt.tight_layout()
f7 = plt.figure(7, dpi=3000)
plt.figure(figsize=(15,4))
plt.tight_layout()
#plt.xlabel('time, '+r'$t$', size = 20)
plt.ylabel(r'$\{\dot{\theta _i}\}$', size = 44)
plt.xlim([0, 2032])
plt.ylim([-6.1,6.1])
plt.tick_params(labelbottom=False, bottom = False)
plt.rcParams.update({'font.size': 40})
#plt.title("Frequencies over time")
plt.tight_layout()
# plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
# plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
# plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
IF = np.array(inst_freqs)
for i in range(N):
    half_1 = [a for a in IF[0:len(IF) // 2, i]]
    half_2 = [a for a in IF[len(IF) // 2:, i]]
    plt.plot(timeArray[0:len(timeArray) // 2], half_1, color='blue', linewidth = 0.1)
    plt.plot(timeArray[len(timeArray) // 2 : ], half_2, color='red', linewidth=0.1)
    #plt.plot(timeArray, IF[:, i], color='blue', linewidth=0.05)
plt.savefig('massless_freqs__nolabel_notick.pdf')
plt.savefig('massless_freqs__nolabel_notick.png')

'''

OPs = []
ta_OPs = []
ss_diffs = []
end_states = []

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
standard_ER = gg.readMatrixFromFile(open(head_dir+"/static ER.txt", "r"))
for i in range(25):
    ER_Files.append(open(os.path.join(path_ER + str(i) + ".txt"), "r"))
    ER_Dense_Files.append(open(os.path.join(path_ER_dense + str(i) + ".txt"), "r"))
    ER_Sparse_Files.append(open(os.path.join(path_ER_sparse+str(i)+".txt"),"r"))
    #misaligned_files.append(open(os.path.join(path_misaligned) + str(i) + ".txt", "r"))
    #mod_Files.append(open(os.path.join(path_modular) + str(i) + ".txt", "r"))
    #dense_mod_Files.append(open(os.path.join(path_dense_mod) + str(i) + ".txt", "r"))
    SA_Files.append(open(os.path.join(path_SA + str(i) + ".txt"), "r"))
    #SA_Sparse_Files.append(open(os.path.join(path_SA_sparse+str(i)+".txt"),"r"))
    #freq_mod_Files.append(open(os.path.join(path_freq_mod+str(i)+".txt"),"r"))
    #sparsefreq_mod_Files.append(open(os.path.join(path_sparsefreq_mod+str(i)+".txt"), "r"))


#arg_1 = int(sys.argv[1]) - 1 #Job number, ranging from 0 to 255

arg_1 = 0

ER_Graphs.append(gg.readMatrixFromFile(ER_Files[arg_1]))
SA_Graphs.append(gg.readMatrixFromFile(SA_Files[1]))
ER_Sparse_Graphs.append(gg.readMatrixFromFile(ER_Sparse_Files[arg_1]))
ER_Dense_Graphs.append(gg.readMatrixFromFile(ER_Dense_Files[arg_1]))

AList, freqs = get_AList(ER_Graphs[0], SA_Graphs[0], dens_const=True)
ICs = get_ICs(open(path_random_ICs+str(arg_1)+".txt", "r"))

runSim(AList, ICs, freqs)
timeArray = [dt* i for i in range(len(standardOPData))]

x = np.ones((100, 3))
y = np.ones((100, 3))

x[:, 0:3] = (1, 0, 0)
y[:, 0:3] = (0, 1, 0)
c = np.linspace(0, 1, 100)[:, None]
gradient = x + (y - x) * c
f_phase_fancy = plt.figure(10, dpi = 200)
plt.tight_layout()
IP = (np.array(inst_phases) + np.pi) % (2 * np.pi) - np.pi
IF = np.array(inst_freqs)

sorted_indices = np.argsort(gg.freqs)
IP_sorted = np.transpose(IP)[sorted_indices[::1]]
IF_sorted = np.transpose(IF)[sorted_indices[::1]]
IP_sorted = np.transpose(IP_sorted)
IF_sorted = np.transpose(IF_sorted)
plt.pause(15)

for i in range(0, len(timeArray), 3):
    plt.clf()
    plt.xlabel('phase')
    plt.ylabel('frequency')
    title = "Phase Space Plot" + " (Time: " + str(round(i * dt)) + ")"
    if i / steps <= startDelay:
        title += " (Start Delay)"
    elif i / steps >= len(AList) + startDelay:
        title += " (End Delay)"
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-np.pi, np.pi])
    axes.set_ylim([-5, 5])
    if i >= 4:
        plt.scatter(IP_sorted[i - 4, ::], IF_sorted[i - 4, ::], c=gradient, s=8, alpha=.2)
        plt.scatter(IP_sorted[i-3, ::], IF_sorted[i-3, ::], c=gradient, s=11, alpha = .4)
        plt.scatter(IP_sorted[i - 2, ::], IF_sorted[i - 2, ::], c=gradient, s=14, linewidth=.2,
                    alpha=.6)
        plt.scatter(IP_sorted[i - 1, ::], IF_sorted[i - 1, ::], c=gradient, s=17, linewidth=.2,
                    alpha=.8)
    # curr_graph = AList[i // steps]
    # x_coords = []
    # y_coords = []
    # for r in range(len(curr_graph)):
    #     for c in range(r, len(curr_graph)):
    #         if curr_graph[r][c] == 1:
    #             x_coords.append(IP_sorted[i, r])
    #             x_coords.append(IP_sorted[i, c])
    #             y_coords.append(IF_sorted[i, r])
    #             y_coords.append(IF_sorted[i, c])
    # plt.plot(x_coords, y_coords, linewidth=.1, color="black", alpha=.3)
    plt.scatter(IP_sorted[i], IF_sorted[i], c=gradient, s=20, edgecolors = 'black', linewidth = .2)
    plt.pause(1e-10)
plt.close()


# TA_OP_file = open("blank "+str(arg_1)+".txt", "w")
# gg.printArrayToFile(averagedOPData, TA_OP_file)
# TA_OP_file.close()
#
# inst_phases_file = open("inst_phases_density_massless.txt", "w")
# sorted_indices = np.argsort(gg.freqs)
# IP_sorted = np.transpose(inst_phases)[sorted_indices[::1]]
# IP_sorted = np.transpose(IP_sorted)
# IP_sorted = (np.array(IP_sorted) + np.pi) % (2 * np.pi) - np.pi
# IP_sorted = np.transpose(IP_sorted)
# gg.printMatrixToFile(IP_sorted,inst_phases_file)
# inst_phases_file.close()
#
# print(IP_sorted.shape)
# print(len(standardOPData))
# timeseries = plt.figure(1000, dpi=3000)
# plt.rcParams.update({'font.size': 16})
#
# plt.rcParams.update({'font.size': 16})
#
# plt.rcParams.update({'font.size': 16})
# ax = plt.gca()
# timeArray = [dt* i for i in range(len(standardOPData))]
# print(timeArray)
# im = ax.imshow(IP_sorted[:, ::10], cmap='twilight', interpolation='none', extent=[0, max(timeArray), 100, 1],
#                aspect=max(timeArray) / (3 * N))
# plt.xlabel("time, t")
# plt.ylabel("Oscillator ID")
# yticks = np.arange(0, 120, 20)
# plt.yticks(yticks)
# plt.colorbar(im, cax=timeseries.add_axes([0.93, 0.32, 0.023, 0.35]))
# plt.savefig('phase_timeseries_SA_1000_massless.pdf', bbox_inches='tight')
# plt.savefig('phase_timeseries_SA_1000_massless.png', bbox_inches='tight')

# gg.printArrayToFile(end_state_ICs, end_states[0][0])
# gg.printArrayToFile(end_state_ICs, end_states[0][1])
# end_state_ICs.close()


plt.show()
