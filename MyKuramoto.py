from scipy.integrate import ode, odeint
import numpy as np
#from colour import Color
#import functools
import math
import cmath
import os
import sys
import time as tt
import GraphGenerator as gg
# import multiprocessing as mp
# from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import tornado
from matplotlib import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
# from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import random as rd
import small_world as sw
# import pickle

# matplotlib.use('webagg')

head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"
#head_dir = "/data/jux/bqqian/Kuramoto"

N, alpha, dt, frequencyBound, steps, startDelay, endDelay = gg.size, .3, .02, gg.freqBound, 250, 0,0
m = 2
D = 1
f = dt / m
global end_s, end_f, oParameterData, standardOPData, localOPData, averagedOPData, averagedLocalOPData, inst_freqs, inst_phases
oParameterData = []
standardOPData = []
localOPData = []

pos_OPData = []
neg_OPData = []
averagedOPData = []
averagedLocalOPData = []
inst_freqs = []
inst_phases = []
inst_kinetic = []
inst_potential = []
complex_OP = []
end_s = []
end_f = []


def getRandomDistribution(N, lowerBound, upperBound, distribution):
    list = []
    for r in range(N):
        list.append(distribution(lowerBound, upperBound))
    return list


def get_kinetic(inst_freqs):
    return .5 * m * np.dot(inst_freqs, inst_freqs)


def get_potential(A, inst_phases):
    sum = 0
    for r in range(N - 1):
        for c in range(r + 1, N):
            if A[r][c] > 0:
                sum += 1 - math.cos(inst_phases[r] - inst_phases[c])
    return alpha * 2 * sum

def Jacobian(init, t, A, w_nat, a, M):
    theta, omega = init[:N, ], init[N:, ]
    block_11 = np.zeros((N, N))
    block_21 = np.identity(N)
    block_22 = -D/M * np.identity(N)
    block_12 = np.zeros((N,N))
    print("jacobian")
    for i in range(N):
        block_12[i] = np.dot(A[i], np.cos(theta - theta[i]))
        block_12[i,i] = -np.sum(block_21[i])
    block_12 *= a/M
    return np.block([[block_11, block_12], [block_21, block_22]])

def Kura(init, t, A, w_nat, a, M):
    theta, omega = init[:N, ], init[N:, ]
    dot_w = np.zeros(N)
    for i in range(N):
        dot_w[i] = (1 / M) * (-D* omega[i] + w_nat[i] + a * np.dot(A[i], np.sin(theta - theta[i])))
    return np.concatenate([omega, dot_w])


def runRK2(A, phases0, w_nat, w0, a, M, time):
    result = odeint(Kura, np.concatenate([phases0, w0]), time, Dfun = Jacobian, args=(A, w_nat, a, M), rtol = 1e-12, atol = 1e-12)
    theta, omega = result[:, :N], result[:, N:]
    for t in range(len(theta)-1):
        complex_r = complex_OP2(theta[t])
        standardOPData.append(abs(complex_r))
       # complex_OP.append(complex_r)
        inst_phases.append(theta[t])
        inst_freqs.append(omega[t])
        # localOPData.append(localOrderParameter(A, theta[t]))
        # localOPData.append(LOP2(A, theta[t], nz))
    return theta, omega


def stepRungeKutta(A, theta0, w_nat, w0, a, dt, nz, l1=np.zeros(N), l2=np.zeros(N), l3=np.zeros(N), l4=np.zeros(N)):
    for i in range(N):
        l1[i] = f * (- D * w0[i] + w_nat[i] + a * np.sum(np.sin(theta0[nz[i]] - theta0[i])))
    k1 = dt * w0

    omega_rk = w0 + l1 / 2
    theta_rk = theta0 + k1 / 2

    for i in range(N):
        l2[i] = f * (-D * omega_rk[i] + w_nat[i] + a * np.sum(np.sin(theta_rk[nz[i]] - theta_rk[i])))
    k2 = dt * omega_rk

    omega_rk = w0 + l2 / 2
    theta_rk = theta0 + k2 / 2

    for i in range(N):
        l3[i] = f * (-D * omega_rk[i] + w_nat[i] + a * np.sum(np.sin(theta_rk[nz[i]] - theta_rk[i])))
    k3 = dt * omega_rk

    omega_rk = w0 + l3
    theta_rk = theta0 + k3

    for i in range(N):
        l4[i] = f * (-D * omega_rk[i] + w_nat[i] + a * np.sum(np.sin(theta_rk[nz[i]] - theta_rk[i])))
    k4 = dt * omega_rk
    l = (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
    k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return theta0 + k, w0 + l


def runRK(A, phases0, w_nat, w0, dt, num_steps):
    theta, w = np.zeros((N, num_steps + 1)), np.zeros((N, num_steps + 1))
    theta[:, 0], w[:, 0] = phases0, w0
    nz = []
    for i in range(N):
        nz.append(np.nonzero(A[i])[0])
    complex_r = complexOrderParameter(theta, 0)
    standardOPData.append(abs(complex_r))
    # complex_OP.append(complex_r)
    # localOPData.append(localOrderParameter(A, theta, 0))

    # inst_freqs.append(w[:, 0])
    inst_phases.append(theta[:, 0])
    # inst_kinetic.append(get_kinetic(w[:, 0]))
    # inst_potential.append(get_potential(A, theta[:, 0]))

    for t in range(num_steps - 1):
        theta[:, t + 1], w[:, t + 1] = stepRungeKutta(A, theta[:, t], w_nat, w[:, t], alpha, dt, nz)

        # inst_freqs.append(w[:, t+1])
        inst_phases.append(theta[:, t + 1])
        # inst_kinetic.append(get_kinetic(w[:, t+1]))
        # inst_potential.append(get_potential(A, theta[:, t+1]))

        complex_r = complexOrderParameter(theta, t + 1)
        standardOPData.append(abs(complex_r))
        # complex_OP.append(complex_r)
        # localOPData.append(localOrderParameter(A, theta, t+1))

    theta[:, num_steps], w[:, num_steps] = stepRungeKutta(A, theta[:, num_steps - 1], w_nat, w[:, num_steps - 1], alpha,
                                                          dt, nz)
    return theta, w


def orderParameter(theta, t):
    return (1.0 / N) * abs(sum(cmath.exp(complex(0, a)) for a in theta[:, t]))


def complexOrderParameter(theta, t):
    return (1.0 / N) * sum(cmath.exp(complex(0, a)) for a in theta[:, t])


def complex_OP2(theta):
    return (1.0 / N) * sum(cmath.exp(complex(0, a)) for a in theta)


def structuralOrderParameter(A, theta, t):
    count = 0.0
    sum = 0.0
    for r in range(N):
        for c in range(r, N):
            if A[r][c] > 0:
                count += A[r][c]
                sum += .5 * A[r][c] * abs(cmath.exp(complex(0, theta[r][t])) + cmath.exp(complex(0, theta[c][t])))
    return sum / count


def localOrderParameter(A, theta):
    degSum = 0.0
    total = 0.0
    currentSum = 0.0
    for r in range(N - 1):
        for c in range(r + 1, N):
            if A[r][c] > 0:
                currentSum += cmath.exp(complex(0, theta[c]))
                degSum += A[r][c]
        total += abs(currentSum)
        currentSum = 0
    return total / degSum


def LOP2(A, theta, nz):
    total = 0
    for r in range(N):
        total += abs(sum(cmath.exp(complex(0, x)) for x in theta[nz[r]]))
    return total / sum(len(a) for a in nz)



def averagedOP(start, end, OPData):
    return np.sum(OPData[start:end]) / (end - start)


def add_averaged_data(frac=1):
    global standardOPData, localOPData, averagedOPData, averagedLocalOPData
    averagedOPData.append(averagedOP(len(standardOPData) - steps // frac + 1, len(standardOPData), standardOPData))
    averagedLocalOPData.append(averagedOP(len(localOPData) - steps // frac + 1, len(localOPData), localOPData))


def runSim(AMList, natFreqs):
    global end_start, oParameterData, standardOPData, localOPData, averagedOPData, averagedLocalOPData, inst_freqs, inst_phases

    # init_phases = np.array(getRandomDistribution(N, -np.pi, np.pi, rd.uniform))
    # init_phases -= np.mean(init_phases)
    #     #
    #     # init_freqs = np.array(gg.getRandomDistribution(N, -gg.freqBound, gg.freqBound))
    #     # init_freqs -= np.mean(init_freqs)

    f = open("standard ICs.txt", "r")
    ICs = gg.readMatrixFromFile(f)
    init_phases = ICs[0]
    init_freqs = ICs[1]
    theta_0, w_0 = runRK(AMList[0], init_phases, natFreqs, init_freqs, dt, steps)
    endTheta, endW = theta_0[:, len(theta_0[0]) - 1], w_0[:, len(w_0[0]) - 1]
    add_averaged_data()
    for i in range(startDelay):
        theta, w = runRK(AMList[0], endTheta, natFreqs, endW, dt, steps)
        endTheta, endW = theta[:, len(theta[0]) - 1], w[:, len(w[0]) - 1]
        end_start = (endTheta, endW)
        print(i)
        add_averaged_data()
    for i in range(1, len(AMList)):
        theta, w = runRK(AMList[i], endTheta, natFreqs, endW, dt, steps)
        endTheta, endW = theta[:, len(theta[0]) - 1], w[:, len(w[0]) - 1]
        print(i)
        add_averaged_data()
    for i in range(endDelay):
        theta, w = runRK(AMList[len(AMList) - 1], endTheta, natFreqs, endW, dt, steps)
        endTheta, endW = theta[:, len(theta[0]) - 1], w[:, len(w[0]) - 1]
        print(i)
        add_averaged_data()
    # f = open("standard ICs.txt", "w")
    # gg.printArrayToFile(init_phases, f)
    # gg.printArrayToFile(init_freqs, f)
    return standardOPData, averagedOPData


def get_ICs(f=None, removeMean=False):
    if f is not None:
        ICs = gg.readMatrixFromFile(f)
        if removeMean:
            return ICs[0] - np.mean(ICs[0]), ICs[1] - np.mean(ICs[1])
        else:
            return ICs[0], ICs[1]
    else:
        init_phases = np.array(getRandomDistribution(N, -np.pi, np.pi, rd.uniform))
        init_freqs = np.array(gg.getRandomDistribution(N, -gg.freqBound, gg.freqBound))
        if removeMean:
            init_phases -= np.mean(init_phases)
            init_freqs -= np.mean(init_freqs)
        return init_phases, init_freqs


def runSim2(AMList, ICs, natFreqs, a, M):
    global end_s, end_f, oParameterData, standardOPData, localOPData, averagedOPData, averagedLocalOPData, inst_freqs, inst_phases

    init_phases, init_freqs = ICs
    init_phases = np.array(init_phases)
    init_freqs = np.array(init_freqs)
    time = np.linspace(0, dt * steps, steps)
    theta_0, w_0 = runRK2(AMList[0], init_phases, natFreqs, init_freqs, a, M, time)
    endTheta, endW = theta_0[len(theta_0) - 1, :], w_0[len(w_0) - 1, :]
    add_averaged_data()
    for i in range(startDelay):
        theta, w = runRK2(AMList[0], endTheta, natFreqs, endW, a, M, time)
        endTheta, endW = theta[len(theta) - 1, :], w[len(w) - 1, :]
        if i == startDelay - 1:
            end_s.append((endTheta, endW))
        print(i)
        add_averaged_data()
    for i in range(1, len(AMList)):
        theta, w = runRK2(AMList[i], endTheta, natFreqs, endW, a, M, time)
        endTheta, endW = theta[len(theta) - 1, :], w[len(w) - 1, :]
        print(str(i))
        add_averaged_data()
    for i in range(endDelay):
        theta, w = runRK2(AMList[len(AMList) - 1], endTheta, natFreqs, endW, a, M, time)
        endTheta, endW = theta[len(theta) - 1, :], w[len(w) - 1, :]
        if i == endDelay - 1:
            end_f.append((endTheta, endW))
        print(i)
        add_averaged_data()
    # f = open("ER to SA to ER end-state, (static ER, sync aligned whole function 500 edges, 0 mean, 3 bounded).txt", "w")
    # gg.printArrayToFile(endTheta, f)
    # gg.printArrayToFile(endW, f)
    # f.close()
    return standardOPData, averagedOPData, endTheta, endW


def get_phase_diffs(phases):
    result = np.zeros((N, N))
    for r in range(N - 1):
        for c in range(r + 1, N):
            result[r][c] = phases[r] - phases[c]
            result[c][r] = -result[r][c]
    return result


def get_ss_diff(ta_OP):
    return ta_OP[len(ta_OP) - 1] - ta_OP[startDelay]



def plot(AList, col, width=.4, size=25, transp=1, e_s=end_s, e_f=end_f, IP=inst_phases, IF=inst_freqs,
         OP=standardOPData, co_OP=complex_OP,  LOP=localOPData, ta_OP=averagedOPData,
         ta_LOP=averagedLocalOPData, inst_kinetic = inst_kinetic, inst_potential = inst_potential):
    timeArray = np.array([dt * i for i in range(len(OP))])
    # timeArray = [dt* i for i in range(len(oParameterData))]
    '''
    for i in range(N):2
        plt.subplot(N+1, 1, 1 + i)
        plt.plot(timeArray,theta[i])
    '''

    titleText = str(alpha) + " " + str(frequencyBound) + " " + str(dt) + " " + str(steps) + " " + str(
        gg.size) + " " + str(gg.numTransitionGraphs) + "x" + str(gg.dE) + " " + str(gg.modules) + " " + str(
        gg.pModular) + " " + str(gg.finalEdges) + " " + str(gg.rSeed)
    '''
    plt.rcParams.update({'font.size': 40})
    
    plt.rcParams.update({'font.size': 40})
    f2 = plt.figure(2, dpi=3000)
    plt.figure(figsize=(15, 9))
    axes = plt.gca()
    axes.set_ylim([0, 1])
    #plt.xlim([0, 2032])
    plt.xlabel('time, ' + r'$t$', size = 44)
    plt.rcParams.update({'font.size': 40})
    plt.ylabel('Order Parameter, ' + r'$R$', size=44)
    plt.tight_layout()
    #plt.title(titleText)
    # plt.axvline(x=(len_A / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    # plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    # plt.axvline(x=(len_A + startDelay) * steps * dt, color='red', linewidth=0.2)
    # plt.axvline(x=(startDelay*steps)*dt, color='red')
    # plt.axvline(x=((startDelay+len(AList))*steps)*dt, color='red')
    # plt.plot(timeArray, OP, color=col, lw=width, alpha=transp * 8)
    plt.plot(timeArray[0:len(timeArray) // 2], OP[0 : (len(OP)//2)], color='blue', linewidth = 1)
    #plt.plot(timeArray[0:len(timeArray) // 2], list(reversed(OP[(len(OP) // 2): ])), color='red', linewidth = .4)
    plt.plot(timeArray[len(timeArray) // 2 :], OP[(len(OP) // 2):], color='red', linewidth=1)
    plt.savefig('OP_mass_randomstuff.pdf')
    #plt.savefig('OP_mass_randomstuff.png')
    '''

    f4 = plt.figure(4)
    plt.xlabel('Graph')
    plt.ylabel('Time-averaged Order Parameter')
    plt.title(titleText)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.scatter(list(range(len(ta_OP) // 2)), ta_OP[0: (len(ta_OP) // 2)], color='blue',
                marker='>', facecolors='none', s=size, lw=.5, alpha=transp)
    plt.scatter(list(range(len(ta_OP) // 2)), list(reversed(ta_OP[len(ta_OP) // 2:])),
                color='red', marker='<', s=size, lw=.5, edgecolors='black', alpha=transp)
    # plt.scatter(list(range(len(ta_OP))), ta_OP, color='black', s = 15)
    # plt.axvline(x=(len(AList) + startDelay), color='red', linewidth=0.2)
    # plt.axvline(x=(startDelay), color='red', linewidth=0.2)
    plt.axvline(x=(len(AList) / 2.0 + startDelay), color='orange', linewidth=0.2)

    # IP = np.array(IP)
    # IF = np.array(IF)
    # f_phase_snapshot_cont = plt.figure(162)
    #
    # for i in range(0, len(timeArray), 50):
    #     plt.clf()
    #     axes = plt.gca()
    #     axes.set_ylim([-np.pi, np.pi])
    #     plt.xlabel('Oscillator ID')
    #     plt.ylabel('Instantaneous phase')
    #     curr_phase = np.array(IP[i, :] + np.pi) % (2 * np.pi) - np.pi
    #     combined_phase = [x for y, x in sorted(zip(gg.freqs, curr_phase))]
    #     plt.scatter(range(N), combined_phase, c='blue', s=size)
    #     plt.pause(.00001)
    #
    #
    # f_freq_snapshot_cont = plt.figure(163)
    # for i in range(0, len(timeArray), 50):
    #     plt.clf()
    #     axes = plt.gca()
    #     axes.set_ylim([-5, 5])
    #     plt.xlabel('Oscillator ID')
    #     plt.ylabel('Instantaneous frequency')
    #     curr_freqs = IF[i, :]
    #     combined_freqs = [x for y, x in sorted(zip(gg.freqs, curr_freqs))]
    #     plt.scatter(range(N), combined_freqs, c='blue', s=size)
    #     plt.pause(.00001)

    '''
    f3 = plt.figure(3)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.xlabel('time')
    plt.ylabel('Local Order Parameter')
    plt.title(titleText)
    # plt.axvline(x=(delay)*dt)
    # plt.axvline(x=(startDelay*steps)*dt, color='red')
    # plt.axvline(x=((startDelay+len(AList))*steps)*dt, color='red')
    plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    plt.plot(timeArray, LOP, color='black', lw = .4)
    # plt.plot(timeArray[0:len(timeArray) // 2], LOP[0 : (len(LOP)//2)], color='blue', linewidth = .4)
    # plt.plot(timeArray[0:len(timeArray) // 2], list(reversed(LOP[(len(LOP)//2) :])), color='red', linewidth = .4)
    f5 = plt.figure(5)
    plt.xlabel('Graph')
    plt.ylabel('Time-averaged Local Order Parameter')
    plt.title(titleText)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    rand_col = np.random.rand(3,)
    plt.scatter(list(range(len(ta_LOP) // 2)), ta_LOP[0: (len(ta_LOP) // 2)], color = 'blue')
    plt.scatter(list(range(len(ta_LOP) // 2)), list(reversed(ta_LOP[len(ta_LOP) // 2:])), color = 'red')
    #plt.axvline(x=(len(AList) + startDelay), color='red', linewidth=0.2)
    plt.axvline(x=(startDelay), color='red', linewidth=0.2)
    #plt.axvline(x=(len(AList) / 2.0 + startDelay), color='orange', linewidth=0.2)
    #plt.scatter(list(range(1, len(ta_LOP) + 1)), ta_LOP, color=np.random.rand(3,),s = 5)
    '''

    '''
    f_OP_phase = plt.figure(15)
    axes = plt.gca()
    plt.figure(dpi=200)
    plt.xlabel('Re(r)')
    plt.ylabel('Im(r)')
    plt.pause(15)
    for i in range(0, len(timeArray), 3):
        #plt.clf()
        title = "Order Parameter Plot" + " (Time: " + str(round(i * dt)) + ")"
        if i / steps <= startDelay:
            title += " (Start Delay)"
        elif i / steps >= len(AList) + startDelay:
            title += " (End Delay)"
        plt.title(title)
        axes = plt.gca()
        axes.set_xlim([-1, 1])
        axes.set_ylim([-1, 1])
        # if i >= 4:
        #     plt.scatter(co_OP[i-4].real, co_OP[i-4].imag, c="deepskyblue", s=8, alpha=.2)
        #     plt.scatter(co_OP[i-3].real, co_OP[i-3].real, c="deepskyblue", s=11, alpha=.4)
        #     plt.scatter(co_OP[i-2].real, co_OP[i-2].real, c="deepskyblue", s=14, linewidth=.2,
        #                 alpha=.6)
        #     plt.scatter(co_OP[i-1].real, co_OP[i-1].real, c="deepskyblue", s=17, linewidth=.2,
        #                 alpha=.8)
        plt.scatter(co_OP[i].real, co_OP[i].imag, color = cm.hot(i/len(timeArray)), s=4, edgecolors= 'black', linewidth = .2, alpha = .4)
        plt.pause(1e-10)
    plt.close()
    
    IP = (np.array(IP) + np.pi) % (2 * np.pi) - np.pi
    '''
    '''
    f6 = plt.figure(6)
    plt.xlabel('Time')
    plt.ylabel('Instantaneous phases')
    plt.title("Phases over time")
    plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    IP = np.array(IP)
    for i in range(0, N, 10):
        # half_1 = [a for a in np.array(inst_phases)[0:len(inst_phases) // 2, i]]
        # half_2 = [a for a in np.array(inst_phases)[len(inst_phases) // 2:, i]]
        # plt.plot(timeArray[0:len(timeArray) // 2], half_1, color='blue', linewidth = 0.1)
        # plt.plot(timeArray[len(timeArray) // 2 :], half_2, color='red', linewidth=0.1)
        plt.plot(timeArray, IP[:,i], color='blue', linewidth=0.1)
    plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    '''


    x = np.ones((N, 3))
    y = np.ones((N, 3))

    x[:, 0:3] = (1, 0, 0)
    y[:, 0:3] = (0, 1, 0)
    c = np.linspace(0, 1, N)[:, None]
    gradient = x + (y - x) * c
    f_phase_fancy = plt.figure(10)
    #plt.figure(dpi=200)
    plt.tight_layout()
    IP = (np.array(IP) + np.pi) % (2 * np.pi) - np.pi
    IF = np.array(IF)

    sorted_indices = np.argsort(gg.freqs)
    IP_sorted = np.transpose(IP)[sorted_indices[::1]]
    IF_sorted = np.transpose(IF)[sorted_indices[::1]]
    IP_sorted = np.transpose(IP_sorted)
    IF_sorted = np.transpose(IF_sorted)
    plt.pause(3)
    for i in range(0, len(timeArray), 15):
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
        #         if curr_graph[r][c] != 0:
        #             x_coords.append(IP_sorted[i, r])
        #             x_coords.append(IP_sorted[i, c])
        #             y_coords.append(IF_sorted[i, r])
        #             y_coords.append(IF_sorted[i, c])
        # plt.plot(x_coords, y_coords, linewidth=.1, color="black", alpha=.2)
        plt.scatter(IP_sorted[i], IF_sorted[i], c=gradient, s=20, edgecolors = 'black', linewidth = 1)

        plt.pause(1e-10)
    plt.close()

    # sorted_indices = np.argsort(gg.freqs)
    # IP_sorted = np.transpose(IP)[sorted_indices[::1]]
    # IP_sorted = np.transpose(IP_sorted)
    # IP_sorted = (np.array(IP_sorted) + np.pi) % (2 * np.pi) - np.pi
    # IP_sorted = np.transpose(IP_sorted)
    # timeseries = plt.figure(1000, dpi = 3000)
    # plt.rcParams.update({'font.size': 16})
    #
    # plt.rcParams.update({'font.size': 16})
    #
    # plt.rcParams.update({'font.size': 16})
    # ax = plt.gca()
    # print(timeArray)
    # im = ax.imshow(IP_sorted[:, ::10], cmap = 'twilight', interpolation = 'none', extent=[0, max(timeArray), 100, 1], aspect=max(timeArray)/(3*N))
    # plt.xlabel("time, t")
    # plt.ylabel("Oscillator ID")
    # yticks = np.arange(0, 120, 20)
    # plt.yticks(yticks)
    # plt.colorbar(im, cax = timeseries.add_axes([0.93, 0.32, 0.023, 0.35]))
    # plt.savefig('phase_timeseries_density_point3.pdf', bbox_inches='tight')
    # plt.savefig('phase_timeseries_density_point3.png', bbox_inches='tight')


    '''
    # inst_phases = np.arctan2(np.sin(inst_phases), np.cos(inst_phases))
    sample = np.random.choice(N, 5, replace=False)
    f_phase = plt.figure(10)
    plt.xlabel('phase')
    plt.ylabel('frequency')
    plt.title("Phase space plot")
    IF = np.array(IF)
    # plt.scatter(inst_phases[:inst_phases.shape[0] // 3, sample[0]], inst_freqs[:inst_freqs.shape[0] // 3, sample[0]], c='red', s = .1)
    # plt.scatter(inst_phases[inst_phases.shape[0] // 3: 2*inst_phases.shape[0] // 3, sample[0]], inst_freqs[inst_freqs.shape[0] // 3:inst_freqs.shape[0]*2 // 3, sample[0]], c='orange', s = .1)
    # plt.scatter(inst_phases[2*inst_phases.shape[0] // 3:, sample[0]], inst_freqs[2*inst_freqs.shape[0] // 3:, sample[0]], c='green', s = .1)
    plt.scatter(IP[::, sample[0]], IF[::, sample[0]], c=timeArray[::], s = .01)
    plt.scatter(IP[::, sample[1]], IF[::, sample[1]], c=timeArray[::], s = .01)
    plt.scatter(IP[::, sample[2]], IF[::, sample[2]], c=timeArray[::], s = .01)
    img = plt.scatter(IP[::, sample[3]], IF[::, sample[3]], c=timeArray[::], s = .01)
    plt.colorbar(img)
    #plt.scatter([a for a in inst_phases[:, sample[4]]], [a for a in inst_freqs[:, sample[4]]], c='black', s=.01)
    #img = plt.scatter(inst_phases[inst_phases.shape[0] // 2:, 0], inst_freqs[inst_phases.shape[0] // 2:, 0], c=timeArray[len(timeArray) // 2:], s=.05)
    #plt.colorbar(img)
    '''
    '''
    f_phase_snapshot = plt.figure(160)
    plt.xlabel('Oscillator ID')
    plt.ylabel('Instantaneous phase')
    e_s_phase = (np.array(e_s[0][0]) + np.pi) % (2 * np.pi) - np.pi
    e_f_phase = (np.array(e_f[0][0]) + np.pi) % (2 * np.pi) - np.pi
    combined_p0 = zip(gg.freqs, e_s_phase)
    combined_pf = zip(gg.freqs, e_f_phase)
    combined_p0 = sorted(combined_p0)
    combined_pf = sorted(combined_pf)
    e_s_phase = np.sin([x for y, x in combined_p0])
    e_f_phase = np.sin([x for y, x in combined_pf])
    plt.scatter(range(N), e_s_phase, c='blue', s = size)
    plt.scatter(range(N), e_f_phase, c='red', s= size)
    f_freq_snapshot = plt.figure(161)
    plt.xlabel('Oscillator ID')
    plt.ylabel('Instantaneous frequency')
    combined_p0 = zip(gg.freqs, e_s[0][1])
    combined_pf = zip(gg.freqs, e_f[0][1])
    combined_p0 = sorted(combined_p0)
    combined_pf = sorted(combined_pf)
    e_s_freq = [x for y, x in combined_p0]
    e_f_freq = [x for y, x in combined_pf]
    plt.scatter(range(N), e_s_freq, c='blue', s = size)
    plt.scatter(range(N), e_f_freq, c='red', s = size)
    '''

    '''
    plt.rcParams.update({'font.size': 40})
    plt.tight_layout()
    f7 = plt.figure(7, dpi=3000)
    plt.figure(figsize=(15, 4))
    plt.tight_layout()
    #plt.xlabel('time, ' + r'$t$', size=24)
    plt.ylabel(r'$\{\dot{\theta _i}\}$', size=44)
    #plt.xlim([0, 2032])
    plt.ylim([-6.1, 6.1])
    plt.rcParams.update({'font.size': 40})
    plt.tick_params(labelbottom=False, bottom=False)
    plt.tight_layout()
    plt.ylim([-6, 6])
    #plt.title("Frequencies over time")
    # plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    # plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    # plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    IF = np.array(IF)
    for i in range(N):
        half_1 = [a for a in IF[0:len(IF) // 2, i]]
        half_2 = [a for a in IF[len(IF) // 2:, i]]
        plt.plot(timeArray[0:len(timeArray) // 2], half_1, color='blue', linewidth = 0.1)
        plt.plot(timeArray[len(timeArray) // 2 : ], half_2, color='red', linewidth=0.1)
        #plt.plot(timeArray, IF[:, i], color='blue', linewidth=0.05)
    # plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    # plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    # plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    #plt.savefig('mass_freqs_nolabel_randomstuff.pdf')
    plt.savefig('mass_freqs_nolabel_randomstuff.png')
    '''
    '''
    f7 = plt.figure(8)
    plt.xlabel('Time')
    plt.ylabel('Instantaneous diff between freq and nat freq')
    plt.title("Frequencies over time")
    # plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    # plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    # plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    IF = np.array(IF)
    for i in range(0, N, 10):
        # half_1 = [a for a in np.array(inst_freqs)[0:len(inst_freqs) // 2, i]]
        # half_2 = [a for a in np.array(inst_freqs)[len(inst_freqs) // 2:, i]]
        # plt.plot(timeArray[0:len(timeArray) // 2], half_1, color='blue', linewidth = 0.1)
        # plt.plot(timeArray[len(timeArray) // 2 : ], half_2, color='red', linewidth=0.1)
        plt.plot(timeArray, IF[:, i] - gg.freqs[i], color='blue', linewidth=0.05)
    plt.axvline(x=(len(AList) / 2.0 + startDelay) * steps * dt, color='orange', linewidth=0.2)
    plt.axvline(x=(startDelay) * steps * dt, color='red', linewidth=0.2)
    plt.axvline(x=(len(AList) + startDelay) * steps * dt, color='red', linewidth=0.2)
    '''
    '''
    f_energy = plt.figure(11)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy over time')
    inst_kinetic = np.array(inst_kinetic)
    inst_potential = np.array(inst_potential)
    plt.plot(timeArray, inst_potential, color = 'blue', lw = .4)
    plt.plot(timeArray, inst_kinetic, color = 'red', lw = .4)
    plt.plot(timeArray, inst_kinetic + inst_potential, color = 'black', lw = .4)
    '''
    '''
    # f8 = plt.figure(8)
    # plt.xlabel('Time')
    # plt.ylabel('Average Frequency')
    # plt.title('Average frequency experiment')
    # plt.plot(timeArray, [np.average(a) for a in np.array(inst_freqs)], color = 'black', linewidth = 1)
    '''


    '''
    f_phase_diffs = plt.figure(30)
    plt.xlabel('Oscillator ID')
    plt.ylabel('Oscillator ID')
    plt.title('Phase Differences')
    IP = np.array(IP)
    IP = (np.array(IP) + np.pi) % (2 * np.pi) - np.pi
    diffs = np.zeros((N,N, len(timeArray)))
    for i in range(len(timeArray)):
        diffs[:,:,i] = get_phase_diffs(IP[i,:])
    im = plt.imshow(np.zeros((N,N)), cmap='jet', vmin = -np.pi, vmax = np.pi)
    plt.colorbar(im)
    for i in range(0, len(timeArray), 50):
        im.set_data(diffs[:,:,i])
        plt.pause(.0001)
        plt.draw()
    '''


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
        # last_element = AList[len(AList)-1]
        #AList = AList[::10]
        #AList.append(last_element)
        AList = AList + list(reversed(AList))
    return np.array(AList), np.array(gg.freqs)


def run_and_plot(numTrials, AList, ICs, a, M, freqs, ss_diffs_file, ta_OP_file, end_state=None):
    for i in range(numTrials):
        # frequencies = getRandomDistribution(N,-frequencyBound,frequencyBound,random.uniform)
        OP, ta_OP, end_theta, end_freq = runSim2(AList, ICs, freqs, a, M)
        ss_diffs_file.write(str(0) + "\t" + str(a) + "\t" + str(M) + "\t" + str(get_ss_diff(ta_OP)))
        ss_diffs_file.write("\n")
        gg.printArrayToFile(ta_OP, ta_OP_file)
        print(ta_OP)
        print(len(ta_OP))
    if end_state is not None:
        for i in range(numTrials):
            end_state.append((end_theta, end_freq))


if __name__ == '__main__':
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
        misaligned_files.append(open(os.path.join(path_misaligned) + str(i) + ".txt", "r"))
        #mod_Files.append(open(os.path.join(path_modular) + str(i) + ".txt", "r"))
        #dense_mod_Files.append(open(os.path.join(path_dense_mod) + str(i) + ".txt", "r"))
        SA_Files.append(open(os.path.join(path_SA + str(i) + ".txt"), "r"))
        #SA_Sparse_Files.append(open(os.path.join(path_SA_sparse+str(i)+".txt"),"r"))
        freq_mod_Files.append(open(os.path.join(path_freq_mod+str(i)+".txt"),"r"))
        #sparsefreq_mod_Files.append(open(os.path.join(path_sparsefreq_mod+str(i)+".txt"), "r"))

    #
    # for i in range(25):
    #     ER_Graphs.append(gg.readMatrixFromFile(ER_Files[i]))
    #     #ER_Dense_Graphs.append(gg.readMatrixFromFile(ER_Dense_Files[i]))
    #     #ER_Sparse_Graphs.append(gg.readMatrixFromFile(ER_Sparse_Files[i]))
    #     freq_mod_Graphs.append(gg.readMatrixFromFile(freq_mod_Files[i]))
    #     #sparsefreq_mod_Graphs.append(gg.readMatrixFromFile(sparsefreq_mod_Files[i]))
    #     # MA_Graphs.append(gg.readMatrixFromFile(misaligned_files[i]))
    #     #SA_Graphs.append(gg.readMatrixFromFile(SA_Files[i]))
    #     #SA_Sparse_Graphs.append(gg.readMatrixFromFile((SA_Sparse_Files[i])))
    #     #mod_Graphs.append(gg.readMatrixFromFile(mod_Files[i]))
    #     #dense_mod_Graphs.append(gg.readMatrixFromFile(dense_mod_Files[i]))

    # To save memory, only reading in one graph for now:
# argument 1: which graphs to use. arg 2: which coupling. arg 3: num mass trials

   # arg_1 = int(sys.argv[1]) - 1 #Job number, ranging from 0 to 255
    arg_1 = 7

    ER_Graphs.append(gg.readMatrixFromFile(ER_Files[arg_1]))
    ER_Graphs.append(gg.readMatrixFromFile(ER_Files[arg_1+1]))
    ER_Sparse_Graphs.append(gg.readMatrixFromFile(ER_Sparse_Files[arg_1]))
    ER_Dense_Graphs.append(gg.readMatrixFromFile(ER_Dense_Files[arg_1]))
    SA_Graphs.append(gg.readMatrixFromFile(SA_Files[arg_1]))
    #MA_Graphs.append(gg.readMatrixFromFile(misaligned_files[0]))
    #freq_mod_Graphs.append(gg.readMatrixFromFile(freq_mod_Files[0]))

    AList, freqs = get_AList(ER_Graphs[0], SA_Graphs[0], dens_const=True)
    ICs = get_ICs()
    #freqs = gg.readMatrifxFromFile(open(path_random_nat_freqs+str(2)+".txt", "r"))[0]
    #ICs = get_ICs(open(path_random_ICs+str(arg_1)+".txt", "r"))
    #ICs = get_ICs(open(path_final_states+str(arg_1)+".txt", "r"))
    #ICs = get_ICs()

    ss_diffs_file = open("blank.txt", "w")
    TA_OP_file = open("blank2.txt", "w")
    #TA_OP_file = open("Graph_resolution_500" + str(arg_1) + ".txt", "w")
    #steps = 2500000 // 500
    #steps = 50
    #startDelay = 10 * (50 // 3)
    #endDelay = 10 * (50 // 3)
    #I really have 251 transition graphs from the jank way that I did this

    # coupling_vals = np.linspace(.2, .5, 16)
    # M_vals = np.linspace(1, 2.5, 16)
    # chosen_coupling = coupling_vals[arg_1 // 16]
    # chosen_mass = M_vals[arg_1 % 16]

    run_and_plot(1, AList, ICs, alpha, m, freqs, ss_diffs_file, TA_OP_file, end_state=end_states)
    ss_diffs_file.close()
    TA_OP_file.close()

    # gg.printArrayToFile(end_state_ICs, end_states[0][0])
    # gg.printArrayToFile(end_state_ICs, end_states[0][1])
    # end_state_ICs.close()

    # inst_phases_file = open("inst_phases_density.txt", "w")
    # sorted_indices = np.argsort(gg.freqs)
    # IP_sorted = np.transpose(inst_phases)[sorted_indices[::1]]
    # IP_sorted = np.transpose(inst_phases)
    # IP_sorted = (np.array(inst_phases) + np.pi) % (2 * np.pi) - np.pi
    # IP_sorted = np.transpose(inst_phases)
    # gg.printMatrixToFile(IP_sorted,inst_phases_file)
    # inst_phases_file.close()


    plot(AList, 'black')



    plt.show()