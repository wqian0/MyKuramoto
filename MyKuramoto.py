
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
# import tornado
# from matplotlib import pylab
# from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import random as rd
# import pickle

# matplotlib.use('webagg')

head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"
#head_dir = "/data/jux/bqqian/Kuramoto"

N, alpha, dt, frequencyBound, steps, startDelay, endDelay = gg.size, .3, .02, gg.freqBound, 50000, gg.numTransitionGraphs // 3, gg.numTransitionGraphs // 3
m = 2
D = 1
f = dt / m
global end_s, end_f, oParameterData, standardOPData, localOPData, clusterOPData, averagedOPData, averagedLocalOPData, averagedClusterData, inst_freqs, inst_phases
oParameterData = []
standardOPData = []
localOPData = []
clusterOPData = []

pos_OPData = []
neg_OPData = []
averagedOPData = []
averagedLocalOPData = []
averagedClusterData = []
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


def Kura(init, t, A, w_nat, a, M):
    theta, omega = init[:N, ], init[N:, ]
    # dot_w = np.zeros(N)
    delta_theta = np.subtract.outer(theta, theta)
    dot_w = (1/M) * (-D * omega + w_nat + a * np.einsum('ij,ji->i', A, np.sin(delta_theta)))
    # for i in range(N):
    #     dot_w[i] = (1 / M) * (-D * omega[i] + w_nat[i] + a * np.dot(A[i], np.sin(theta - theta[i])))
    return np.concatenate([omega, dot_w])


def Kura2(t, init, A, w_nat, a, M):
    return Kura(init, t, A, w_nat, a, M)

def Jacobian(init, t, A, w_nat, a, M):
    theta, omega = init[:N, ], init[N:, ]
    block_11 = np.zeros((N, N))
    block_21 = np.identity(N)
    block_22 = -D/M * np.identity(N)
    block_12 = np.zeros((N,N))
    for i in range(N):
        block_12[i] = np.dot(A[i], np.cos(theta - theta[i]))
        block_12[i,i] = -np.sum(block_21[i])
    block_12 *= a/M
    return np.block([[block_11, block_12], [block_21, block_22]])

def runRK2(A, phases0, w_nat, w0, a, M, time):
    result = odeint(Kura, np.concatenate([phases0, w0]), time, args=(A, w_nat, a, M))
    theta, omega = result[:, :N], result[:, N:]
    theta_slice = theta[:-1, :]
    standardOPData.extend(np.abs(np.apply_along_axis(complex_OP2, 1, theta_slice)))
    # for t in range(len(theta) - 1):
    #     complex_r = complex_OP2(theta[t])
    #     standardOPData.append(abs(complex_r))
    return theta, omega


def runRK3(A, phases0, w_nat, w0, a, time):
    result = ode(Kura2).set_integrator('dopri5', rtol=1e-9)
    result.set_initial_value(np.concatenate([phases0, w0]))
    result.set_f_params(A, w_nat, a)
    output = []
    while result.successful() and result.t < time[len(time) - 1]:
        output.append(result.integrate(result.t + dt))
    output = np.array(output)
    theta, omega = output[:, :N], output[:, N:]
    for t in range(len(theta) - 1):
        complex_r = complex_OP2(theta[t])
        standardOPData.append(abs(complex_r))
        inst_phases.append(theta[t])
        inst_freqs.append(omega[t])
        # localOPData.append(localOrderParameter(A, theta[t]))
        # localOPData.append(LOP2(A, theta[t], nz))
        # clusterOPData.append(clusterOrderParameter(gg.clusters, theta[t]))
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
    # clusterOPData.append(clusterOrderParameter(gg.clusters, theta, 0))

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
        # clusterOPData.append(clusterOrderParameter(gg.clusters, theta, t+1))

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


def clusterOrderParameter(clusters, theta):
    result = np.zeros(len(clusters))
    for i in range(len(result)):
        result[i] = (1.0 / len(clusters[i])) * abs(sum(cmath.exp(complex(0, a)) for a in theta[clusters[i]]))
    return result


def averagedOP(start, end, OPData):
    return np.sum(OPData[start:end]) / (end - start)


def add_averaged_data():
    global standardOPData, localOPData, clusterOPData, averagedOPData, averagedLocalOPData, averagedClusterData
    averagedOPData.append(averagedOP(len(standardOPData) - steps // 10, len(standardOPData), standardOPData))
    averagedLocalOPData.append(averagedOP(len(localOPData) - steps // 10, len(localOPData), localOPData))
    averagedClusterData.append(
        [averagedOP(len(clusterOPData) - steps // 10, len(clusterOPData), np.array(clusterOPData)[:, i]) for i in
         range(len(gg.clusters))])


def runSim(AMList, natFreqs):
    global end_start, oParameterData, standardOPData, localOPData, clusterOPData, averagedOPData, averagedLocalOPData, averagedClusterData, inst_freqs, inst_phases
    # oParameterData, standardOPData, localOPData, averagedOPData, averagedLocalOPData, clusterOPData, averagedClusterData, inst_freqs, inst_phases =[],[], [], [] ,[] ,[] ,[] ,[], []

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
    global end_s, end_f, oParameterData, standardOPData, localOPData, clusterOPData, averagedOPData, averagedLocalOPData, averagedClusterData, inst_freqs, inst_phases
    # oParameterData, standardOPData, localOPData, averagedOPData, averagedLocalOPData, clusterOPData, averagedClusterData, inst_freqs, inst_phases =[],[], [], [] ,[] ,[] ,[] ,[], []

    init_phases, init_freqs = ICs
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
    return np.array(AList), np.array(gg.freqs)

def run_and_plot(graphNum, AList, ICs, a, M, freqs, ss_diffs_file, curves_file, end_state=None):
    # frequencies = getRandomDistribution(N,-frequencyBound,frequencyBound,random.uniform)
    OP, ta_OP, end_theta, end_freq = runSim2(AList, ICs, freqs, a, M)
    ss_diffs_file.write(str(graphNum) + "\t" + str(a) + "\t" + str(M) + "\t" + str(get_ss_diff(ta_OP)))
    ss_diffs_file.write("\n")
    gg.printArrayToFile(ta_OP, curves_file)
    plt.plot(ta_OP)
    if end_state is not None:
        end_state.append((end_theta, end_freq))


if __name__ == '__main__':
    OPs = []
    ta_OPs = []
    ss_diffs = []
    end_states = []

    path_ER = head_dir+'/ER Graphs 1000 edges/'
    path_ER_dense = head_dir+'/ER Graphs 2000 edges/'
    path_SA = head_dir+'/Laplace-Optimized 1000 edges/'
    path_TA_OPs = head_dir+'/Time-Averaged OPs/ER to SA and back/100 Transition Graphs/'
    path_final_states = head_dir+'/ICs/ER to SA to ER (1000, .3) final states/'
    path_misaligned = head_dir+'/Synchrony Misaligned 1000 edges/'
    path_modular = head_dir+'/Modular Graphs 500 edges (.9)/'
    path_dense_mod = head_dir+'/Modular Graphs 1000 edges/'
    path_ICs = head_dir+'/ICs/'
    misaligned_files = []
    IC_Files = []
    TA_OP_Files = []
    ER_Files = []
    ER_Dense_Files = []
    SA_Files = []
    mod_Files = []
    dense_mod_Files = []
    ICs = []
    ER_Graphs = []
    ER_Dense_Graphs = []
    SA_Graphs = []
    MA_Graphs = []
    mod_Graphs = []
    dense_mod_Graphs = []
    standard_ER = gg.readMatrixFromFile(open(head_dir+"/static ER.txt", "r"))
    for i in range(25):
        ER_Files.append(open(os.path.join(path_ER + str(i) + ".txt"), "r"))
        SA_Files.append(open(os.path.join(path_SA + str(i) + ".txt"), "r"))
        ICs.append(open(os.path.join(path_ICs + str(i) + ".txt"), "r"))

    #arg_1 = int(sys.argv[1]) - 1 #Job number, ranging from 0 to 255
    #arg_1 = 6387
    #network = arg_1 // 256
    # To save memory, only reading in one graph for now:
    ER.append(gg.readMatrixFromFile(ER_Files[0]))
    SA_Graphs.append(gg.readMatrixFromFile(SA_Files[0]))
# argument 1: which graphs to use. arg 2: which coupling. arg 3: num mass trials

    AList, freqs = get_AList(ER_Graphs[0], SA_Graphs[0], dens_const=True)
    ICs = get_ICs(ICs[0])



    coupling_vals = np.linspace(.2, .5, 16)
    M_vals = np.linspace(1, 2.5, 16)

    chosen_coupling = coupling_vals[(arg_1 % 256) // 16]
    chosen_mass = M_vals[arg_1 % 16]

    result = open("ss_diffs_2020.txt", "a")
    TA_OP_curve = open("TA_OP 2020 "+str(network)+" "+str(chosen_coupling)+" " + str(chosen_mass) +" " + ".txt", "w")

    run_and_plot(network, AList, ICs, chosen_coupling, chosen_mass, freqs, result, TA_OP_curve, end_state=None)
    result.close()
    TA_OP_curve.close()
    #plot(len(AList), 'black')



    # print(str(gg.current_milli_time() - time) + "\t HELLO?!")

    plt.show()
