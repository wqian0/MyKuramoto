from scipy.integrate import ode, odeint
import numpy as np
import cmath
import os
import sys
import GraphGenerator as gg
import random as rd
import small_world as sw

# head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"
head_dir = "/data/jux/bqqian/Kuramoto"

N, alpha, dt, frequencyBound, steps = gg.size, .05, .02, gg.freqBound, 50000
startDelay, endDelay = gg.numTransitionGraphs // 3,gg.numTransitionGraphs // 3

m, D = 2, 1

oParameterData, standardOPData, averagedOPData = [], [], []
inst_freqs, inst_phases, complex_OP, end_s, end_f = [], [], [], [], []
pairwise_heatmap = np.zeros((N, N))

def Kura(init, t, A, w_nat, a, M):
    theta, omega = init[:N, ], init[N:, ]
    delta_theta = np.subtract.outer(theta, theta)
    dot_w = (1/M) * (-D * omega + w_nat + a * np.einsum('ij,ji->i', A, np.sin(delta_theta)))
    return np.concatenate([omega, dot_w])


def runRK(A, phases0, w_nat, w0, a, M, time, communities = None):
    result = odeint(Kura, np.concatenate([phases0, w0]), time, args=(A, w_nat, a, M))
    theta, omega = result[:, :N], result[:, N:]
    for t in range(len(theta) - 1):
        complex_r = complex_OP(theta[t])
        standardOPData.append(abs(complex_r))
        #inst_phases.append(theta[t])
        #inst_freqs.append(omega[t])
    return theta, omega


def Jacobian(init, t, A, w_nat, a, M):
    theta, omega = init[:N, ], init[N:, ]
    block_11 = np.zeros((N, N))
    block_21 = np.identity(N)
    block_22 = -D / m * np.identity(N)
    block_12 = np.zeros((N, N))
    for i in range(N):
        block_12[i] = np.inner(A[i], np.cos(theta - theta[i]))
        block_12[i, i] = -np.sum(block_21[i])
    block_12 *= a / M
    return np.block([[block_11, block_12], [block_21, block_22]])
def orderParameter(theta, t):
    return (1.0 / N) * abs(sum(cmath.exp(complex(0, a)) for a in theta[:, t]))


def complexOrderParameter(theta, t):
    return (1.0 / N) * sum(cmath.exp(complex(0, a)) for a in theta[:, t])


def complex_OP(theta):
    return (1.0 / N) * sum(cmath.exp(complex(0, a)) for a in theta)


def averagedOP(start, end, OPData):
    return np.sum(OPData[start:end]) / (end - start)


def add_averaged_data(frac=1):
    global standardOPData, averagedOPData
    averagedOPData.append(averagedOP(len(standardOPData) - steps // frac + 1, len(standardOPData), standardOPData))

def get_ICs(f=None, removeMean=False):
    if f is not None:
        ICs = gg.readMatrixFromFile(f)
        if removeMean:
            return ICs[0] - np.mean(ICs[0]), ICs[1] - np.mean(ICs[1])
        else:
            return ICs[0], ICs[1]
    else:
        init_phases = np.array(gg.getRandomDistribution(N, -np.pi, np.pi, rd.uniform))
        init_freqs = np.array(gg.getRandomDistribution(N, -gg.freqBound, gg.freqBound))
        if removeMean:
            init_phases -= np.mean(init_phases)
            init_freqs -= np.mean(init_freqs)
        return init_phases, init_freqs


def runSim(AMList, ICs, natFreqs, a, M, comms = None):
    global end_s, end_f, oParameterData, standardOPData, averagedOPData, inst_freqs, inst_phases, pairwise_heatmap

    init_phases, init_freqs = ICs
    init_phases = np.array(init_phases)
    init_freqs = np.array(init_freqs)
    time = np.linspace(0, dt * steps, steps)
    theta_0, w_0 = runRK(AMList[0], init_phases, natFreqs, init_freqs, a, M, time, comms)
    endTheta, endW = theta_0[len(theta_0) - 1, :], w_0[len(w_0) - 1, :]
    add_averaged_data()
    for i in range(startDelay):
        theta, w = runRK(AMList[0], endTheta, natFreqs, endW, a, M, time, comms)
        endTheta, endW = theta[len(theta) - 1, :], w[len(w) - 1, :]
        if i == startDelay - 1:
            end_s.append((endTheta, endW))
        print(i)
        add_averaged_data()
    for i in range(1, len(AMList)):
        theta, w = runRK(AMList[i], endTheta, natFreqs, endW, a, M, time, comms)
        endTheta, endW = theta[len(theta) - 1, :], w[len(w) - 1, :]
        print(str(i))
        add_averaged_data()
        if i == len(AMList) // 2:
            fill_heatmap(theta, comms)
    for i in range(endDelay):
        theta, w = runRK(AMList[len(AMList) - 1], endTheta, natFreqs, endW, a, M, time, comms)
        endTheta, endW = theta[len(theta) - 1, :], w[len(w) - 1, :]
        if i == endDelay - 1:
            end_f.append((endTheta, endW))
        print(i)
        add_averaged_data()
    return standardOPData, averagedOPData, endTheta, endW, pairwise_heatmap

def get_cluster_OP(A, theta, t, communities):
    total = sum(1 / len(c) * abs(sum(cmath.exp(complex(0, a)) for a in theta[t, c])) for c in communities)
    return total / len(communities)

def cross_cluster_sync(A, theta, t, communities):
    total = 0
    for i in range(len(communities) - 1):
        for j in range(i + 1, len(communities)):
            comm_1 = (1 / len(communities[i]))*sum(cmath.exp(complex(0, a)) for a in theta[t, communities[i]])
            comm_2 = (1 / len(communities[j])) * sum(cmath.exp(complex(0, a)) for a in theta[t, communities[j]])
            total += .5 * abs(comm_1 + comm_2)
    return total / (len(communities) * (len(communities) - 1)/2)

def fill_heatmap(theta, communities):
    global pairwise_heatmap
    B = np.zeros((N, N))
    comms = []
    comms.append(communities[3])
    comms.append(communities[2])
    comms.append(communities[1])
    comms.append(communities[0])
    comms.append(communities[4])
    print(communities)
    np.random.shuffle(comms)
    theta_complex = np.exp(1j * np.mean(theta, axis = 0))
    B = .5 * abs(np.add.outer(theta_complex, theta_complex))
    index = 0
    print(B)
    sorted_indices = np.zeros(N, dtype = int)
    # pairwise_heatmap[:,:] = B
    for i in range(len(comms)):
        for j in range(len(comms[i])):
            sorted_indices[index] = comms[i][j]
            index += 1
    index = 0
    for i in range(len(comms)):
        for j in range(len(comms[i])):
            pairwise_heatmap[index] = (B[comms[i][j]])[sorted_indices]
            pairwise_heatmap[:, index] = pairwise_heatmap[index]
            index += 1
    print(pairwise_heatmap)

def get_phase_diffs(phases):
    result = np.zeros((N, N))
    for r in range(N - 1):
        for c in range(r + 1, N):
            result[r][c] = phases[r] - phases[c]
            result[c][r] = -result[r][c]
    return result


def get_ss_diff(ta_OP):
    return ta_OP[len(ta_OP) - 1] - ta_OP[startDelay]


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


def run(numTrials, AList, ICs, a, M, freqs, ss_diffs_file, ta_OP_file, hm_file = None, end_state=None):
    for i in range(numTrials):
        OP, ta_OP,  end_theta, end_freq, pairwise_map = runSim(AList, ICs, freqs, a, M)
        ss_diffs_file.write(str(0) + "\t" + str(a) + "\t" + str(M) + "\t" + str(get_ss_diff(ta_OP)))
        ss_diffs_file.write("\n")
        gg.printArrayToFile(ta_OP, ta_OP_file)
        if hm_file is not None:
            gg.printMatrixToFile(pairwise_map, hm_file)
    if end_state is not None:
        for i in range(numTrials):
            end_state.append((end_theta, end_freq))


if __name__ == '__main__':

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

    OPs, ta_OPs, ss_diffs, end_states = [], [], [], []
    misaligned_files, IC_Files, TA_OP_Files, ER_Files, ER_Sparse_Files, ER_Dense_Files = [], [], [], [], [], []
    SA_Files, SA_Sparse_Files, mod_Files, freq_mod_Files, sparsefreq_mod_Files, dense_mod_Files  = [], [], [], [], [], []
    ICs, ER_Graphs, ER_Dense_Graphs, ER_Sparse_Graphs, SA_Graphs, SA_Sparse_Graphs = [], [], [], [], [], []
    MA_Graphs, mod_Graphs, freq_mod_Graphs, sparsefreq_mod_Graphs, dense_mod_Graphs = [], [], [], [], []

    for i in range(25):
        ER_Files.append(open(os.path.join(path_ER + str(i) + ".txt"), "r"))
        ER_Dense_Files.append(open(os.path.join(path_ER_dense + str(i) + ".txt"), "r"))
        ER_Sparse_Files.append(open(os.path.join(path_ER_sparse + str(i) + ".txt"), "r"))
        misaligned_files.append(open(os.path.join(path_misaligned) + str(i) + ".txt", "r"))
        # mod_Files.append(open(os.path.join(path_modular) + str(i) + ".txt", "r"))
        dense_mod_Files.append(open(os.path.join(path_dense_mod) + str(i) + ".txt", "r"))
        SA_Files.append(open(os.path.join(path_SA + str(i) + ".txt"), "r"))
        SA_Sparse_Files.append(open(os.path.join(path_SA_sparse+str(i)+".txt"),"r"))
        freq_mod_Files.append(open(os.path.join(path_freq_mod + str(i) + ".txt"), "r"))
        # sparsefreq_mod_Files.append(open(os.path.join(path_sparsefreq_mod+str(i)+".txt"), "r"))

    # arg_1 = int(sys.argv[1]) - 1 #Job number, ranging from 0 to 255
    arg_1 = 0

    dense_mod_Graphs.append(gg.readMatrixFromFile(dense_mod_Files[arg_1]))
    ER_Sparse_Graphs.append(gg.readMatrixFromFile(ER_Sparse_Files[arg_1]))
    SA_Sparse_Graphs.append(gg.readMatrixFromFile(SA_Sparse_Files[arg_1]))
    ER_Graphs.append(gg.readMatrixFromFile(ER_Files[arg_1]))
    SA_Graphs.append(gg.readMatrixFromFile((SA_Files[arg_1])))
    ER_Dense_Graphs.append(gg.readMatrixFromFile(ER_Dense_Files[arg_1]))
    # MA_Graphs.append(gg.readMatrixFromFile(misaligned_files[0]))
    freq_mod_Graphs.append(gg.readMatrixFromFile(freq_mod_Files[arg_1]))

    AList, freqs = get_AList(ER_Graphs[0], dense_mod_Graphs[0], dens_const=True)

    ICs = get_ICs()
    # freqs = gg.readMatrixFromFile(open(path_random_nat_freqs+str(arg_1)+".txt", "r"))[0]

    ss_diffs_file = open("blank.txt", "w")
    TA_OP_file = open("TA_OP_reg_modular_2020 "+str(arg_1)+".txt", "w")
    heatmap_file = open("reg_pairwise_heatmap " + str(arg_1) + ".txt", "w")
    run(1, AList, ICs, alpha, m, freqs, ss_diffs_file, TA_OP_file, end_state=end_states, hm_file= heatmap_file)
    ss_diffs_file.close()
    TA_OP_file.close()
    heatmap_file.close()

