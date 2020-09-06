
import numpy as np
from numpy import random as nrd
import random as rd
import copy
from random import choice
from array import *
import time
import os
import small_world as sw
import matplotlib as mpl
import scipy as sp
from scipy import stats
import multiprocessing as mp
import networkx as nx
from networkx.algorithms import community
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rc

rSeed = 1731
rd.seed(rSeed)
nrd.seed(rSeed)
freqs, size, dE, numTransitionGraphs, modules, pModular, finalEdges, freqBound = [], 100, 20, 50, 5, 0.98, 1000, 3
clusters = []

# head_dir = "/data/jux/bqqian/Kuramoto"
head_dir = "C:/Users/billy/PycharmProjects/Kuramoto"


def matrix(m, n, val):
    M = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(val)
        M.append(row)
    return np.array(M)

def getRandomDistribution(N, lowerBound, upperBound):
    list = [rd.uniform(lowerBound, upperBound) for i in range(N)]
    return list

def get_truncated_normal(mean, sd, low, upp):
    return stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def getNumEdges(M):
    return np.sum(M) / 2

def printMatrixToFile(M, file):
    for r in range(len(M)):
        for c in range(len(M[0])):
            file.write(str(M[r][c]) + "\t")
        file.write("\n")
    file.write("\n")


def printArrayToFile(A, file):
    for r in range(len(A)):
        file.write(str(A[r]) + "\t")
    file.write("\n")


def readMatrixFromFile(f):
    result = []
    for line in f:
        if not line.strip():
            continue
        line = [float(j) for j in line.strip().split('\t')]
        result.append(line)
    return np.array(result)


def getMatrixCopy(M):
    copy = matrix(len(M), len(M[0]), 0)
    for r in range(len(M)):
        for c in range(len(M[0])):
            copy[r][c] = M[r][c]
    return np.array(copy)

def get_laplacian(A):
    result = -getMatrixCopy(A)
    for i in range(len(result[0])):
        result[i][i] += np.sum(A[i])
    return result

def get_sync_alignment(n, freqs, A):
    L = get_laplacian(A)
    evals, v = sp.linalg.eigh(L, eigvals=(1, n - 1))
    sync_func = 0
    for i in range(len(evals)):
        if np.linalg.norm(v[:, i]) > 0:
            sync_func += ((1 / evals[i]) * np.dot(v[:, i] / np.linalg.norm(v[:, i]), freqs)) ** 2
    return sync_func

def get_random_laplaced(n, edges, frequencies, gAttempts, attempts, shared=None):
    diffs = np.array([a - np.mean(frequencies) for a in frequencies])
    diffs = diffs / np.linalg.norm(diffs)
    bestScore = -float('inf')
    for i in range(gAttempts):
        ER = get_random_graph(n, edges)
        val, currA = getBetterLapArrangement(n, ER, diffs, attempts)
        if val > bestScore:
            bestScore = val
            best = currA
        if shared is not None:
            shared.append(best)
    return best


def get_random_graph(n, edges):
    A = matrix(n, n, 0)
    possibleEdges = []
    for r in range(n - 1):
        for c in range(r + 1, n):
            possibleEdges.append((r, c))
    picked = nrd.choice(len(possibleEdges), edges, replace=False)
    for i in range(edges):
        A[possibleEdges[picked[i]][0]][possibleEdges[picked[i]][1]] = 1
        A[possibleEdges[picked[i]][1]][possibleEdges[picked[i]][0]] = 1
    return A


# p is probability the edge is forced to be within a module
def get_random_modular(n, modules, edges, p, getCommInfo=False, shared=None):
    pairings = {}
    assignments = np.zeros(n)
    for i in range(modules):
        pairings[i] = []
    adjMatrix = matrix(n, n, 0)
    for i in range(n):
        randomModule = rd.randint(0, modules)
        pairings[randomModule].append(i)
        assignments[i] = randomModule

    def add_modular_edge():
        randomComm = rd.randint(0, modules)
        selection = nrd.choice(pairings[randomComm], 2, replace=False)
        while adjMatrix[selection[0]][selection[1]] != 0:
            randomComm = rd.randint(0, modules)
            selection = nrd.choice(pairings[randomComm], 2, replace=False)
        #    print(str(selection[0])+" "+selection[1])
        adjMatrix[selection[0]][selection[1]] += 1
        adjMatrix[selection[1]][selection[0]] += 1

    def add_random_edge():
        randEdge = nrd.choice(n, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0 or assignments[randEdge[0]] == assignments[randEdge[1]]:
            randEdge = nrd.choice(n, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
        adjMatrix[randEdge[1]][randEdge[0]] += 1

    for i in range(int(edges)):
        if rd.uniform(0, 1) < p:
            add_modular_edge()
        else:
            add_random_edge()
    if shared is not None:
        shared.add(adjMatrix)
    if getCommInfo:
        return adjMatrix, pairings, assignments
    else:
        return adjMatrix

def rewire_to_endGraph(n, A_start, A_end, numGraphs, file):
    toRewire = []
    available = []
    diff = 0
    copy = getMatrixCopy(A_start)
    for r in range(n):
        for c in range(r, n):
            if A_start[r][c] == 1:
                if A_end[r][c] != 1:
                    toRewire.append((r, c))
                    toRewire.append((c, r))
            else:
                if A_end[r][c] == 1:
                    available.append((r, c))
                    available.append((c, r))
            if A_start[r][c] != A_end[r][c]:
                diff += 1
    print(str(diff) + "\t DIFF " + str(len(toRewire)))

    def rewire_edge():
        randIndex = rd.randrange(len(toRewire))
        remove = toRewire[randIndex]

        randIndex = rd.randrange(len(available))
        add = available[randIndex]

        copy[remove[0]][remove[1]] = 0
        copy[remove[1]][remove[0]] = 0
        copy[add[0]][add[1]] = 1
        copy[add[1]][add[0]] = 1

        toRewire.remove((remove[0], remove[1]))
        toRewire.remove((remove[1], remove[0]))
        available.remove((add[0], add[1]))
        available.remove((add[1], add[0]))

    totalEdges = len(available) // 2
    numPerGraph = totalEdges // numGraphs
    residue = totalEdges % numGraphs

    extras = nrd.choice(numGraphs, residue, replace=False)
    printMatrixToFile(A_start, file)
    for i in range(numGraphs):
        if i in extras:
            edgesGiven = numPerGraph + 1
        else:
            edgesGiven = numPerGraph
        for j in range(edgesGiven):
            rewire_edge()
        printMatrixToFile(copy, file)
        print(i)
    return copy


def rewire_randomly(n, A_start, numGraphs, numRewired, file):
    toRewire = []
    available = {}

    for i in range(n):
        available[i] = []

    copy = getMatrixCopy(A_start)
    for r in range(n):
        for c in range(r, n):
            if A_start[r][c] == 1:
                toRewire.append((r, c))
                toRewire.append((c, r))
            else:
                available[r].append(c)
                available[c].append(r)

    def rewire_edge():
        randIndex = rd.randrange(len(toRewire))
        randPair = toRewire[randIndex]
        while len(available[randPair[0]]) == 0 and len(available[randPair[1]]) == 0:
            toRewire.remove((randPair[0], randPair[1]))
            toRewire.remove((randPair[1], randPair[0]))
            randIndex = rd.randrange(len(toRewire))
            randPair = toRewire[randIndex]
        if len(available[randPair[0]]) > 0:
            tupleIndex = 0
        else:
            tupleIndex = 1

        randAvailableIndex = rd.randrange(len(available[randPair[tupleIndex]]))
        randAvailable = available[randPair[tupleIndex]][randAvailableIndex]

        copy[randPair[0]][randPair[1]] = 0
        copy[randPair[1]][randPair[0]] = 0
        copy[randPair[tupleIndex]][randAvailable] = 1
        copy[randAvailable][randPair[tupleIndex]] = 1
        toRewire.remove((randPair[0], randPair[1]))
        toRewire.remove((randPair[1], randPair[0]))
        available[randPair[tupleIndex]].remove(randAvailable)
        available[randAvailable].remove(randPair[tupleIndex])

    numPerGraph = numRewired // numGraphs
    residue = numRewired % numGraphs

    extras = nrd.choice(numGraphs, residue, replace=False)
    for i in range(numGraphs):
        if i in extras:
            edgesGiven = numPerGraph + 1
        else:
            edgesGiven = numPerGraph
        for j in range(edgesGiven):
            rewire_edge()
        printMatrixToFile(copy, file)
    return copy


def get_random_freq_modular(n, modules, edges, p, frequencies, rangeStart, rangeEnd, getCommInfo):
    pairings = {}
    assignments = np.zeros(n)
    for i in range(modules):
        pairings[i] = []
    adjMatrix = matrix(n, n, 0)
    for i in range(n):
        module = int(np.round((frequencies[i] - rangeStart) / (rangeEnd - rangeStart) * (modules - 1)))
        pairings[module].append(i)
        assignments[i] = module

    def add_modular_edge():
        randomComm = rd.randint(0, modules - 1)
        selection = nrd.choice(pairings[randomComm], 2, replace=False)
        while adjMatrix[selection[0]][selection[1]] != 0:
            randomComm = rd.randint(0, modules - 1)
            selection = nrd.choice(pairings[randomComm], 2, replace=False)
        adjMatrix[selection[0]][selection[1]] += 1
        adjMatrix[selection[1]][selection[0]] += 1

    def add_random_edge():
        randEdge = nrd.choice(n, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = nrd.choice(n, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
        adjMatrix[randEdge[1]][randEdge[0]] += 1

    for i in range(int(edges)):
        # print(i)
        if rd.uniform(0, 1) < p:
            add_modular_edge()
        else:
            add_random_edge()
    if getCommInfo:
        return adjMatrix, pairings, assignments
    else:
        return adjMatrix

def getModularity(A, modules):
    result = 0
    e_ii = 0
    a_i = 0
    numEdges = getNumEdges(A)

    for i in range(len(modules)):
        for j in modules[i]:
            for k in modules[i]:
                if A[j][k] == 1:
                    e_ii += 1
        e_ii /= 2 * numEdges
        for j in modules[i]:
            a_i += sum(A[j])
        a_i /= 2 * numEdges
        result += e_ii - a_i ** 2
        e_ii = 0
        a_i = 0
    return result


def flipEntry(aMat, target, r, c):
    aMat[r][c] = target[r][c]
    aMat[c][r] = target[c][r]

def rearrangeMatrix(A, rearrangements):
    B = np.array(A)
    for i in range(rearrangements):
        randIndex1 = rd.randint(0, len(B) - 1)
        randIndex2 = rd.randint(0, len(B) - 1)
        tempRow = copy.deepcopy(B[randIndex1])
        B[randIndex1] = copy.deepcopy(B[randIndex2])
        B[randIndex2] = tempRow
        tempCol = copy.deepcopy(B[:, randIndex1])
        B[:, randIndex1] = copy.deepcopy(B[:, randIndex2])
        B[:, randIndex2] = tempCol
    return B


def flipEdge(n, A):
    B = getMatrixCopy(A)
    edges = []
    empty = []
    for r in range(n - 1):
        for c in range(r + 1, n):
            if B[r][c] == 1:
                edges.append((r, c))
            else:
                empty.append((r, c))
    rE = rd.randrange(len(edges))
    rA = rd.randrange(len(empty))
    B[edges[rE][0]][edges[rE][1]] = 0
    B[edges[rE][1]][edges[rE][0]] = 0
    B[empty[rA][0]][empty[rA][1]] = 1
    B[empty[rA][1]][empty[rA][0]] = 1
    return B


def getBetterLapArrangement(n, A, freq_vec, attempts):
    currScore = 0
    maxVal = -float('inf')
    curr = getMatrixCopy(A)
    best = curr
    for i in range(attempts):
        evals, v = sp.linalg.eigh(get_laplacian(curr), eigvals=(n - 1, n - 1))
        currScore = get_sync_alignment(n, freq_vec, curr)
        if currScore > maxVal:
            maxVal = currScore
            best = curr
        curr = flipEdge(n, best)
    return maxVal, best


def getBetterArrangement(n, A_start, A_end, attempts):
    count = 0
    minCount = float('inf')
    copy = np.array(getMatrixCopy(A_end))
    best = getMatrixCopy(copy)
    for i in range(attempts):
        for r in range(n):
            for c in range(r, n):
                if A_start[r][c] != copy[r][c]:
                    count += 1
        if count < minCount:
            minCount = count
            best = getMatrixCopy(copy)
        copy = rearrangeMatrix(best, 1)
        count = 0
        print(minCount)
    return best


def getWorstArrangement(n, A_start, A_end, attempts):
    count = 0
    maxCount = 0
    copy = np.array(getMatrixCopy(A_end))
    best = getMatrixCopy(copy)
    for i in range(attempts):
        for r in range(n):
            for c in range(r, n):
                if A_start[r][c] != copy[r][c]:
                    count += 1
        if count > maxCount:
            maxCount = count
            best = getMatrixCopy(copy)
        copy = rearrangeMatrix(best, 1)
        count = 0
        print(maxCount)
    return best


# numGraphs includes the end graph, but not the start graph
def get_inbetween_matrices(n, A_start, A_end, numGraphs, file):
    delete = []
    add = []
    copy = getMatrixCopy(A_start)
    copyEnd = getMatrixCopy(A_end)
    for r in range(n):
        for c in range(r, n):
            if A_start[r][c] == 1 and copyEnd[r][c] == 0:
                delete.append((r, c))
            if A_start[r][c] == 0 and copyEnd[r][c] == 1:
                add.append((r, c))

    numAddedPerGraph = len(add) // numGraphs
    residue_add = len(add) % numGraphs
    extra_add = nrd.choice(numGraphs, residue_add, replace=False)

    numDelPerGraph = len(delete) // numGraphs
    residue_del = len(delete) % numGraphs
    extra_del = nrd.choice(numGraphs, residue_del, replace=False)
    printMatrixToFile(copy, file)
    for i in range(numGraphs):
        if i in extra_add:
            edgesGiven = numAddedPerGraph + 1
        else:
            edgesGiven = numAddedPerGraph
        if i in extra_del:
            edgesRemoved = numDelPerGraph + 1
        else:
            edgesRemoved = numDelPerGraph
        for j in range(edgesGiven):
            randPair = add[rd.randint(0, len(add) - 1)]
            flipEntry(copy, copyEnd, randPair[0], randPair[1])
            add.remove(randPair)
        for j in range(edgesRemoved):
            randPair = delete[rd.randint(0, len(delete) - 1)]
            flipEntry(copy, copyEnd, randPair[0], randPair[1])
            delete.remove(randPair)
        printMatrixToFile(copy, file)
        print(str(getNumEdges(copy)) + "\t edge-check", i)

def make_grid(rows, cols):
    n = rows * cols
    M = matrix(n, n, 0)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            # Two inner diagonals
            if c > 0:
                M[i - 1][i] = M[i][i - 1] = 1
            # Two outer diagonals
            if r > 0:
                M[i - cols][i] = M[i][i - cols] = 1
    return M
def addEdges(M, edges):
    copy = getMatrixCopy(M)
    for i in range(edges):
        randomRow = rd.randint(0, len(copy) - 1)
        randomCol = rd.randint(0, len(copy[0]) - 1)
        while randomRow == randomCol or copy[randomRow][randomCol] == 1:
            randomCol = rd.randint(0, len(copy[0]) - 1)
            randomRow = rd.randint(0, len(copy) - 1)
        copy[randomRow][randomCol] += 1
        copy[randomCol][randomRow] += 1
    return copy

def get_communities(A):
    G = nx.from_numpy_matrix(A)
    # first compute the best partition
    communities = sorted(community.greedy_modularity_communities(G), key=len, reverse=True)

    comm_mapping = np.zeros(size)
    for i in range(len(communities)):
        communities[i] = list(communities[i])
    for i in range(len(communities)):
        for j in range(len(communities[i])):
            comm_mapping[communities[i][j]] = i
    return communities, comm_mapping

def main(transitions, start=None, end=None, final=None, dens_const=True):
    global freqs, clusters
    f = open("adjacency matrices.txt", "w")
    f2 = open(head_dir+"/freqs.txt", "r")
    freqs = readMatrixFromFile(f2)[0]
    #freqs = get_truncated_normal(0, 2, -3, 3).rvs(100)
    print(freqs)
    if start is not None and final is None:
        if dens_const:
            rewire_to_endGraph(size, start, end, transitions, f)
        else:
            get_inbetween_matrices(size, start, end, transitions, f)
    elif start is not None and final is not None:
        if dens_const:
            rewire_to_endGraph(size, start, end, transitions, f)
            rewire_to_endGraph(size, end, final, transitions, f)
        else:
            get_inbetween_matrices(size, start, end, transitions, f)
            get_inbetween_matrices(size, end, final, transitions, f)
    else:
        if dens_const:
            rewire_randomly(size, make_grid(10, 10), transitions, 180, f)
        else:
            get_inbetween_matrices(size, start, end, transitions, f)
        # printMatrixToFile(ER_1k, f2)


if __name__ == "__main__":
    main(numTransitionGraphs)
    f2 = open(head_dir + "/freqs.txt", "r")
    freqs = readMatrixFromFile(f2)[0]
    path_dense_mod = head_dir + '/Modular Graphs 1000 edges/'
    path_freq_mod = head_dir + '/Frequency Modular 1000/'
    mod_network = readMatrixFromFile(open(os.path.join(path_dense_mod) + str(0) + ".txt", "r"))
    freq_mod_network = readMatrixFromFile(open(os.path.join(path_freq_mod) + str(0) + ".txt", "r"))
    G = nx.from_numpy_matrix(freq_mod_network)
    # first compute the best partition
    communities = sorted(community.greedy_modularity_communities(G), key=len, reverse=True)

    comm_mapping = np.zeros(size)
    for i in range(len(communities)):
        communities[i] = list(communities[i])
    print(communities)
    for i in range(len(communities)):
        for j in range(len(communities[i])):
            comm_mapping[communities[i][j]] = i + 1
    # draw the graph
    print(communities)
    x = np.ones((size, 3))
    y = np.ones((size, 3))

    x[:, 0:3] = (1, 0, 0)
    y[:, 0:3] = (0, 1, 0)
    c = np.linspace(0, 1, size)[:, None]
    gradient = x + (y - x) * c
    sorted_indices = np.argsort(freqs)
    col_val = np.zeros((size, 3))
    cmap = plt.get_cmap("RdBu_r")
    new_cmap = ListedColormap(gradient, name = 'frequencies')
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])
    for i in range(size):
        index = np.where(sorted_indices == i)
        col_val[i] = gradient[index]
    fig = plt.figure(dpi = 3000)
    plt.box(False)
    print(G.nodes)

    nodecolor = [cmap(freqs[i]) for i in range(100)]
    print(freqs)
    pos = nx.spring_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=100, node_color= col_val, linewidths = .5)
    nodes.set_edgecolor('black')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width = .5)
    cbar = plt.colorbar(sm, ticks=np.linspace(-3, 3, 5), fraction=0.046, pad=0.04)
    cbar.set_label(r'$\omega _i$', labelpad=-1, fontsize = 22)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig('Freq Modular Network.pdf')
    #plt.show()

