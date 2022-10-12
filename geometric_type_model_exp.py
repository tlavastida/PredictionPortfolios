# code for geometric type model experiment
# instances based on clustering

import random
import numpy.random as np_random
import numpy as np 
import time
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from BipartiteGraph import BipartiteGraph
from MinWeightPerfectMatching import MinWeightPerfectMatching, check_certificates
from UtilityFunctions import generate_newline_locations, load_random_lines, train_duals, JaccardMatching

# we have point sets organized as follows:
# L = L1 + L2 + ... + Lk
# R = R1 + R2 + ... + Rk

# To generate U we sample k points from L, one from each Li
# To generate V we sample k points from R, one from each Ri
# for x in U and y in V, w(x,y) = d(x,y) where d is some 
# distance function, e.g. euclidean.


# L/R are dictionaries with range(k) as keys
# L[i] = list of d-dimensional points
# k is a non-neg integer
# dist is a distance function
# outputs an instance according to the specification above
def generate_instance(L, R, k, dist):

    U = range(k)
    V = range(k,2*k)

    U_pts = []
    V_pts = []

    # sample from each side
    for i in range(k):
        U_pts.append(random.choice(L[i]))
        V_pts.append(random.choice(R[i]))


    E = []
    W = []
    for u in U:
        for v in V:
            E.append((u, v))
            # v is in [k,2*k), subtract k to put it in [0,k)
            w = int(dist(U_pts[u], V_pts[v-k])) # round to an integer arbitrarily
            W.append(w)

    return BipartiteGraph(U, V, E, W)


def euclid_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def taxicab_dist(x, y):
    return np.sum( np.abs(x-y) )

# returns k-means clustering of data
# returns partition in form of dictionary
# keys are integers 0,1,2,...,k-1
# values are lists of points w/ same cluster label
def kmeans(data, k):
    labels = KMeans(k).fit_predict(data)

    clusters = {i:[] for i in range(k)}

    for i in range(len(data)):
        x = data[i]
        lx = labels[i]
        if lx not in clusters.keys():
            print("Something is wrong....")
        else:
            clusters[lx].append(x)
    
    return clusters

# n is the number of points to load
# k is the number of clusters/nodes on each side
def setup_clusters(fname, n, k, method='A'):

    left_clusters = None
    right_clusters = None

    newline_locs = generate_newline_locations(fname)
    data = load_random_lines(fname,newline_locs,n)

    # method A
    if method == "A":
        left = data[:(n//2)]
        right = data[(n//2):]

        left_clusters = kmeans(left,k)
        right_clusters = kmeans(right,k)

    # method B
    else: 
        clusters = kmeans(data,2*k)

        # split into left and right randomly
        order = list(np_random.permutation(2*k))

        left_clusters = {i:None for i in range(k)}
        right_clusters = {i:None for i in range(k)}

        for i in range(k):

            left_clusters[i] = clusters[order[i]] 
            right_clusters[i] = clusters[order[i+k]]

    return left_clusters, right_clusters



def run_exps_old(s,fname,results_fname,method):
    # pylint: disable=unbalanced-tuple-unpacking
    random.seed(0)
    np_random.seed(0)

    # open the output file (CSV)
    outf = open(results_fname,'w')
    header = 'n, Trial, Method, Setup, Optimize, Runtime, Iters\n'
    outf.write(header)

    samples = 20
    tests = 10
    
    k = 500

    dist = euclid_dist

    n_list = [ 1000*j for j in range(1,41)]
    for n in n_list:
        print("Starting n = {}".format(n),flush=True)
        L,R = setup_clusters(fname,n,k,method)

        def instance_gen():
            return generate_instance(L,R,k,dist)

        # do the training step
        start = time.time()
        learned_duals = train_duals(samples,instance_gen)
        stop = time.time()
        print('Training time: {}'.format(stop-start),flush=True)
        
        opt_solutions = []
        for t in range(tests):
            print("Running test {}".format(t),flush=True)

            TestG = instance_gen()

            # test the standard method
            M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,None,True,False)

            if not check_certificates(TestG,M,d):
                print('optimality check failed for standard')

            opt_solutions.append(M)

            runtime = setup_elapsed + optimize_elapsed
            line = '{}, {}, {}, {}, {}, {}, {}\n'.format(n,t,'Hungarian',setup_elapsed,optimize_elapsed,runtime,iter_count)
            outf.write(line)


            #  test learned duals
            M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,learned_duals,True,True)

            if not check_certificates(TestG,M,d):
                print('optimality check failed for learned duals')

            runtime = setup_elapsed + optimize_elapsed
            line = '{}, {}, {}, {}, {}, {}, {}\n'.format(n,t,'Learned Duals',setup_elapsed,optimize_elapsed,runtime,iter_count)
            outf.write(line)

            
    outf.close()
    print("Done with experiments",flush=True)

# fix n and vary k
def run_exps(s,n,fname,results_fname,method):
    # pylint: disable=unbalanced-tuple-unpacking
    random.seed(0)
    np_random.seed(0)

    # open the output file (CSV)
    outf = open(results_fname,'w')
    header = 'n,k,Test,Method,Setup,Optimize,Runtime,Iters\n'
    outf.write(header)

    samples = 20
    tests = 10
    
    

    dist = euclid_dist

    k_list = [100*i for i in range(1,10+1)]
    for k in k_list:
        print("Starting k = {}".format(k),flush=True)
        L,R = setup_clusters(fname,n,k,method)

        def instance_gen():
            return generate_instance(L,R,k,dist)

        # do the training step
        start = time.time()
        learned_duals = train_duals(samples,instance_gen)
        stop = time.time()
        print('Training time: {}'.format(stop-start),flush=True)
        
        opt_solutions = []
        for t in range(tests):
            print("Running test {}".format(t),flush=True)

            TestG = instance_gen()

            # test the standard method
            M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,None,True,False)

            if not check_certificates(TestG,M,d):
                print('optimality check failed for standard')

            opt_solutions.append(M)

            runtime = setup_elapsed + optimize_elapsed
            line = '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(n,k,t,'Hungarian',setup_elapsed,optimize_elapsed,runtime,iter_count)
            outf.write(line)


            #  test learned duals
            M,d,setup_elapsed,optimize_elapsed,iter_count = MinWeightPerfectMatching(TestG,learned_duals,True,True)

            if not check_certificates(TestG,M,d):
                print('optimality check failed for learned duals')

            runtime = setup_elapsed + optimize_elapsed
            line = '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(n,k,t,'Learned Duals',setup_elapsed,optimize_elapsed,runtime,iter_count)
            outf.write(line)

        # compute average jaccard similarity
        denom = 0
        tot = 0
        for M1,M2 in combinations(opt_solutions,2):
            tot += JaccardMatching(M1,M2)
            denom += 1
    
        print("Average Jaccard: {}".format(tot/denom),flush=True)
            
    outf.close()
    print("Done with experiments",flush=True)


def quick_test():
    # pylint: disable=unbalanced-tuple-unpacking

    random.seed(0)
    np_random.seed(0)

    fname = "Skin100k.txt"

    n = 10000
    k = 200

    dist = euclid_dist

    L,R = setup_clusters(fname,n,k,method='B')

    cluster_sizes = [len(L[i]) for i in L.keys()] + [len(R[i]) for i in R.keys()]
    sns.histplot(data=cluster_sizes)
    plt.show()

    G = generate_instance(L,R,k,dist)
    print(G)
    M,d = MinWeightPerfectMatching(G,None,False,True) 
    print(check_certificates(G,M,d))

    G = generate_instance(L,R,k,dist)
    print(G)
    M,d = MinWeightPerfectMatching(G,None,False,True)
    print(check_certificates(G,M,d))

    G = generate_instance(L,R,k,dist)
    print(G)
    M,d = MinWeightPerfectMatching(G,None,False,True)
    print(check_certificates(G,M,d))

        
if __name__ == "__main__":

    #quick_test()

    # input the file to read from here
    
    #fname = 'Shuttle.txt'
    #results_fname = '../Data/clustering_type_model_new/results_shuttle_method_B.csv'
    #n = 43500
    #method = 'B'
    
    #fname = "blogData_train.csv"
    #results_fname = "../Data/clustering_type_model_new/results_blogfeedback_method_B.csv"
    #n = 52397
    #method = 'B'

    fname = 'kdd_sample.csv'
    results_fname = '../Data/clustering_type_model_new/results_kdd_method_B.csv'
    n = 98942
    method = 'B'
    
    run_exps(0,n,fname,results_fname,method)

    






