import time
import random
import numpy as np
import numpy.random as np_random
import copy

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from UtilityFunctions import median_duals, train_duals

from BipartiteGraph import BipartiteGraph

from MinWeightPerfectMatching import MinWeightPerfectMatching, check_certificates, MinWeightPerfectMatchingMultipleDuals

from KMedian import KMedian, quick_and_dirty_k_median, local_search_k_median

from geometric_type_model_exp import generate_instance, setup_clusters, euclid_dist

def create_instance_gen(fname, num_points, n):
    L, R = setup_clusters(fname, num_points, n, 'A')

    def instance_gen():
        return generate_instance(L, R, n, euclid_dist)

    return instance_gen

def create_mixture_instance_gen(fname_list, num_points, n):
    k = len(fname_list)
    generator_list = []
    for fname in fname_list:
        generator_list.append(create_instance_gen(fname, num_points, n))
    
    def instance_gen():
        l = random.randrange(k)
        return generator_list[l](), l
    
    return instance_gen

# transfers dual_list to numpy arrays to pass to sklearn's kmeans
# runs k-means clustering, then extracts clusters and runs
# our prediction alg on each cluster
# returns a list of length k, each list entry is a dual solution
def kmeans_then_predict(dual_list, k):
    # get numpy arrays
    points = []
    for d in dual_list:
        x = np.array(list(d.values()))
        points.append(x)

    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    
    labels = KMeans(k).fit_predict(points)

    clusters = {l:[] for l in range(k)}

    for i in range(len(dual_list)):
        d = dual_list[i]
        ld = labels[i]
        if ld not in clusters.keys():
            print("Something is wrong....")
        else:
            clusters[ld].append(d)

    prediction_list = []
    for l in range(k):
        prediction_list.append(median_duals(clusters[l]))

    return prediction_list


# Use clustering to learn k initial dual solutions
# should have len(sample_list) == len(instance_gen_list)
def train_multiple_duals(sample_list, instance_gen_list, k_list, restarts, method="Lloyds"):
    i = 0
    dual_list = []
    true_k = len(sample_list)
    for l in range(true_k):
        samples = sample_list[l]
        instance_gen = instance_gen_list[l]
        for _ in range(samples):
            print('\tRunning training sample {}'.format(i))
            i += 1
            G = instance_gen()
            _, d = MinWeightPerfectMatching(G) # pylint: disable=unbalanced-tuple-unpacking
            dual_list.append(d)

    print("\tAggregating duals")

    fixed_duals = median_duals(dual_list)
    print("Finished computing fixed duals")

    multiple_duals = {}
    if method == "Lloyds":
        tol = 0.1
        method = "Lloyds"
        #restarts = 10
        for k in k_list:
            print("Computing duals for k = {}".format(k))
            multiple_duals[k] = KMedian(dual_list, k, tol, method, restarts)

    else:
        for k in k_list:
            print("Computing duals for k = {} with kmeans".format(k))
            multiple_duals[k] = kmeans_then_predict(dual_list, k)

    return fixed_duals, multiple_duals

# test the algorithms
# sample_list = list of number of samples to use for each instance type
# tests_list = list of number of tests to use for each instance type
# k_list = list of k values to test
# instance_gen_list = list of instance types to use
# should have len(sample_list) == len(instance_gen_list) == len(tests_list)
def test_algs(results_fname, dataset_list, sample_list, tests_list, restarts, k_list, instance_gen_list):
    # do the training step (and measure time)
    start = time.time()
    print("\tRunning the learning algorithms")
    fixed_duals, multiple_duals = train_multiple_duals(sample_list, instance_gen_list, k_list, restarts)
    stop = time.time()
    print('Training time: {}'.format(stop-start))

    # open the output file (CSV)
    outf = open(results_fname,'w')
    header = 'dataset, trial, std setup, std optimize, std iters, fixed_duals setup, fixed_duals optimize, fixed_duals iters'
    for k in k_list:
        header += ', k={}_duals setup, k={}_duals optimize, k={}_duals iters'.format(k, k, k)
    header += "\n"
    outf.write(header)

    true_k = len(sample_list)
    for l in range(true_k):
        dataset = dataset_list[l]
        print("\tRunning tests for dataset: {}".format(dataset))
        instance_gen = instance_gen_list[l]
        for t in range(tests_list[l]):
            print("\tRunning test {}".format(t))
            outf.write('{}, {}, '.format(dataset, t))
            TestG = instance_gen()

            # test the standard method
            M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, None, True, False)
            if not check_certificates(TestG, M, d):
                print('optimality check failed for standard')
            outf.write('{}, {}, {}, '.format(setup_elapsed, optimize_elapsed, iter_count))

            # test fixed predicted dual solution
            duals_to_use = copy.deepcopy(fixed_duals)
            M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, fixed_duals, True, True)
            if not check_certificates(TestG, M, d):
                print('optimality check failed for fixed duals')
            outf.write('{}, {}, {}'.format(setup_elapsed, optimize_elapsed, iter_count))

            # test each value of k
            for k in k_list:
                print('\ttesting k = {}'.format(k))
                duals_to_use = copy.deepcopy(multiple_duals[k])
                if len(duals_to_use) != k:
                    print("Error when running k = {}, wrong number of dual solutions".format(k))
                M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, duals_to_use)
                if not check_certificates(TestG, M, d):
                    print('optimality check failed for k = {} duals'.format(k))
                outf.write(',{}, {}, {}'.format(setup_elapsed, optimize_elapsed, iter_count))

            outf.flush()
            outf.write('\n')

    # close the output file after last test
    outf.close()

    print("All tests completed successfully")

def test_algs_no_clustering(s):
    # set up seeds
    random.seed(s)
    np_random.seed(s)

    # parameters and set up instances
    samples = 15
    tests = 5
    # number of points to sample
    num_points = 20000
    # number of nodes on each side per instance
    n = 150
    # filenames to use
    fname_list = ['kdd_sample.csv', 'Shuttle.txt', 'Skin100k.txt']

    # create each instance generator
    kdd_gen = create_instance_gen(fname_list[0], num_points, n)
    shuttle_gen = create_instance_gen(fname_list[1], num_points, n)
    skin_gen = create_instance_gen(fname_list[2], num_points, n)

    # do the training step (and measure time)
    start = time.time()
    
    # collect samples
    kdd_duals = []
    shuttle_duals = []
    skin_duals = []
    i = 0
    for _ in range(samples):
        print('Running training sample {}'.format(i))
        i += 1
        G = kdd_gen()
        _, d = MinWeightPerfectMatching(G)
        kdd_duals.append(d)

        print('Running training sample {}'.format(i))
        i += 1
        G = shuttle_gen()
        _, d = MinWeightPerfectMatching(G)
        shuttle_duals.append(d)

        print('Running training sample {}'.format(i))
        i += 1
        G = skin_gen()
        _, d = MinWeightPerfectMatching(G)
        skin_duals.append(d)

    print("Aggregating duals")
    # k = 1
    fixed_duals = median_duals(kdd_duals + shuttle_duals + skin_duals)
    # k = 2
    k_2_dual_list = KMedian(kdd_duals + shuttle_duals + skin_duals, 2, tol=0.01, method='lloyds', restarts=10)
    # k = 3
    k_3_dual_list = KMedian(kdd_duals + shuttle_duals + skin_duals, 3, tol=0.01, method='lloyds', restarts=10)
    #k_3_dual_list = [median_duals(kdd_duals), median_duals(shuttle_duals), median_duals(skin_duals)]
    k_4_dual_list = KMedian(kdd_duals + shuttle_duals + skin_duals, 4, tol=0.01, method='Lloyds', restarts=10)

    stop = time.time()
    print('Training time: {}'.format(stop-start))

    # run kdd tests
    print("Testing on kdd instances")
    for t in range(tests):
        print("\trunning test " + str(t))
        TestG = kdd_gen()

        # test the standard method
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, None, True, False)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for standard')
        print("\tStandard method iterations: {}".format(iter_count))

        # test k = 1
        duals_to_use = copy.deepcopy(fixed_duals)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, duals_to_use, True, True)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 1 duals iterations: {}".format(iter_count))

        # test k = 2
        dual_list_to_use = copy.deepcopy(k_2_dual_list)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, dual_list_to_use)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 2 duals iterations: {}".format(iter_count))

        # test k = 3
        dual_list_to_use = copy.deepcopy(k_3_dual_list)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, dual_list_to_use)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 3 duals iterations: {}".format(iter_count))

    # run shuttle tests
    print("Testing on shuttle instances")
    for t in range(tests):
        print("\trunning test " + str(t))
        TestG = shuttle_gen()

        # test the standard method
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, None, True, False)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for standard')
        print("\tStandard method iterations: {}".format(iter_count))

        # test k = 1
        duals_to_use = copy.deepcopy(fixed_duals)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, duals_to_use, True, True)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 1 duals iterations: {}".format(iter_count))

        # test k = 2
        dual_list_to_use = copy.deepcopy(k_2_dual_list)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, dual_list_to_use)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 2 duals iterations: {}".format(iter_count))

        # test k = 3
        dual_list_to_use = copy.deepcopy(k_3_dual_list)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, dual_list_to_use)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 3 duals iterations: {}".format(iter_count))

    # run skin tests
    print("Testing on skin instances")
    for t in range(tests):
        print("\trunning test " + str(t))
        TestG = kdd_gen()

        # test the standard method
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, None, True, False)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for standard')
        print("\tStandard method iterations: {}".format(iter_count))

        # test k = 1
        duals_to_use = copy.deepcopy(fixed_duals)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, duals_to_use, True, True)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 1 duals iterations: {}".format(iter_count))

        # test k = 2
        dual_list_to_use = copy.deepcopy(k_2_dual_list)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, dual_list_to_use)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 2 duals iterations: {}".format(iter_count))

        # test k = 3
        dual_list_to_use = copy.deepcopy(k_3_dual_list)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatchingMultipleDuals(TestG, dual_list_to_use)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tk = 3 duals iterations: {}".format(iter_count))



def run_exps(s):
    # set up seeds
    random.seed(s)
    np_random.seed(s)

    # true value of k
    true_k = 3

    # parameters
    sample_list = [20, 20, 20]
    tests_list = [10, 10, 10]
    k_list = [2, 3, 4, 5]

    # filenames to use
    fname_list = ['kdd_sample.csv', 'Shuttle.txt', 'Skin100k.txt']
    dataset_list = ['KDD', 'Shuttle', 'Skin']

    # restart parameter for training
    restarts = 5

    # number of points to sample
    num_points = 20000

    # number of nodes on each side per instance
    n = 150

    generator_list = []
    for fname in fname_list:
        print("Setting up instance generator for " + fname)
        generator_list.append(create_instance_gen(fname, num_points, n))

    # file name parameters
    DIR = "./Data/geometric_model_mixture/"
    EXP_NAME = "small_scale_kmeans_with_normalization"
    BODY = '_gmm_N_{}_n_{}'.format(num_points, n)

    results_fname = DIR + EXP_NAME + BODY + '.csv'

    test_algs(results_fname, dataset_list, sample_list, tests_list, restarts, k_list, generator_list)


def quick_test(s):
    # set up seeds
    random.seed(s)
    np_random.seed(s)

    # filenames to use
    fname_list = ['kdd_sample.csv', 'Shuttle.txt']
    # number of points to sample
    num_points = 20000
    # number of nodes on each side per instance
    n = 150
    generator_list = []
    for fname in fname_list:
        print("Setting up instance generator for " + fname)
        generator_list.append(create_instance_gen(fname, num_points, n))

    samples = 20
    tests = 10

    wrong_duals = train_duals(samples, generator_list[0])
    good_duals = train_duals(samples, generator_list[1])

    instance_gen = generator_list[1]

    for t in range(tests):
        print("running test " + str(t))
        TestG = instance_gen()

        # test the standard method
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, None, True, False)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for standard')
        print("\tStandard method iterations: {}".format(iter_count))

        # test fixed predicted dual solution
        duals_to_use = copy.deepcopy(wrong_duals)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, duals_to_use, True, True)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tWrong duals iterations: {}".format(iter_count))

        # test fixed predicted dual solution
        duals_to_use = copy.deepcopy(good_duals)
        M, d, setup_elapsed, optimize_elapsed, iter_count = MinWeightPerfectMatching(TestG, duals_to_use, True, True)
        if not check_certificates(TestG, M, d):
            print('optimality check failed for fixed duals')
        print("\tWrong duals iterations: {}".format(iter_count))
        




if __name__ == "__main__":
    run_exps(0)
    #quick_test(0)
    #test_algs_no_clustering(0)