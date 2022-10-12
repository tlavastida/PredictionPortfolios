# Quick implementations of K-median Heuristics for training the dual solutions

import random

from UtilityFunctions import median_duals

def l1_distance(p,q):
    V = p.keys()
    return sum(abs(p[v] - q[v]) for v in V)

# returns clusters and current assignment cost
def assign_to_clusters(points, centers, k):
    cost = 0
    clusters = {l:[] for l in range(k)}
    for p in points:
        best_idx = None
        best_dist = None
        for l in range(k):
            c = centers[l]
            dist = l1_distance(p, c)
            if best_idx is None or dist < best_dist:
                best_idx = l
                best_dist = dist
        
        clusters[best_idx].append(p)
        cost += best_dist
    return clusters, cost


# Runs a Lloyd's style algorithm for computing a k-median-ish clustering
# Uses a random initialization for the centers
def quick_and_dirty_k_median(points, k, tol=0.1, max_iter=100000):
    # random initialization
    random_indexes = random.sample(range(len(points)), k)
    centers = {l:points[random_indexes[l]] for l in range(k)}

    # assign points to clusters
    clusters, cost = assign_to_clusters(points, centers, k)
    #print("initial cost = " + str(cost))

    iteration = 0
    while iteration < max_iter:
        # recompute centers
        for l in range(k):
            centers[l] = median_duals(clusters[l])
        
        # reassign clusters
        clusters, new_cost = assign_to_clusters(points, centers, k)

        #print("new cost = " + str(new_cost))
        if abs(new_cost - cost) < tol:
            break
        cost = new_cost

    #print("final cost = " + str(cost))
    return centers, cost

# Runs swap-based local search for k-median clustering
# Uses a random initialization for the centers
def local_search_k_median(points, k, tol=0.1, max_iter=100000):

    # random initialization
    random_indexes = random.sample(range(len(points)), k)
    centers = {l:points[random_indexes[l]] for l in range(k)}

    # assign points to clusters
    clusters, cost = assign_to_clusters(points, centers, k)
    #print("initial cost = " + str(cost))

    iteration = 0
    while iteration < max_iter:
        improvement = 0
        for p in points:
            for l in range(k):
                # consider swapping p with centers[l]
                new_centers = centers
                new_centers[l] = p
                new_clusters, new_cost = assign_to_clusters(points, new_centers, k)

                if new_cost < cost:
                    improvement += cost - new_cost
                    cost = new_cost
                    centers = new_centers

        if improvement == 0:
            break

        
    #print("final cost = " + str(cost))
    return centers, cost

def assign_labels(points, centers, k):
    cost = 0
    labels = []

    for p in points:
        best_idx = None
        best_dist = None
        for l in range(k):
            c = centers[l]
            dist = l1_distance(p, c)
            if best_idx is None or dist < best_dist:
                best_idx = l
                best_dist = dist

        cost += best_dist        
        labels.append(best_idx)
    
    return labels, cost      
    
# Runs swap-based local search for k-median clustering
# Uses a random initialization for the centers
def local_search_k_median_bad(points, k, tol=0.1):
    # random initialization
    random_indexes = random.sample(range(len(points)), k)
    centers = {l:points[random_indexes[l]] for l in range(k)}

    # assign labels to points
    labels, cost = assign_labels(points, centers, k)
    #print("initial cost = " + str(cost))

    #print("initial labels = " + str(labels))

    while True:
        # Consider swaps
        best_improvement = None
        best_swap = None
        for i in range(len(points)):
            for j in range(i):
                if labels[i] != labels[j]:
                    #print(labels[i])
                    ci = centers[labels[i]]
                    p = points[i]
                    cj = centers[labels[j]]
                    q = points[j]
                    
                    improvement = (l1_distance(ci, q) + l1_distance(cj, p)) - (l1_distance(ci, p) + l1_distance(cj, q))
                    #print(improvement)
                    if best_improvement is None or improvement < best_improvement:
                        best_improvement = improvement
                        best_swap = (i,j)

        if best_improvement >= -1*tol:
            break

        print("best improvement = " + str(best_improvement))
        # perform the swap
        i,j = best_swap
        tmp = labels[i]
        labels[i] = labels[j]
        labels[j] = tmp

        # recompute centers
        for l in range(k):
            cluster = [points[i] for i in range(len(points)) if labels[i] == l]
            centers[l] = median_duals(cluster)

        # recompute labels
        labels, cost = assign_labels(points, centers, k)

    print("final cost = " + str(cost))
    return centers

# Runs swap-based local search for k-median clustering
# Uses a random initialization for the centers
def local_search_k_median_maybe(points, k, tol=0.1):

    # random initialization
    random_indexes = random.sample(range(len(points)), k)
    centers = {l:points[random_indexes[l]] for l in range(k)}

    # assign points to clusters
    clusters, cost = assign_to_clusters(points, centers, k)
    print("initial cost = " + str(cost))

    while True:
        # Consider swaps
        best_improvement = None
        best_swap = None
        for p in points:
            for l in range(k):
                # consider swapping p with centers[l]
                improvement = 0
                c = centers[l]
                for q in clusters[l]:
                    improvement += (l1_distance(p, q) - l1_distance(c, q))

                if best_improvement is None or improvement < best_improvement:
                    best_improvement = improvement
                    best_swap = (p, l)

        # perform the swap
        p, l = best_swap
        centers[l] = p

        # reassign clusters
        clusters, new_cost = assign_to_clusters(points, centers, k)

        if abs(new_cost - cost) < tol:
            break
        cost = new_cost


    print("final cost = " + str(cost))
    return centers

def KMedian(points, k, tol=0.1, method="local_search", restarts=10):
    # set the clustering method
    cluster_func = local_search_k_median if method == "local_search" else quick_and_dirty_k_median
    
    # repeat several times (since random inits are used for both)
    best_cost = None
    best_centers = None
    for t in range(restarts):
        centers, cost = cluster_func(points, k, tol)

        #print("Cost of restart # {} = {}".format(t,cost))
        if best_cost is None or cost < best_cost:
            best_centers = centers
            best_cost = cost

    centers = [best_centers[l] for l in range(k)]
    return centers
