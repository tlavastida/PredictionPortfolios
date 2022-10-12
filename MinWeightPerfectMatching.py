
# minimum weight perfect matching via the hungarian algorithm
# also some stuff for warm starting it with any set of feasible duals

import time

from BipartiteGraph import BipartiteGraph
from MaxBipartiteMatching import GreedyMatching,HopcroftKarp,MatchingToAugmentingSets


# define infinity
INF = float("inf")


# returns true if M is a perfect matching on G and false otherwise
def check_matching(G,M):
    ret = True
    # first check that M is large enough
    if len(M) != len(G.U):
        print('Matching is too small')
        ret = False

    # check that M is a matching
    covered_verts = set()
    for e in M:
        u,v = e
        # check that each vertex is matched only once
        if u in covered_verts or v in covered_verts:
            print('Matching covers a vertex twice')
            ret = False
        
        covered_verts.add(u)
        covered_verts.add(v)

        if u not in G.Adj[v] or v not in G.Adj[u]:
            print('Matching chooses a non-edge')
            ret = False
    
    return ret


# returns true if duals are feasible for G and 0 otherwise
def check_duals(G,duals):
    ret = True
    for e in G.E:
        u,v = e
        if duals[u] + duals[v] > G.W[e]:
            print('infeasible edge: {} + {} > {}'.format(duals[u],duals[v],G.W[e]))
            ret = False
    return ret

# checks that M,d is a certificate of optimality in bipartite graph G
# returns true if M,d certify optimal solutions and false otherwise
def check_certificates(G,M,d):
    ret = True
    if not check_matching(G,M) or not check_duals(G,d):
        print('One of M or d is an infeasible solution')
        ret = False
    else:
        primal_obj = G.matching_weight(M)
        dual_obj = sum(d[v] for v in d.keys())
        if abs(primal_obj - dual_obj) > 1/100:
            print('primal != dual, gap = primal_obj - dual_obj')
            ret = False
    return ret

# Initializes a set of feasible duals greedily
def GreedyInitialDuals(G):

    duals = {v:0 for v in G.AllVerts}
    # duals[u] + duals[v] <= W[(u,v)] for all edges (u,v)
    # thus if duals[v] = 0 for all v in G.V
    # can set duals[u] to min weight in its neighborhood
    for u in G.U:
        duals[u] = min(G.W[(u,v)] for v in G.Adj[u])
    
    return duals

# G is of type BipartiteGraph
def ProjectDuals(G,duals):
    # initial perturbation vector    
    delta = {u:0 for u in G.AllVerts}

    # set of unvisited vertices
    unvisited_verts = set(G.AllVerts)

    # set of visited edges
    visited_edges = set()

    # set delta greedily
    # starting from some start vertex u
    u = unvisited_verts.pop()
    unvisited_verts.add(u)

    while len(unvisited_verts) > 0:

        # iterate over the edges in u's neighborhood
        max_pseudoweight = None
        next_endpoint = None
        for v in G.Adj[u]:
            if (u,v) not in visited_edges and (v,u) not in visited_edges:
                pseudoweight = duals[u] + duals[v] - G.W[(u,v)]

                if pseudoweight > 0:

                    # update max_pseudoweight and next_endpoint
                    if max_pseudoweight is None:
                        max_pseudoweight = pseudoweight
                        next_endpoint = v
                    else:
                        next_endpoint = v if pseudoweight > max_pseudoweight else next_endpoint
                        max_pseudoweight = max(pseudoweight,max_pseudoweight)

            # mark this edge as visited
            visited_edges.add((u,v))
        
        # set perturbation and set up for next iteration
        if max_pseudoweight is not None and next_endpoint is not None:
            # set u's perturbation
            delta[u] = max_pseudoweight

            if u in unvisited_verts:
                unvisited_verts.remove(u)
            
            u = next_endpoint
        else:
            u = unvisited_verts.pop()

    # add the perturbations to get the labels
    for u in G.AllVerts:
        duals[u] -= delta[u]

    return duals

# duals are assumed to be feasible,
# i.e. duals[u] + duals[v] <= G.W[(u,v)] for all edges (u,v)
def GreedyTightenDuals(G,duals):
    for u in G.U:
        min_slack = None
        for v in G.Adj[u]:
            slack = G.W[(u,v)] - duals[u] - duals[v]
            if slack < 0:
                raise ValueError("duals are infeasible")
            else:
                min_slack = slack if min_slack is None or slack < min_slack else min_slack
        
        duals[u] += min_slack
    
    return duals


# MWPM
# G is of class Bipartite Graph
# duals is a dictionary defined on G.AllVerts
# It is assumed that there is a matching covering all of G.U in G
def MinWeightPerfectMatching(G,duals=None,measure_perf=False,tighten=False):
    if measure_perf:
        start_time = time.time()

    if G.W is None:
        raise ValueError("G must be a weighted bipartite graph")

    # initialize the dual variables
    if duals is None:
        duals = GreedyInitialDuals(G)
    else:
        duals = ProjectDuals(G,duals)
        duals = GreedyTightenDuals(G,duals) if tighten else duals
        
    # initial matching
    U_matches = {v: None for v in G.U}
    V_matches = {v: None for v in G.V}

    if measure_perf:
        stop_time = time.time()
        setup_elapsed = stop_time - start_time

        iteration_count = 0
        start_time = time.time()

    # assume that U is the smaller side
    # i.e. there exists a matching covering all of U in G
    full_size = len(G.U)

    while True:

        if measure_perf:
            iteration_count += 1

        # generate the dual-tight subgraph
        E_tight = [e for e in G.E if duals[e[0]] + duals[e[1]] == G.W[e]]
        H = BipartiteGraph(G.U,G.V,E_tight)

        # find the maximum matching
        U_matches,V_matches = GreedyMatching(H,U_matches,V_matches)  #this step seems to speed things up a bit and its a simple optimization that always works
        U_matches,V_matches = HopcroftKarp(H,False,U_matches,V_matches,False)

        matching_size = sum(1 if U_matches[u] is not None else 0 for u in G.U)

        if matching_size < full_size:

            LU,LV = MatchingToAugmentingSets(H,U_matches,V_matches)

            #cover_size = len(G.U) - len(LU) + len(LV)
            #print(matching_size)#,cover_size)

            def _f(u,v):
                return INF if (u,v) not in G.E else G.W[(u,v)] - duals[u] - duals[v]


            epsilon = min(_f(u,v) for u in LU for v in G.V - LV)

            for u in LU:
                duals[u] += epsilon
        
            for v in LV:
                duals[v] -= epsilon

        else:
            break

    # get the final matching
    M = set((u,U_matches[u]) for u in G.U)

    if measure_perf:
        stop_time = time.time()
        optimize_elapsed = stop_time - start_time

        return M, duals, setup_elapsed, optimize_elapsed, iteration_count
    else:
        return M, duals


# MWMP with multiple initial duals
# dual_list is a list of dual variable assignments
def MinWeightPerfectMatchingMultipleDuals(G, dual_list):
    if G.W is None:
        raise ValueError("G must be a weighted bipartite graph")

    # start measuring setup time
    start_time = time.time()

    # Initialize dual variables
    k = len(dual_list)
    duals = None
    best_val = None
    for d in dual_list:
        d = GreedyTightenDuals(G, ProjectDuals(G, d))
        val = sum([d[v] for v in d.keys()])
        if duals is None or val > best_val:
            duals = d
            best_val = val
    

    # initial matching
    U_matches = {v: None for v in G.U}
    V_matches = {v: None for v in G.V}

    # compute setup time
    stop_time = time.time()
    setup_elapsed = stop_time - start_time

    # setup for optimization step and measuring optimize time
    iteration_count = 0
    start_time = time.time()

    # assume that U is the smaller side
    # i.e. there exists a matching covering all of U in G
    full_size = len(G.U)

    while True:

        iteration_count += 1

        # generate the dual-tight subgraph
        E_tight = [e for e in G.E if duals[e[0]] + duals[e[1]] == G.W[e]]
        H = BipartiteGraph(G.U,G.V,E_tight)

        # find the maximum matching
        U_matches,V_matches = GreedyMatching(H,U_matches,V_matches)  #this step seems to speed things up a bit and its a simple optimization that always works
        U_matches,V_matches = HopcroftKarp(H,False,U_matches,V_matches,False)

        matching_size = sum(1 if U_matches[u] is not None else 0 for u in G.U)

        if matching_size < full_size:

            LU,LV = MatchingToAugmentingSets(H,U_matches,V_matches)

            def _f(u,v):
                return INF if (u,v) not in G.E else G.W[(u,v)] - duals[u] - duals[v]

            epsilon = min(_f(u,v) for u in LU for v in G.V - LV)

            for u in LU:
                duals[u] += epsilon
        
            for v in LV:
                duals[v] -= epsilon

        else:
            break

    # get the final matching
    M = set((u,U_matches[u]) for u in G.U)

    # compute times
    stop_time = time.time()
    optimize_elapsed = stop_time - start_time
    
    return M, duals, setup_elapsed, optimize_elapsed, iteration_count



        

        


