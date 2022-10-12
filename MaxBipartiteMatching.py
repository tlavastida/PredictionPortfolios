'''
A warm-start implementation of maximum cardinality bipartite matching
via Hopcroft-Karp algorithm

warm-start means that it can start from any feasible matching

references: 
https://github.com/matejker/network/blob/beb591a570a8fb66d5dc4c4b48d3ff1e4f8e9134/network/algorithms/matching.py

https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm#Pseudocode

https://github.com/sofiatolaosebikan/hopcroftkarp 

John E. Hopcroft and Richard M. Karp. (1973),
        "An n^{5 / 2} Algorithm for Maximum Matchings in Bipartite Graphs",
         SIAM Journal of Computing 2.4, pp. 225--231. <https://doi.org/10.1137/0202019>.

'''

from collections import deque

from BipartiteGraph import BipartiteGraph


# finds a greedy matching in the graph G
# here G is assumed to be of class BipartiteGraph
# returns dictionaries U_matches,V_matches
# U_matches[u] = v if (u,v) in M and None otherwise
# V_matches[v] = u if (u,v) in M and None otherwise
def GreedyMatching(G,U_matches = None,V_matches = None):
    
    if U_matches is None:
        U_matches = {v: None for v in G.U}
    if V_matches is None:
        V_matches = {v: None for v in G.V}

    # attempt to match every vertex
    for u in G.U:
        # if u not matched already try to find a match
        if U_matches[u] is None:
            for v in G.Adj[u]:
                # v is free, then match u,v
                if V_matches[v] is None:
                    U_matches[u] = v
                    V_matches[v] = u
                    break

    return U_matches,V_matches


# define infinity
INF = float("inf")

# checks that a valid matching is represented by
# U_matches and V_matches in G
def check_matches(G,U_matches,V_matches):
    U = G.U
    V = G.V
    # first check that the dicts are well defined and that
    # valid edges are chosen
    for u in U:
        if u not in U_matches.keys() or (U_matches[u] is not None and U_matches[u] not in G.Adj[u]):
            raise ValueError('U_matches is not valid')
    for v in V:
        if v not in V_matches.keys() or (V_matches[v] is not None and V_matches[v] not in G.Adj[v]):
            raise ValueError('V_matches is not valid')
    
    # check that each vertex is used at most once
    match_count = {}
    for v in G.AllVerts:
        match_count[v] = 0
        
    for u in U:
        v = U_matches[u]
        if v is not None:
            match_count[v] += 1
            if match_count[v] > 1:
                raise ValueError('{} is overmatched in U_matches'.format(v))

    for v in V:
        u = V_matches[v]
        if u is not None:
            match_count[u] += 1
            if match_count[u] > 1:
                raise ValueError('{} is overmatched in V_matches'.format(u))
        
        # otherwise it's okay
    return True


# G is of class Bipartite Graph with
# left and right sides U,V
# U_matches, V_matches are dictionaries
# defining each sides matches
# U_matches[u] = v if (u,v) in M and None otherwise
# V_matches[v] = u if (u,v) in M and None otherwise
# does a validation on the input matching if check_matching is set to True
def HopcroftKarp(G, one_side=False, U_matches = None,V_matches = None,
                check_matching=False):

    if type(G) is not BipartiteGraph:
        raise TypeError('Input G must be of class BipartiteGraph')

    U = G.U
    V = G.V

    # If not a warm start, initialize the matching
    # Both must be initialized to use warmstart
    if U_matches is None or V_matches is None:
        U_matches = {v: None for v in U}
        V_matches = {v: None for v in V}
    # If doing warm start check to see if they form a valid matching
    elif check_matching is True:
        check_matches(G,U_matches,V_matches)


    # distances
    dist = {}
    # queue for BFS
    queue = deque()

    # aux BFS function
    def bfs():
        for u in U:
            if U_matches[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        
        dist[None] = INF
    
        while queue:
            u = queue.popleft()
            if dist[u] < dist[None]:
                for v in G.Adj[u]:
                    if dist[V_matches[v]] is INF:
                        dist[V_matches[v]] = dist[u] + 1
                        queue.append(V_matches[v])
        
        return dist[None] is not INF

    # aux DFS function
    def dfs(u):
        if u is not None:
            for v in G.Adj[u]:
                if dist[V_matches[v]] == dist[u] + 1:
                    if dfs(V_matches[v]):
                        V_matches[v] = u
                        U_matches[u] = v
                        return True
            
            dist[u] = INF
            return False
        
        return True

    # now the actual algo

    while bfs():
        for u in U:
            if U_matches[u] is None:
                dfs(u)

    return U_matches if one_side else (U_matches,V_matches)


# converts a matching given by U_matches,V_matches to a vertex cover
# returns sets LU, LV which are subsets of U and V respectively
# L = LU + LV is the set of vertices reachable from unmatched vertices in U
# where we can only cross from U to V by unmatched edges
# and from V to U by matched edges
def MatchingToAugmentingSets(G,U_matches,V_matches):
    # initialize
    L = set(u for u in G.U if U_matches[u] is None)

    LU = set()
    LV = set()

    while L:
        x = L.pop()
        if x in G.U:
            LU.add(x)
            for y in G.Adj[x]:
                if U_matches[x] != y and y not in LV:
                    L.add(y)

        else: # x in G.V
            LV.add(x)
            for y in G.Adj[x]:
                if V_matches[x] == y and y not in LU:
                    L.add(y)

    return LU,LV

            


