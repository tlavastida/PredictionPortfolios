
'''
Bipartite graph class

Adapted from: https://github.com/dilsonpereira/BipartiteMatching
'''

class BipartiteGraph():
    def __init__(self, U, V, E = [], W = None):
        '''Class constructor
        U and V are iterables containing the vertices in the left and right partitions
        vertices can be any hashable objects
        E is an iterable of edges (edges are iterables of size 2)
        W is an iterable containing the weights of the edges
        '''
        self.U = set(U)
        self.V = set(V)
        self.AllVerts = self.U|self.V
        self.Adj = {v:set() for v in self.AllVerts}
        for (v1, v2) in E:
            self.Adj[v1].add(v2)
            self.Adj[v2].add(v1)

        if W != None:
            self.W = {tuple(e):W[k] for (k, (u, v)) in enumerate(E) for e in [(u,v), (v,u)] }
        else:
            self.W = None
        self.E = set(tuple(e) for e in E)

        if not self.is_bipartite():
            raise ValueError('Input graph failed bipartite check')

    # function for printing a bipartite graph
    def __str__(self):
        ret = ''
        for u in self.U:
            ret += str(u) + ' : { '
            for v in self.V:
                if (u,v) in self.E:
                    if self.W is None:
                        ret += str(v) + ', '
                    else:
                        ret += str(v) + ':' + str(self.W[(u,v)]) + ', '
            
            # cut off last comma and space
            ret = ret[:len(ret)-2]
            # set up for next line
            ret += ' }\n'

        # all done now
        return ret

    def is_bipartite(self):
        '''
        returns true if graph is bipartite with bipartition U,V and false otherwise
        '''
        for e in self.E:
            u,v = e
            if (u in self.U and v in self.U) or (u in self.V and v in self.V):
                print('({},{}) is a bad edge!'.format(u,v))
                return False
        return True
            
        
            
    def matching_weight(self,M):
        '''
        Computes the weight of matching M in this graph
        M is an iterable containing edges (iterables of size 2) in a valid matching
        Returns the cardinality of the matching if weights are undefined
        '''
        tot = 0
        for e in M:
            if e in self.E:
                tot += 1 if self.W is None else self.W[e]
    
        return tot