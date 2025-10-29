import numpy as np
import networkx as nx

def skipTree(S: nx.Graph, root: int=0, edge_index_verts=None):
    n = S.number_of_nodes()
    index = 0
    label = [0]*n
    visited = set()
    
    def label_first(v:int,skip:bool):
        nonlocal index
        visited.add(v)
        label[index] = v
        index=index+1

        children = [nbr for nbr in S.neighbors(v) if nbr not in visited]
        for child_idx, child in enumerate(children):
            last_in_gen = (child_idx == len(children) - 1)
            if last_in_gen and not skip:
                label_first(child,skip=False)
            else:
                label_last(child)

    def label_last(v:int):
        nonlocal index
        visited.add(v)
        for child in S.neighbors(v):
            if child not in visited:
                label_first(child,skip=True)
        label[index] = v
        index=index+1
        
    label_first(root,skip=False)
    
    P = np.zeros((n,n))
    for l, v in enumerate(label):
        P[v, l] = 1
    
    
    if not edge_index_verts: 
        T = np.zeros((n,n-1))
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}
    else:
        T = np.zeros((n,len(edge_index_verts)))

    for l in range(n-1):
        path = nx.shortest_path(S, source=label[l], target=label[(l + 1)%n])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l,edge_index_verts[e]] = 1
    return T, P
    