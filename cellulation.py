import numpy as np
import networkx as nx

def cellulate_long_cycles(G, edge_qubit_to_vertices, vert_to_edge, G_mat, max_len=6):
    cycles = nx.cycle_basis(G)
    new_edges = []
    next_edge_index = max(edge_qubit_to_vertices.keys()) + 1
    
    for cycle in cycles:
        while len(cycle) > max_len: # just BB case
            n = len(cycle)
            i = 0
            j = i + n // 2
            u = cycle[i]
            v = cycle[j % n]

            u, v = sorted((u, v))
            
            if not G.has_edge(u, v):
                # Add edge to graph
                G.add_edge(u, v)
                new_edges.append((u, v))
                
                # Update dictionaries
                edge_qubit_to_vertices[next_edge_index] = (u, v)
                vert_to_edge[(u, v)] = next_edge_index
                
                # Update G_mat by appending a new row for this edge
                n_vertices = G_mat.shape[1]
                new_row = np.zeros((1, n_vertices))
                new_row[0, u] = 1
                new_row[0, v] = 1
                G_mat = np.vstack([G_mat, new_row])
                
                next_edge_index += 1
            
            # Recompute cycles to pick up remaining long cycles
            cycle = nx.cycle_basis(G)[0]

    return new_edges, edge_qubit_to_vertices, vert_to_edge, G_mat