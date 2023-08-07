import time

from qtensor import QAOA_energy
import networkx as nx
import numpy as np


start = time.perf_counter()
G = nx.complete_graph(20)
gamma, beta = [np.pi/4], [np.pi/4]

E = QAOA_energy(G, gamma, beta)
end = time.perf_counter()
print(E)
print(f'Time elapsed: {end - start}')
