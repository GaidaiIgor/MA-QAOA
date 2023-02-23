import networkx as nx

from src.optimization import optimize_qaoa_angles


def run_main():
    multi_angle = False
    use_analytical = False
    p = 2
    graph = nx.read_weighted_edgelist('graphs/reg3_simple.wel', nodetype=int)
    # edge_list = [(0, 1)]
    edge_list = None
    objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')


# TODO: starting angles range?
if __name__ == "__main__":
    run_main()
