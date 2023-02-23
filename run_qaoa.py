import networkx as nx

from src.optimization import optimize_qaoa_angles


def run_main():
    multi_angle = True
    use_analytical = True
    p = 1
    graph = nx.read_weighted_edgelist('graphs/simple_reg3.wel', nodetype=int)
    edge_list = [(4, 5)]
    # edge_list = None
    objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')


# TODO: starting angles range?
if __name__ == "__main__":
    run_main()
