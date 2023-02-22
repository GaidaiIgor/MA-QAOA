import networkx as nx

from src.optimization import optimize_qaoa_angles


def run_main():
    multi_angle = False
    use_analytical = False
    p = 1
    graph = nx.read_weighted_edgelist('graphs/simple_3reg.wel', nodetype=int)
    edge_list = [(0, 1)]
    objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
    print(f'Best achieved objective: {objective_best}')


# TODO: starting angles range
# TODO: fix objective max
if __name__ == "__main__":
    run_main()
