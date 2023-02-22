from qaoa_core import *


def run_main():
    multi_angle = True
    use_analytical = False
    p = 1
    graph = nx.read_weighted_edgelist('graphs/simple_3reg.wel', nodetype=int)
    objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph)
    print(f'Best achieved objective: {objective_best}')


# TODO: choice between single edge or whole graph expectation
# TODO: starting angles range
if __name__ == "__main__":
    run_main()
