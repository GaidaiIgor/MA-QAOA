from qaoa_core import *


def main():
    # TODO: graphs db interface
    # TODO: choice between single edge or whole graph expectation
    # TODO: starting angles range
    multi_angle = True
    use_analytical = False
    p = 1

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(0, 2, weight=1)
    graph.add_edge(0, 4, weight=1)
    graph.add_edge(1, 3, weight=1)
    graph.add_edge(1, 5, weight=1)
    graph.add_edge(2, 3, weight=1)
    graph.add_edge(2, 5, weight=1)
    graph.add_edge(3, 4, weight=1)
    graph.add_edge(4, 5, weight=1)

    objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph)
    print(f'Objective best: {objective_best}')


if __name__ == '__main__':
    main()
    # cProfile.run('main()')
