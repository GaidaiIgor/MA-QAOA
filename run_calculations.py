from functools import partial
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.data_processing import collect_results_from, calculate_edge_diameter, calculate_min_p, merge_dfs, numpy_str_to_array
from src.graph_utils import get_edge_diameter, get_max_edge_depth, find_non_isomorphic, is_isomorphic
from src.parallel import optimize_expectation_parallel, worker_standard_qaoa, calculate_maxcut_parallel


def collect_results_xqaoa():
    aggregator = np.mean
    df_gqaoa = collect_results_from('graphs/xqaoa/output', ['GQAOA'], aggregator)
    df_xqaoa = collect_results_from('simulation_data', ['XQAOA', 'Geomans_Williamson'], aggregator)
    return df_gqaoa.join(df_xqaoa)


def generate_graphs():
    num_graphs = 1000
    max_attempts = 10 ** 10
    nodes = 10
    depth = 6
    edge_prob = 0.1
    out_path = f'graphs/new/nodes_{nodes}/depth_{depth}'

    graphs = np.empty(num_graphs, dtype=object)
    graphs_generated = 0
    for i in range(max_attempts):
        next_graph = nx.gnp_random_graph(nodes, edge_prob)
        if not nx.is_connected(next_graph):
            continue
        if get_max_edge_depth(next_graph) != depth:
            continue
        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated}')
        if graphs_generated == num_graphs:
            break
    else:
        raise 'Failed to generate connected set'
    print('Generation done')

    # print('Calculating depth')
    # depths = [get_max_edge_depth(graph) for graph in graphs]
    # histogram = np.histogram(depths, bins=range(1, nodes))
    # print(histogram)
    # return

    # print('Checking isomorphisms')
    # isomorphisms = find_non_isomorphic(graphs)
    # print(f'Number of non-isomorphic: {sum(isomorphisms)}')

    for i in range(len(graphs)):
        nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


def run_graphs_init():
    num_graphs = 1000
    nodes = list(range(9, 13))
    depths = list(range(4, 7))
    # for node in nodes:
    #     data_path = f'graphs/new/nodes_{node}/depth_3/'
    for depth in depths:
        data_path = f'graphs/new/nodes_12/depth_{depth}/'
        num_workers = 20
        reader = partial(nx.read_gml, destringizer=int)
        paths = [f'{data_path}/{i}.gml' for i in range(num_graphs)]
        calculate_maxcut_parallel(paths, num_workers, reader)


def get_out_path(data_path: str, method: str, initial_guess: str, p: int) -> str:
    extra = f'p_{p}' if initial_guess == 'random' else ''
    return f'{data_path}/output/{method}/{initial_guess}/{extra}/out.csv'


def get_starting_angles_col_name(initial_guess: str, p: int) -> str | None:
    if initial_guess == 'random':
        return None
    elif initial_guess == 'interp':
        return f'p_{p - 1}_angles'
    elif initial_guess == 'qaoa':
        return f'p_{p}_starting_angles'


def init_dataframe(initial_guess: str, data_path: str, num_graphs: int, out_path: str):
    if initial_guess == 'interp':
        df = pd.read_csv(f'{data_path}/output/qaoa/random/p_1/out.csv', index_col=0)
        df = df.filter(regex='r_10').rename(columns=lambda name: f'p_1{name[4:]}')
    elif initial_guess == 'qaoa':
        df = pd.read_csv(f'{data_path}/output/qaoa/interp/out.csv', index_col=0)
        df = df.filter(regex=r'p_\d+_angles').rename(columns=lambda name: f'{name[:-7]}_starting_angles')
    else:
        paths = [f'{data_path}/{i}.gml' for i in range(num_graphs)]
        df = DataFrame(paths).set_axis(['path'], axis=1).set_index('path')
    df.to_csv(out_path)


def run_graphs_parallel():
    num_graphs = 1000
    num_workers = 20
    num_restarts = 1
    worker = 'standard'
    initial_guess = 'qaoa'
    methods = ['ma']
    ps = {'qaoa': list(range(2, 11)), 'ma': list(range(1, 6))}
    nodes = list(range(9, 13))
    depths = list(range(3, 7))
    restarts = list(range(1, 2))
    reader = partial(nx.read_gml, destringizer=int)
    copy_better = True
    convergence_threshold = 0.9995

    for method in methods:
        for node in nodes:
            node_depths = [3] if node < 12 else depths
            for depth in node_depths:
                for p in ps[method]:
                    for r in restarts:
                        search_space = method
                        guess_format = 'qaoa' if initial_guess == 'qaoa' else method
                        data_path = f'graphs/new/nodes_{node}/depth_{depth}/'
                        out_path = get_out_path(data_path, method, initial_guess, p)
                        out_col_prefix = 'r' if initial_guess == 'random' else 'p'
                        counter = r if out_col_prefix == 'r' else p
                        starting_angles_col = get_starting_angles_col_name(initial_guess, p)
                        out_col_name = f'{out_col_prefix}_{counter}'
                        rows_func = lambda df: None if counter == 1 else df[f'{out_col_prefix}_{counter - 1}'] < convergence_threshold
                        copy_col = None if counter == 1 else f'{out_col_prefix}_{counter - 1}'
                        copy_p = p if out_col_prefix == 'r' else p - 1

                        if not path.exists(out_path):
                            init_dataframe(initial_guess, data_path, num_graphs, out_path)

                        optimize_expectation_parallel(out_path, rows_func, num_workers, worker, reader, search_space, p, initial_guess, guess_format, starting_angles_col,
                                                      num_restarts, copy_col, copy_p, copy_better, out_col_name)


def run_graph_sequential():
    data = ('graphs/new/nodes_9/depth_3/0.gml', numpy_str_to_array('[-1.33029499 -2.06502225]'))
    reader = partial(nx.read_gml, destringizer=int)
    p = 1
    search_space = 'ma'
    guess_format = 'qaoa'
    num_restarts = 1
    path, ar, angles, nfev = worker_standard_qaoa(data, reader, p, search_space, guess_format, num_restarts)
    print('Done')


def run_merge():
    copy_better = True
    nodes = [9, 10, 11, 12]
    depths = [3, 4, 5, 6]
    methods = ['qaoa', 'ma']
    ps = [list(range(1, 11)), list(range(1, 6))]
    restarts = 1
    convergence_threshold = 0.9995
    for method_ind, method in enumerate(methods):
        for node in nodes:
            node_depths = [3] if node < 12 else depths
            for depth in node_depths:
                base_path = f'graphs/new/nodes_{node}/depth_{depth}/output/{method}/random'
                merge_dfs(base_path, ps[method_ind], restarts, convergence_threshold, f'{base_path}/out_r{restarts}.csv', copy_better)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # collect_results_xqaoa()
    # extend_csv()

    # df = pd.read_csv('graphs/nodes_7/output/ma/random/merged_r5.csv', index_col=0)
    # df = calculate_edge_diameter(df)
    # df = calculate_min_p(df)

    # run_merge()
    run_graph_sequential()
    # generate_graphs()
    # run_graphs_init()
    # run_graphs_parallel()
