from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

from src.data_processing import collect_results_from, calculate_edge_diameter, calculate_min_p
from src.graph_utils import get_edge_diameter
from src.parallel import optimize_expectation_parallel, worker_standard_qaoa


def collect_results_xqaoa():
    aggregator = np.mean
    df_gqaoa = collect_results_from('graphs/xqaoa/output', ['GQAOA'], aggregator)
    df_xqaoa = collect_results_from('simulation_data', ['XQAOA', 'Geomans_Williamson'], aggregator)
    return df_gqaoa.join(df_xqaoa)


def run_graphs_parallel():
    ps = list(range(1, 11))
    for p in ps:
        input_path = f'graphs/nodes_10/output/qaoa/random/p_{p}/out.csv'
        num_workers = 20
        worker = 'standard'
        search_space = 'qaoa'
        initial_guess = 'random'
        guess_format = 'qaoa'
        num_restarts = 1
        reader = partial(nx.read_gml, destringizer=int)
        copy_better = True

        # for p in range(5, 6):
        #     rows_func = lambda df: None if p == 1 else df[f'p_{p - 1}'] < 0.9995
        #     starting_angles_col = f'p_{p}_starting_angles'
        #     copy_p = p - 1
        #     copy_col = None if p == 1 else f'p_{copy_p}'
        #     out_col_name = f'p_{p}'
        #     optimize_expectation_parallel(input_path, rows_func, num_workers, worker, reader, search_space, p, initial_guess, guess_format, starting_angles_col, num_restarts,
        #                                   copy_col, copy_p, copy_better, out_col_name)

        # p = 10
        starting_angles_col = None
        for r in range(1, 2):
            out_col_name = f'r_{r}'
            rows_func = lambda df: None if r == 1 else df[f'r_{r - 1}'] < 0.9995
            copy_col = None if r == 1 else f'r_{r - 1}'
            optimize_expectation_parallel(input_path, rows_func, num_workers, worker, reader, search_space, p, initial_guess, guess_format, starting_angles_col, num_restarts,
                                          copy_col, p, copy_better, out_col_name)


def generate_graphs():
    out_path = 'graphs/nodes_12/'
    num_graphs = 1000
    nodes = 12
    edge_prob = 0.365
    graphs = []
    diameters = []
    for i in range(num_graphs):
        connected = False
        while not connected:
            next_graph = nx.gnp_random_graph(nodes, edge_prob)
            connected = nx.is_connected(next_graph)
        graphs.append(next_graph)
        diameters.append(get_edge_diameter(next_graph))
    print(np.mean(diameters))
    print(np.std(diameters) / np.sqrt(num_graphs) * 2)

    if abs(np.mean(diameters) - 4) < 5e-3:
        for i in range(len(graphs)):
            nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


def merge_dfs():
    base_path = 'graphs/nodes_10/output/ma/random'
    r = [15, 15, 15, 15]
    # r = [10] * 4
    df = pd.DataFrame()
    for p in range(1, 5):
        next_df = pd.read_csv(f'{base_path}/p_{p}/out.csv', index_col=0)
        next_df = next_df[[f'r_{r[p - 1]}']].set_axis([f'p_{p}'], axis=1)
        if df.empty:
            df = next_df
        else:
            df = df.join(next_df)
    df = calculate_edge_diameter(df)
    df = calculate_min_p(df)
    df.to_csv(f'{base_path}/merged_r{r[0]}.csv')


def run_graph_sequential():
    data = ('graphs/nodes_8/ed_4/333.gml', np.array([2.812676, -0.41387697, 0.10024993, -2.36888893]))
    reader = partial(nx.read_gml, destringizer=int)
    p = 4
    search_space = 'qaoa'
    guess_format = 'qaoa'
    num_restarts = 1
    _, exp, angles = worker_standard_qaoa(data, reader, p, search_space, guess_format, num_restarts)
    print('Done')


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # collect_results_xqaoa()
    # extend_csv()

    # df = pd.read_csv('graphs/nodes_7/output/ma/random/merged_r5.csv', index_col=0)
    # df = calculate_edge_diameter(df)
    # df = calculate_min_p(df)

    run_graphs_parallel()

    # generate_graphs()
