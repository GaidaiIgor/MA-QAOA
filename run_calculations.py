from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

from src.data_processing import collect_results_from, calculate_edge_diameter, calculate_min_p
from src.graph_utils import get_edge_diameter
from src.parallel import optimize_expectation_parallel


def collect_results_xqaoa():
    aggregator = np.mean
    df_gqaoa = collect_results_from('graphs/xqaoa/output', ['GQAOA'], aggregator)
    df_xqaoa = collect_results_from('simulation_data', ['XQAOA', 'Geomans_Williamson'], aggregator)
    return df_gqaoa.join(df_xqaoa)


def run_graphs_parallel():
    input_path = 'graphs/nodes_10/output/ma/random/p_5/out.csv'
    num_workers = 20
    worker = 'standard'
    search_space = 'ma'
    initial_guess = 'random'
    guess_format = 'ma'
    num_restarts = 1
    out_path = input_path
    reader = partial(nx.read_gml, destringizer=int)

    # for p in range(1, 6):
    #     input_df = pd.read_csv(input_path, index_col=0)
    #     col_name = f'p_{p}'
    #     starting_angles_col = f'p_{p}_angles'
    #     # rows = (input_df['p_rel_ed'] == 1) & (input_df['edge_diameter'] == 3)
    #     rows = None if p == 1 else input_df[f'p_{p - 1}'] < 0.9995
    #     comparison_col = None if p == 1 else f'p_{p - 1}'
    #     optimize_expectation_parallel(input_df, num_workers, worker, reader, search_space, p, num_restarts, out_path, col_name, initial_guess, guess_format, starting_angles_col,
    #                                   rows, comparison_col)

    p = 5
    starting_angles_col = None
    for r in range(1, 11):
        input_df = pd.read_csv(input_path, index_col=0)
        col_name = f'r_{r}'
        rows = None if r == 1 else input_df[f'r_{r - 1}'] < 0.9995
        comparison_col = None if r == 1 else f'r_{r - 1}'
        optimize_expectation_parallel(input_df, num_workers, worker, reader, search_space, p, num_restarts, out_path, col_name, initial_guess, guess_format, starting_angles_col,
                                      rows, comparison_col)


def generate_graphs():
    out_path = 'graphs/nodes_8/ed_5.5'
    num_graphs = 10000
    nodes = 8
    edge_prob = 0.05
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

    # if abs(np.mean(diameters) - 5) < 5e-3:
    #     for i in range(len(graphs)):
    #         nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


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


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # collect_results_xqaoa()
    # extend_csv()

    # df = pd.read_csv('graphs/nodes_7/output/ma/random/merged_r5.csv', index_col=0)
    # df = calculate_edge_diameter(df)
    # df = calculate_min_p(df)

    run_graphs_parallel()

    # for i in range(1):
    #     generate_graphs()
