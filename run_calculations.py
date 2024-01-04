"""
Entry points for large scale parallel calculation functions.
"""

import os
from functools import partial
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.data_processing import merge_dfs, numpy_str_to_array
from src.graph_utils import get_max_edge_depth, is_isomorphic
from src.parallel import optimize_expectation_parallel, WorkerFourier, WorkerStandard, WorkerBaseQAOA, WorkerInterp, WorkerGreedy, WorkerMA, WorkerLinear, WorkerCombined, \
    WorkerConstant


def generate_graphs():
    num_graphs = 1000
    max_attempts = 10 ** 5
    nodes = 7
    target_depth = 3
    edge_prob = 0.6
    out_path = f'graphs/other/nodes_{nodes}/'

    graphs = np.empty(num_graphs, dtype=object)
    valid_count = 0
    disconnected_count = 0
    wrong_depth_count = 0
    isomorphic_count = 0
    avg_depth = 0
    for i in range(max_attempts):
        next_graph = nx.gnp_random_graph(nodes, edge_prob)
        connected = nx.is_connected(next_graph)
        depth = get_max_edge_depth(next_graph)
        isomorphic = is_isomorphic(next_graph, graphs[:valid_count])
        avg_depth = (avg_depth * i + depth) / (i + 1) if i > 0 else depth

        if not connected:
            disconnected_count += 1
        if depth != target_depth:
            wrong_depth_count += 1
        if isomorphic:
            isomorphic_count += 1
        if connected and depth == target_depth and not isomorphic:
            graphs[valid_count] = next_graph
            valid_count += 1
            print(f'\rGraphs generated: {valid_count}', end='')
        if valid_count == num_graphs:
            success = True
            break
    else:
        success = False

    print(f'\nTotal disconnected: {disconnected_count}')
    print(f'Total wrong depth: {wrong_depth_count}')
    print(f'Total isomorphic: {isomorphic_count}')
    print(f'Average depth: {avg_depth}')
    print(f'Success rate: {valid_count / i}')
    if success:
        print('Generation done')
    else:
        raise Exception('Failed to generate a valid graph set')

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


def init_dataframe(data_path: str, worker: WorkerBaseQAOA, out_path: str):
    if worker.initial_guess_from is None:
        paths = [f'{data_path}/{i}.gml' for i in range(1000)]
        df = DataFrame(paths).set_axis(['path'], axis=1).set_index('path')
    elif isinstance(worker, (WorkerInterp, WorkerFourier, WorkerGreedy, WorkerCombined)) or hasattr(worker, 'guess_provider') and isinstance(worker.guess_provider, WorkerInterp):
        df = pd.read_csv(f'{data_path}/output/{worker.search_space}/random/p_1/out.csv', index_col=0)
        prev_nfev = df.filter(regex=r'r_\d_nfev').sum(axis=1).astype(int)
        df = df.filter(regex='r_10').rename(columns=lambda name: f'p_1{name[4:]}')
        df['p_1_nfev'] += prev_nfev
        if isinstance(worker, (WorkerInterp, WorkerFourier)):
            df = df.rename(columns={'p_1_angles': 'p_1_angles_unperturbed'})
            df['p_1_angles_best'] = df['p_1_angles_unperturbed']
    elif isinstance(worker, WorkerMA):
        df = pd.read_csv(f'{data_path}/output/qaoa/constant/0.2/out.csv', index_col=0)
        # df = df.filter(regex=r'p_\d+_angles').rename(columns=lambda name: f'{name[:-7]}_starting_angles')
        df = df.filter(regex=r'p_\d+_angles')
    else:
        raise Exception('No init for this worker')
    df.to_csv(out_path)


def run_graphs_parallel():
    nodes = list(range(9, 10))
    depths = list(range(3, 7))
    ps = list(range(1, 9))

    num_workers = 20
    convergence_threshold = 0.9995
    reader = partial(nx.read_gml, destringizer=int)

    for p in ps:
        out_path_suffix = 'output/qaoa/random/attempts_1/nfev/out.csv'
        out_col = f'p_{p}'
        initial_guess_from = None if p == 1 else f'p_{p - 1}'
        initial_guess_from = f'p_{p}'
        transfer_from = None if p == 1 else f'p_{p - 1}'
        transfer_p = None if p == 1 else p - 1

        worker_standard = WorkerStandard(reader=reader, p=p, out_col=out_col, initial_guess_from=None, transfer_from=transfer_from, transfer_p=transfer_p, search_space='qaoa')
        # worker_constant = WorkerConstant(reader=reader, p=p, out_col=out_col, initial_guess_from=None, transfer_from=transfer_from, transfer_p=transfer_p)
        # worker_tqa = WorkerLinear(reader=reader, p=p, out_col=out_col, initial_guess_from=None, transfer_from=transfer_from, transfer_p=transfer_p, search_space='tqa')
        # worker_interp = WorkerInterp(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p, alpha=0.6)
        # worker_fourier = WorkerFourier(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p, alpha=0.6)
        # worker_greedy = WorkerGreedy(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p)
        # worker_combined = WorkerCombined(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p,
        #                                  workers=[worker_interp, worker_greedy], restart_shares=[0.5, 0.5])
        # worker_ma = WorkerMA(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p,
        #                      guess_provider=None, guess_format='qaoa')
        worker = worker_standard

        for node in nodes:
            node_depths = [3] if node < 12 else depths
            for depth in node_depths:
                data_path = f'graphs/main/nodes_{node}/depth_{depth}/'
                out_path = data_path + out_path_suffix

                rows_func = lambda df: np.ones((df.shape[0], 1), dtype=bool) if p == 1 else df[f'p_{p - 1}'] < convergence_threshold
                # rows_func = lambda df: (df[f'p_{p - 1}'] < convergence_threshold) & (df[f'p_{p}'] - df[f'p_{p - 1}'] < 1e-3)
                # rows_func = lambda df: (df[f'p_{p}'] < convergence_threshold) & ((df[f'p_{p}_nfev'] == 1000 * p) | (df[f'p_{p}'] < df[f'p_{p - 1}']))

                # mask = np.zeros((1000, 1), dtype=bool)
                # mask[:] = True
                # rows_func = lambda df: mask

                out_folder = path.split(out_path)[0]
                if not path.exists(out_folder):
                    os.makedirs(path.split(out_path)[0])
                if not path.exists(out_path):
                    init_dataframe(data_path, worker, out_path)

                optimize_expectation_parallel(out_path, rows_func, num_workers, worker)


def run_correct():
    nodes = list(range(9, 13))
    depths = list(range(3, 7))
    for node in nodes:
        node_depths = [3] if node < 12 else depths
        for depth in node_depths:
            data_path = f'graphs/new/nodes_{node}/depth_{depth}/output/qaoa/random/p_1/out.csv'
            df = pd.read_csv(data_path, index_col=0)
            for r in range(1, 11):
                for i in range(1000):
                    angles = numpy_str_to_array(df.loc[f'graphs/new/nodes_{node}/depth_{depth}//{i}.gml', f'r_{r}_angles'])
                    angles = angles[angles != 0]
                    df.loc[f'graphs/new/nodes_{node}/depth_{depth}//{i}.gml', f'r_{r}_angles'] = str(angles)
            df.to_csv(data_path)


def run_merge():
    copy_better = True
    nodes = [9]
    depths = [3, 4, 5, 6]
    methods = ['ma']
    ps_all = {'qaoa': list(range(1, 12)), 'ma': list(range(1, 6))}
    convergence_threshold = 0.9995
    for method in methods:
        ps = ps_all[method]
        for node in nodes:
            node_depths = [3] if node < 12 else depths
            for depth in node_depths:
                base_path = f'graphs/new/nodes_{node}/depth_{depth}/output/{method}/random'
                # restarts = [1] * len(ps)
                restarts = ps
                merge_dfs(base_path, ps, restarts, convergence_threshold, f'{base_path}/attempts_p/out.csv', copy_better)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    generate_graphs()
    # run_graphs_parallel()
    # run_merge()
    # run_correct()
