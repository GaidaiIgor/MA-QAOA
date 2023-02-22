from qaoa_core import *
import pytest


class TestQAOA:
    @pytest.fixture
    def graph(self):
        graph = nx.read_weighted_edgelist('graphs/simple_3reg.wel', nodetype=int)
        return graph

    @pytest.mark.parametrize(("multi_angle", "use_analytical", "p", "expected"),
                             [(False, False, 1, 6.232), (False, True, 1, 6.232), (True, False, 1, 6.5), (True, True, 1, 6.5)])
    def test_optimize_qaoa_angles_all_edges(self, multi_angle, use_analytical, p, graph, expected):
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph)
        assert abs(objective_best - expected) < 1e-3

    @pytest.mark.parametrize(("multi_angle", "use_analytical", "p", "expected"),
                             [(False, False, 1, 0.6924), (False, True, 1, 0.6924), (True, False, 1, 1), (True, True, 1, 1)])
    def test_optimize_qaoa_angles_one_edge(self, multi_angle, use_analytical, p, graph, expected):
        edge_list = [(0, 1)]
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph, edge_list)
        assert abs(objective_best - expected) < 1e-4
