from qaoa_core import *
import pytest


class TestQAOA:
    @pytest.fixture
    def graph(self):
        graph = nx.read_weighted_edgelist('graphs/simple_3reg.wel', nodetype=int)
        return graph

    @pytest.mark.parametrize(("multi_angle", "use_analytical", "p", "expected"),
                             [(False, False, 1, 6.232), (False, True, 1, 6.232), (True, False, 1, 6.5), (True, True, 1, 6.5)])
    def test_optimize_qaoa_angles(self, multi_angle, use_analytical, p, graph, expected):
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph)
        assert abs(objective_best - expected) < 1e-3
