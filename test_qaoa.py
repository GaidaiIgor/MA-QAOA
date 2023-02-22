from qaoa_core import *
import pytest


class TestQAOA:
    @pytest.fixture
    def graph(self):
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
        return graph

    @pytest.mark.parametrize(("multi_angle", "use_analytical", "p", "expected"),
                             [(False, False, 1, 6.232), (False, True, 1, 6.232), (True, False, 1, 6.5), (True, True, 1, 6.5)])
    def test_optimize_qaoa_angles(self, multi_angle, use_analytical, p, graph, expected):
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, p, graph)
        assert abs(objective_best - expected) < 1e-3
