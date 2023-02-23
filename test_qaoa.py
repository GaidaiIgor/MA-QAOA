import networkx as nx
import pytest

from src.optimization import optimize_qaoa_angles


class TestQAOAOptimize:
    @pytest.fixture
    def simple_line_3(self):
        graph = nx.read_weighted_edgelist('graphs/simple_line_3.wel', nodetype=int)
        return graph

    @pytest.fixture
    def simple_line_5(self):
        graph = nx.read_weighted_edgelist('graphs/simple_line_5.wel', nodetype=int)
        return graph

    @pytest.fixture
    def reg3_tree_sub(self):
        graph = nx.read_weighted_edgelist('graphs/reg3_tree_sub.wel', nodetype=int)
        return graph

    @pytest.fixture
    def simple_reg3(self):
        graph = nx.read_weighted_edgelist('graphs/simple_reg3.wel', nodetype=int)
        return graph

    @pytest.mark.parametrize("use_analytical", [False, True])
    @pytest.mark.parametrize(("multi_angle", "expected"), [(False, 3/4), (True, 1)])
    def test_line_3(self, simple_line_3, use_analytical, multi_angle, expected):
        """ Tests expected values of <C> on one edge on a simple line graph with 3 edges (subgraph of a 2-regular graph for p=1).
        Tests both classical and multi-angle approach in simulation and analytical mode.
        Classical QAOA (multi_angle=False) is expected to return 3/4 as known from Farhi et al.
        Multi-angle is expected to do better than classical, 1 in this case.
        Analytical modes are expected to return the same answer as quantum simulation. """
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, 1, simple_line_3, [(0, 1)])
        assert abs(objective_best - expected) < 1e-4

    @pytest.mark.parametrize(("multi_angle", "expected"), [(False, 5/6), (True, 1)])
    def test_line_5_simulation_p2(self, simple_line_5, multi_angle, expected):
        """ Tests expected values of <C> on one edge on a simple line graph with 5 edges (subgraph of a 2-regular graph for p=2).
        Classical QAOA (multi_angle=False) is expected to return 5/6 as known from Farhi et al.
        Multi-angle is expected to do better than classical, 1 in this case.
        Checks that simulation works for p > 1, where no analytical formulas exist. """
        objective_best = optimize_qaoa_angles(multi_angle, False, 2, simple_line_5, [(0, 1)])
        assert abs(objective_best - expected) < 1e-4

    @pytest.mark.parametrize("use_analytical", [False, True])
    @pytest.mark.parametrize(("multi_angle", "expected"), [(False, 0.6924), (True, 1)])
    def test_reg3_tree_sub(self, reg3_tree_sub, multi_angle, use_analytical, expected):
        """ Tests expected values of <C> on one edge on a tree-like subgraph of a 3-regular graph (for p=1).
        Classical QAOA (multi_angle=False) is expected to return 0.6924 as known from Farhi et al.
        Multi-angle is expected to do better than classical, 1 in this case. """
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, 1, reg3_tree_sub, [(0, 1)])
        assert abs(objective_best - expected) < 1e-4

    @pytest.mark.parametrize("use_analytical", [False, True])
    @pytest.mark.parametrize(("multi_angle", "expected"), [(False, 6.232), (True, 6.5)])
    def test_reg3_tree_sub_all_edges(self, multi_angle, use_analytical, simple_reg3, expected):
        objective_best = optimize_qaoa_angles(multi_angle, use_analytical, 1, simple_reg3)
        assert abs(objective_best - expected) < 1e-3
