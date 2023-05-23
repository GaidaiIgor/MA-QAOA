"""
QAOA tests
"""
import networkx as nx
import numpy as np
import pytest

from src.optimization import optimize_qaoa_angles, Evaluator


class TestQAOAOptimize:
    @pytest.fixture
    def reg2_sub_3(self):
        """ Subgraph of a 2-regular graph with 4 nodes and 3 edges """
        graph = nx.read_gml('graphs/simple/reg2_sub_p1.gml', destringizer=int)
        return graph

    @pytest.fixture
    def reg2_4(self):
        """ 2-regular graph with 4 nodes and 4 edges """
        graph = nx.read_gml('graphs/simple/reg2_n4.gml', destringizer=int)
        return graph

    @pytest.fixture
    def reg2_sub_5(self):
        """ Subgraph of a 2-regular graph with 6 nodes and 5 edges """
        graph = nx.read_gml('graphs/simple/reg2_sub_p2.gml', destringizer=int)
        return graph

    @pytest.fixture
    def reg2_6(self):
        """ 2-regular graph with 6 nodes and 6 edges """
        graph = nx.read_gml('graphs/simple/reg2_n6.gml', destringizer=int)
        return graph

    @pytest.fixture
    def reg3_sub_tree(self):
        """ Tree-like subgraph of a 3-regular graph with 6 nodes and 5 edges """
        graph = nx.read_gml('graphs/simple/reg3_sub_tree.gml', destringizer=int)
        return graph

    @pytest.fixture
    def reg3_simple(self):
        """ 3-regular triangle-free graph with 6 nodes and 9 edges """
        graph = nx.read_gml('graphs/simple/reg3_n6_no_triangles.gml', destringizer=int)
        return graph

    def test_reg2_sub_3_qaoa_p1(self, reg2_sub_3):
        """ #1. Tests that expectation of 1 edge cut obtained with QAOA on a 2-regular subgraph with 3 edges matches the result reported in Farhi et al. for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg2_sub_3, 1, [(0, 1)], False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 3/4) < 1e-4

    def test_reg2_sub_3_qaoa_analytical(self, reg2_sub_3):
        """ #2. Tests that analytical version of QAOA gives the same answer as quantum simulation (compare to #1). """
        evaluator = Evaluator.get_evaluator_analytical(reg2_sub_3, [(0, 1)], False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 3/4) < 1e-4

    def test_reg2_4_qaoa_p1(self, reg2_4):
        """ #3. Tests that expectation of all edges cut obtained with QAOA on a 2-regular graph with even number of nodes is the same as expectation of
        one edge cut * number of edges (compare to #2). """
        evaluator = Evaluator.get_evaluator_analytical(reg2_4, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 3) < 1e-4

    def test_reg2_sub_3_qaoa_p2(self, reg2_sub_3):
        """ #4. Tests that expectation grows if p is increased (compare to #1). """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg2_sub_3, 2, [(0, 1)], False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 1) < 1e-4

    def test_reg2_sub_5_qaoa_p2(self, reg2_sub_5):
        """ #5. Tests that 1 edge cut expectation obtained with QAOA on a 2-regular subgraph with 5 edges matches the result reported in Farhi et al. for p=2. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg2_sub_5, 2, [(0, 1)], False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 5/6) < 1e-4

    def test_reg2_sub_3_maqaoa_p1(self, reg2_sub_3):
        """ #6. Tests that MA-QAOA is better than regular QAOA on one edge (compare to #1). """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg2_sub_3, 1, [(0, 1)])
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 1) < 1e-4

    def test_reg2_sub_3_maqaoa_analytical(self, reg2_sub_3):
        """ #7. Tests that analytical version of MA-QAOA gives the same answer as quantum simulation (compare to #6). """
        evaluator = Evaluator.get_evaluator_analytical(reg2_sub_3, [(0, 1)])
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 1) < 1e-4

    def test_reg2_4_maqaoa_p1(self, reg2_4):
        """ #8. Tests that MA-QAOA is still the same as regular QAOA on all edges (compare to #3 and #7). """
        evaluator = Evaluator.get_evaluator_analytical(reg2_4)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 3) < 1e-4

    def test_reg3_sub_tree_qaoa_p1(self, reg3_sub_tree):
        """ #9. Tests that 1 edge cut expectation obtained with QAOA on a 3-regular tree subgraph matches the result reported in Farhi et al. for p=1. """
        evaluator = Evaluator.get_evaluator_analytical(reg3_sub_tree, [(0, 1)], False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 0.6924) < 1e-4

    def test_reg3_simple_qaoa_p1(self, reg3_simple):
        """ #10. Tests that all edges cut expectation obtained with QAOA on a 3-regular graph matches result of #9 * number of edges. """
        evaluator = Evaluator.get_evaluator_analytical(reg3_simple, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 6.2321) < 1e-4

    def test_reg3_simple_maqaoa_p1(self, reg3_simple):
        """ #11. Tests that all edges cut expectation obtained with MA-QAOA on a 3-regular graph is better than the result on regular QAOA (#10). """
        evaluator = Evaluator.get_evaluator_analytical(reg3_simple)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 6.5) < 1e-4

    def test_reg3_simple_qaoa_p2(self, reg3_simple):
        """ #12. Tests that all edges cut expectation obtained with QAOA on a 3-regular graph is better than p=1 (#10), but still less than maximum (=9). """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg3_simple, 2, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 8.0198) < 1e-4

    def test_reg3_simple_maqaoa_p2(self, reg3_simple):
        """ #13. Tests that all edges cut expectation obtained with MA-QAOA on a 3-regular graph is better than p=1 (#11), still better than QAOA (#12) and reaches maximum = 9. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg3_simple, 2)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 9) < 1e-4

    def test_simulation_p1(self, reg3_simple):
        """ #14. Tests that the answer obtained with MA-QAOA by simulation is the same as the one obtained through the analytical expression (#11). """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg3_simple, 1)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 6.5) < 1e-4

    def test_subgraphs_optimum(self, reg3_simple):
        """ #15. Tests that the answer obtained through separation onto subgraphs is identical to the one obtained on the overall graph (#14). """
        evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(reg3_simple, 1)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 6.5) < 1e-4

    def test_subgraphs_arbitrary(self, reg3_simple):
        """ #16. Tests that the answer at an arbitrary non-optimum point is the same for full graph and subgraph simulations. """
        p = 1
        evaluator_full = Evaluator.get_evaluator_standard_maxcut(reg3_simple, p, use_multi_angle=False)
        evaluator_subgraphs = Evaluator.get_evaluator_standard_maxcut_subgraphs(reg3_simple, p, use_multi_angle=False)
        angles = np.array([np.pi / 8] * 2 * p)
        assert abs(evaluator_full.func(angles) - evaluator_subgraphs.func(angles)) < 1e-4
