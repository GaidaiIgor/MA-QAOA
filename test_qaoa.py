"""
QAOA tests.
"""
import networkx as nx
import pytest

from src.optimization import optimize_qaoa_angles, Evaluator


class TestMAQAOA:
    @pytest.fixture
    def reg3_sub_tree(self):
        """ Tree-like subgraph of a 3-regular graph with 6 nodes and 5 edges. """
        graph = nx.read_gml('graphs/other/simple/reg3_sub_tree.gml', destringizer=int)
        return graph

    @pytest.fixture
    def reg4_n7_e14(self):
        """ Slightly harder 4-regular graph with 7 nodes and 14 edges (includes triangles). """
        graph = nx.read_gml('graphs/other/simple/reg4_n7_e14.gml', destringizer=int)
        return graph

    def test_qaoa_simple_edge(self, reg3_sub_tree):
        """ Tests that 1 edge cut expectation obtained with QAOA on a 3-regular tree subgraph matches the result reported in Farhi et al. for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg3_sub_tree, 1, [(0, 1)], False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 0.6924) < 1e-4

    def test_qaoa_simple_full(self, reg3_sub_tree):
        """ Tests all edges cut expectation obtained with QAOA on a 3-regular tree subgraph for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg3_sub_tree, 1, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 3.742) < 1e-3

    def test_qaoa(self, reg4_n7_e14):
        """ Tests cut expectation obtained with QAOA on a 4-regular graph with triangles for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg4_n7_e14, 1, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 8.735) < 1e-3

    def test_qaoa_analytical(self, reg4_n7_e14):
        """ Tests analytical cut expectation obtained with QAOA on a 4-regular graph with triangles for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut_analytical(reg4_n7_e14, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 8.735) < 1e-3

    def test_qaoa_subgraphs(self, reg4_n7_e14):
        """ Tests subgraph cut expectation obtained with QAOA on a 4-regular graph with triangles for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(reg4_n7_e14, 1, use_multi_angle=False)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 8.735) < 1e-3

    def test_ma_qaoa(self, reg4_n7_e14):
        """ Tests cut expectation obtained with MA-QAOA on a 4-regular graph with triangles for p=1. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg4_n7_e14, 1)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 10) < 1e-2

    def test_ma_qaoa_analytical(self, reg4_n7_e14):
        """ Tests that cut expectation obtained with analytical expression for MA-QAOA on a 3-regular tree subgraph is the same as simulation. """
        evaluator = Evaluator.get_evaluator_standard_maxcut_analytical(reg4_n7_e14)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 10) < 1e-2

    def test_ma_qaoa_subgraphs(self, reg4_n7_e14):
        """ Tests that cut expectation obtained with subgraph evaluation for MA-QAOA is the same as full simulation. """
        evaluator = Evaluator.get_evaluator_standard_maxcut_subgraphs(reg4_n7_e14, 1)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 10) < 1e-2

    def test_ma_qaoa_p2(self, reg4_n7_e14):
        """ Tests cut expectation obtained with MA-QAOA on a 4-regular graph with triangles for p=2. """
        evaluator = Evaluator.get_evaluator_standard_maxcut(reg4_n7_e14, 2)
        objective_best = optimize_qaoa_angles(evaluator)[0]
        assert abs(objective_best - 12) < 1e-2
