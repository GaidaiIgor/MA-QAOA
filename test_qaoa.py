"""
QAOA tests
"""
import networkx as nx
import pytest

from src.optimization import optimize_qaoa_angles


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
        """ #1. Tests that QAOA <Cuv> (one edge) on a 2-regular subgraph with 3 edges matches the result reported in Farhi et al. for p=1. """
        objective_best = optimize_qaoa_angles(False, False, 1, reg2_sub_3, [(0, 1)])[0]
        assert abs(objective_best - 3/4) < 1e-4

    def test_reg2_sub_3_qaoa_analytical(self, reg2_sub_3):
        """ #2. Tests that analytical version of QAOA gives the same answer as quantum simulation (compare to #1). """
        objective_best = optimize_qaoa_angles(False, True, 1, reg2_sub_3, [(0, 1)])[0]
        assert abs(objective_best - 3/4) < 1e-4

    def test_reg2_4_qaoa_p1(self, reg2_4):
        """ #3. Tests that QAOA <C> (all edges) on a 2-regular graph with even number of nodes is the same as <Cuv> * number of edges (compare to #2). """
        objective_best = optimize_qaoa_angles(False, True, 1, reg2_4, None)[0]
        assert abs(objective_best - 3) < 1e-4

    def test_reg2_sub_3_qaoa_p2(self, reg2_sub_3):
        """ #4. Tests that <Cuv> grows if p is increased (compare to #1). """
        objective_best = optimize_qaoa_angles(False, False, 2, reg2_sub_3, [(0, 1)])[0]
        assert abs(objective_best - 1) < 1e-4

    def test_reg2_sub_5_qaoa_p2(self, reg2_sub_5):
        """ #5. Tests that QAOA <Cuv> on a 2-regular subgraph with 5 edges matches the result reported in Farhi et al. for p=2. """
        objective_best = optimize_qaoa_angles(False, False, 2, reg2_sub_5, [(0, 1)])[0]
        assert abs(objective_best - 5/6) < 1e-4

    def test_reg2_sub_3_maqaoa_p1(self, reg2_sub_3):
        """ #6. Tests that MA-QAOA is better than regular QAOA on one edge (compare to #1). """
        objective_best = optimize_qaoa_angles(True, False, 1, reg2_sub_3, [(0, 1)])[0]
        assert abs(objective_best - 1) < 1e-4

    def test_reg2_sub_3_maqaoa_analytical(self, reg2_sub_3):
        """ #7. Tests that analytical version of MA-QAOA gives the same answer as quantum simulation (compare to #6). """
        objective_best = optimize_qaoa_angles(True, True, 1, reg2_sub_3, [(0, 1)])[0]
        assert abs(objective_best - 1) < 1e-4

    def test_reg2_4_maqaoa_p1(self, reg2_4):
        """ #8. Tests that MA-QAOA is still the same as regular QAOA on all edges (compare to #3 and #7). """
        objective_best = optimize_qaoa_angles(True, True, 1, reg2_4, None)[0]
        assert abs(objective_best - 3) < 1e-4

    def test_reg3_sub_tree_qaoa_p1(self, reg3_sub_tree):
        """ #9. Tests that QAOA <Cuv> on a 3-regular tree subgraph matches the result reported in Farhi et al. for p=1. """
        objective_best = optimize_qaoa_angles(False, True, 1, reg3_sub_tree, [(0, 1)])[0]
        assert abs(objective_best - 0.6924) < 1e-4

    def test_reg3_simple_qaoa_p1(self, reg3_simple):
        """ #10. Tests that QAOA <C> on a 3-regular graph matches result of #9 * number of edges. """
        objective_best = optimize_qaoa_angles(False, True, 1, reg3_simple, None)[0]
        assert abs(objective_best - 6.2321) < 1e-4

    def test_reg3_simple_maqaoa_p1(self, reg3_simple):
        """ #11. Tests that MA-QAOA <C> on a 3-regular graph is better than the result on regular QAOA (#10). """
        objective_best = optimize_qaoa_angles(True, True, 1, reg3_simple, None)[0]
        assert abs(objective_best - 6.5) < 1e-4

    def test_reg3_simple_qaoa_p2(self, reg3_simple):
        """ #12. Tests that QAOA <C> on a 3-regular graph is better than p=1 (#10), but still less than maximum (=9). """
        objective_best = optimize_qaoa_angles(False, False, 2, reg3_simple, None)[0]
        assert abs(objective_best - 8.0198) < 1e-4

    def test_reg3_simple_maqaoa_p2(self, reg3_simple):
        """ #13. Tests that MA-QAOA <C> on a 3-regular graph is better than p=1 (#11), still better than QAOA (#12) and reaches maximum = 9. """
        objective_best = optimize_qaoa_angles(True, False, 2, reg3_simple, None)[0]
        assert abs(objective_best - 9) < 1e-4
