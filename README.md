# MA-QAOA
Implementation of the original Quantum Approximate Optimization Algorithm (QAOA) and Multi-Angle QAOA (MA-QAOA).

Both algorithms are designed to find approximate solutions to combinatorial optimization problems on quantum computers. 

This implementation is purely classical, so it only calculates expectation values and does not give explicit answers to the combinatorial problems (since this usually requires a quantum computer).
It works on arbitrary weighted graphs, including graphs with triangles.

To run the code, run `run_qaoa.py`. Run parameters and input graph can be specified in the code.

For more info on QAOA refer to https://arxiv.org/abs/1411.4028v1

For more info on MA-QAOA refer to https://www.nature.com/articles/s41598-022-10555-8
