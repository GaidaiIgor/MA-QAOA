# MA-QAOA
Implementation of the original Quantum Approximate Optimization Algorithm (QAOA) and Multi-Angle QAOA (MA-QAOA).

Both algorithms are designed for approximate solutions to combinatorial optimization problems on quantum computers. 

This implementation is purely classical, so it only calculates expectation values, but does not give explicit answers to the combinatorial problems (one needs a quantum computer for that).

The main function is `run_qaoa_main` from `run_qaoa.py`. Run parameters and input graph are specified in the code at the beginning of `run_qaoa_main`.

For more info on QAOA refer to https://arxiv.org/abs/1411.4028v1

For more info on MA-QAOA refer to https://www.nature.com/articles/s41598-022-10555-8
