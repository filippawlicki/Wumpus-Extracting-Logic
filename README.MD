# Interpreting Deep Q-Networks: Rule-Based Comparison with First-Order Logic

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Prolog](https://img.shields.io/badge/SWI--Prolog-Symbolic%20AI-red.svg)](https://www.swi-prolog.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **A framework for bridging the gap between Subsymbolic (Deep Reinforcement Learning) and Symbolic (First-Order Logic) AI in the Wumpus World environment.**

---

## Scientific Context
This repository accompanies the research paper [Interpreting Deep Q-Networks: A Rule-Based Comparison with First-Order Logic in Wumpus World](https://journal.mostwiedzy.pl/TASKQuarterly/article/view/3640).

Deep Reinforcement Learning (DRL) agents, such as DQNs, achieve remarkable performance but often function as "black boxes". This project addresses the interpretability challenge by:
1.  **Training** DQN agents to survive and win in the Wumpus World.
2.  **Implementing** a "Gold Standard" Symbolic Agent using First-Order Logic (Prolog).
3.  **Extracting** symbolic rules from the trained DQN via surrogate Decision Trees.
4.  **Comparing** the neural and symbolic strategies using Jaccard similarity and structural metrics.

## Key Features

* **Custom Wumpus World Environment**: A grid-based environment compatible with OpenAI Gym interfaces, featuring sensation maps (Stench, Breeze, Glitter, etc.).
* **Deep Q-Network (DQN) Agent**:
    * Implemented in **PyTorch**.
    * Features Replay Buffer, Target Networks, and decaying Epsilon-Greedy exploration.
    * Supports state-based inputs and auxiliary memory maps.
* **Symbolic FOL Agent**:
    * Logic engine implemented in **Prolog** (`logic.pl`).
    * Maintains a dynamic Knowledge Base (KB) to infer safe cells and wumpus/pit locations.
    * Guarantees 100% survival rate on solvable maps.
* **Explainable AI (XAI) Module**:
    * Tools to generate datasets from agent trajectories.
    * Decision Tree training to mimic agent policies.
    * Visualization of extracted decision logic.

## Citation

If you use this code or methodology in your research, please cite the following paper:

```bibtex
@article{pawlicki2024interpreting,
  title={Interpreting Deep Q-Networks: A Rule-Based Comparison with First-Order Logic in Wumpus World},
  author={Filip Pawlicki, Kamil Dobies, Marcin Pucek, Karol Draszawka},
  journal={TASK Quarterly},
  volume={28},
  number={4},
  year={2024},
}
```
-----

*Developed at Department of Computer Architecture Faculty of Electronics Telecommunications and Informatics Gdańsk University of Technology.*