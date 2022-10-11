# Refined Zorro Explanations for Graph Neural Networks

This repository contains the code for the paper:

```
@article{Mazur2022,
  title={Improving Zorro Explanations for Sparse Observations with Dense Proxy Data},
  author={Mazur, Andreas and Artelt, André and Hammer, Barbara},
  journal={ESANN},
  doi={https://doi.org/10.14428/esann/2022.ES2022-27},
  pages={527-532},
  year={2022},
  publisher={i6doc.com}
}
```

We analyse and refine explanations for predictions from a deep Q-learning agent in the Taxi-v3 environment [2].
Default integer observations are converted to graph observations. However, those appear to contain sparse feature matrices.
Naively applying Zorro [1] onto the sparse observations results in problematic explanations.
That is why we extend the deep Q-network with a proxy branch.
We train the proxy branch by minimizing a fidelity value comparable to the widely known Fidelty-Minus evaluation metric.
Simultaneously we freeze the weights of the deep Q-network.
The proxy branch outputs dense proxy data which we can use to substitute the sparse original observations.
The explanations retrieved by applying Zorro on the dense proxy data appear to be better interpretable than the initially computed explanations
for the sparse observations.

<!-- ![](https://github.com/andreasMazur/RefinedGNNExplanations/blob/main/Experiment.gif) -->

# Install

This experiment was conducted with Python3.9. Further, install the requirements:

```pip install -r requirements.txt```

If you want to repeat the experiment yourself, then you need to execute the `experiment.py`-script in the repository's root directory.
It will automatically load a pre-trained network with its added and trained proxy-branch. If you want to repeat everything
from scratch, you must:
1. Train a reinforcement learning agent by executing the ``train_agent.train_agent``-script
2. Train the explanation branch by executing the ``learn_proxies.grid_search``-script (requires trained agent from previous step)
3. Conduct experiment by executing `experiment`-script

# References

[1] Funke, Thorben, Megha Khosla, and Avishek Anand. "Hard masking for explaining graph neural networks." (2020).

[2] T Erez, Y Tassa, E Todorov, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition", 2011.
