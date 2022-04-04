# LearningGNNExplanations

This repository contains the code for a project that belongs to the area of computing explanations for graph neural networks. I analyse and refine explanations for predictions from a deep Q-learning agent in the Taxi-v3 environment [2]. To be able to use this environment for my research topic, I convert the default integer observations to graph observations. This yields sparse feature matrices. 

Naively using explanations given by Zorro [1] results in bad explanations because of the sparsity of the data. That is why I extend the deep Q-network with an explanation branch. I train the explanation branch while freezing all other weights by minimizing a fidelity value comparable to the widely known Fidelty-Minus evaluation metric. The size of the output of the explanation branch equals the size of the graph observation. However, this embedding is not sparse. Therefor, I am able to use Zorro in an improved way compared to naively applying it to the original graph observation.

![](https://github.com/andreasMazur/RefinedGNNExplanations/blob/main/Experiment.gif)

# References

[1] Funke, Thorben, Megha Khosla, and Avishek Anand. "Hard masking for explaining graph neural networks." (2020).

[2] T Erez, Y Tassa, E Todorov, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition", 2011.
