from matplotlib import pyplot as plt

import numpy as np


if __name__ == "__main__":
    test_set_name = "test_set_2"
    array = np.load(f"./checkpoints/{test_set_name}.npy")
    beginning = 0
    for idx, x in enumerate(array):
        if x <= 2:
            beginning = idx
            break
    plt.figure(figsize=(12, 6))
    plt.title(
        f"Name: {test_set_name} - "
        f"Minimum: {array.min():.3f} - "
        f"Mean: {np.mean(array[beginning:]):.3f}"
    )
    plt.ylabel("Average episode fidelity")
    plt.xlabel("Episode")
    plt.grid()
    plt.plot(array[beginning:])
    plt.savefig(f"./checkpoints/{test_set_name}_fidelities.png")
    plt.show()
