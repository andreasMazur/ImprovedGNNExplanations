from matplotlib import pyplot as plt

import numpy as np


if __name__ == "__main__":
    """Function used to visualize experiment results."""

    # Fidelities per episode step from episode 0 in ExperimentImages
    fid_ref_mem = [
        .342, .342, .3, .188,
        .164, .044, .365, .051,
        .157, .701, .203, .065,
        .074, .445, .043, .16,
        .29
    ]
    fid_zo_mem = [
        76.344, 66.258, 44.756, 14.823,
        8.944, 5.173, 2.873, 20.114,
        7.624, 12.838, 21.909, 42.556,
        71.521, 63.359, 122.637, 198.016,
        270.338
    ]
    fid_zn_mem = [
        81.210, 68.320, 51.451, 18.969,
        14.161, 4.97, 3.984, 29.913,
        7.492, .88, 9.052, 32.808,
        .074, 47.972, 98.347, 129.869,
        270.338
    ]

    # Action correct or not per episode step from episode 0 in ExperimentImages
    action_correct_ref = [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    action_correct_zo = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    action_correct_zn = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]

    # Plot fidelities
    plt.figure(figsize=(12, 5))
    plt.title("Fidelity Comparison")
    plt.ylabel("Step fidelity")
    plt.xlabel("Episode Step")
    plt.grid()
    plt.plot(fid_ref_mem, label="Fidelity Experiment Branch")
    plt.plot(fid_zo_mem, label="Fidelity Zorro Original")
    plt.plot(fid_zn_mem, label="Fidelity Zorro Noisy")
    plt.legend()
    plt.savefig("./ExperimentImages/FidelityComparison.svg", format="svg")
    plt.show()

    # Plot correct actions
    action_correct_ref = np.sum(action_correct_ref)
    action_correct_zo = np.sum(action_correct_zo)
    action_correct_zn = np.sum(action_correct_zn)
    data = {
        "Network Explanation": action_correct_ref,
        "Zorro Original": action_correct_zo,
        "Zorro Noisy": action_correct_zn
    }
    plt.figure(figsize=(8, 5))
    plt.title("Action Comparison")
    plt.grid()
    plt.bar(list(data.keys()), list(data.values()), width=.3)
    plt.xlabel("Experiment")
    plt.ylabel("Correctly predicted actions")
    plt.savefig("./ExperimentImages/ActionComparison.svg", format="svg")
    plt.show()
