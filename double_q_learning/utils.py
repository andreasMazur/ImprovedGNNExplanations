from matplotlib import pyplot as plt

import numpy as np


def plot_stats(file_name, path, exploration_rate, target_updates_=None, plot=False, **kwargs):
    """Plot the stats of a training procedure

    Parameters
    ----------
    file_name: str
        The file name of the plot
    path: str
        Where to store the plot
    exploration_rate: list
        An array containing the last exploration rates of the passed episodes
    target_updates_: list
        The episode numbers in which the target network has been updated
    plot: boolean
        Whether to display the plot or not
    kwargs:
        The arrays which shall be plotted (see an example in `train_agent.py`)
    """

    if len(kwargs) == 1:
        fig_height = 6
    else:
        fig_height = len(kwargs) * (8 / 3)
    fig, axes = plt.subplots(len(kwargs), 1, figsize=(12, fig_height))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    i = 0
    for key, value in kwargs.items():

        # Plot time stamps where target model is updated.
        if target_updates_:
            for update_episode in target_updates_:
                plt.axvline(x=update_episode, color="red", alpha=0.1)

        # Plot wanted values
        axes[i].plot(value, label=key)
        axes[i].set_ylabel(key)
        axes[i].grid()
        axes[i].legend(loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=3)

        # Plot exploration rate
        expl_ax = axes[i].twinx()
        expl_ax.plot(exploration_rate, label="Expl. Rate", color="orange")
        expl_ax.set_ylabel("Explr. Rate")

        if i == len(kwargs) - 1:
            axes[i].set_xlabel("Episodes")
        i += 1

    plt.subplots_adjust(hspace=.5)

    plt.savefig(f"{path}/{file_name}")
    if plot:
        plt.show()
    else:
        plt.draw()
    plt.close()
