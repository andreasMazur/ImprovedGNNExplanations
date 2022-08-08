from gym.envs.toy_text import TaxiEnv

from preprocessing import preprocess


class AdvancedTaxiEnv(TaxiEnv):
    """The Taxi-environment altered to return graph-observations

    Attributes
    ----------
    last_raw_state: int
        The environment description from the previous step in integer-format
    step_count: int
        The amount of steps which were already taken in the current episode

    Methods
    -------
    reset()
        Resets the environment to a new episode
    step(action)
        Advanced by one step in the environment w.r.t. the given action
    """

    def __init__(self):
        super(AdvancedTaxiEnv, self).__init__()
        self.last_raw_state = None
        self.step_count = 0

    def reset(self, **kwargs):
        state_raw = super(AdvancedTaxiEnv, self).reset()
        self.last_raw_state = state_raw
        self.step_count = 0
        return preprocess(self, state_raw)

    def step(self, action):
        """
        Parameters
        ----------
        action: int
            The action to execute in the environment
        """
        state, reward, done, info = super(AdvancedTaxiEnv, self).step(action)
        self.step_count += 1
        done = True if self.step_count >= 200 else done
        self.last_raw_state = state
        state = preprocess(self, state)
        return state, reward, done, info
