from gym.envs.toy_text import TaxiEnv

from preprocessing import preprocess


class AdvancedTaxiEnv(TaxiEnv):

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
        state, reward, done, info = super(AdvancedTaxiEnv, self).step(action)
        self.step_count += 1
        done = True if self.step_count >= 200 else done
        self.last_raw_state = state
        state = preprocess(self, state)
        return state, reward, done, info
