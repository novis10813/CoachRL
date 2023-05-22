from gymnasium import ObservationWrapper


class TransposeWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        return obs.transpose((2, 1, 0))


def env_wrapper(env):
    env = TransposeWrapper(env)
    return env
