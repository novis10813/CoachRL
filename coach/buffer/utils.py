from typing import Union, Tuple, Dict

import numpy as np
import gymnasium as gym


def get_obs_shape(
    observation_space: gym.spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    if isinstance(observation_space, gym.spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, gym.spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for key, subspace in observation_space.spaces.items()
        }
    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_action_dim(action_space: gym.spaces.Space) -> int:
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        return 1
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return int(len(action_space.nvec))
    elif isinstance(action_space, gym.spaces.MultiBinary):
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return device
