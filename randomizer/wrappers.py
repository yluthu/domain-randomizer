import gym
import json
import numpy as np

import gym.spaces as spaces

from randomizer.dimension import Dimension
import re


class RandomizedEnvWrapper(gym.Wrapper):
    """Creates a randomization-enabled enviornment, which can change
    physics / simulation parameters without relaunching everything
    """

    def __init__(self, env, seed):
        super(RandomizedEnvWrapper, self).__init__(env)
        self.config_file = self.unwrapped.config_file

        self._load_randomization_dimensions(seed)
        self.env.update_randomized_params()
        self.randomized_default = ['random'] * len(self.unwrapped.dimensions)

    def _load_randomization_dimensions(self, seed):
        """ Helper function to load environment defaults ranges
        """
        self.unwrapped.dimensions = []

        with open(self.config_file, mode='r') as f:
            config = json.load(f)

        for dimension in config['dimensions']:
            self.unwrapped.dimensions.append(
                Dimension(
                    default_value=dimension['default'],
                    multiplier_min=dimension['multiplier_min'],
                    multiplier_max=dimension['multiplier_max'],
                    name=dimension['name']
                )
            )

        nrand = len(self.unwrapped.dimensions)
        self.unwrapped.randomization_space = spaces.Box(0, 1, shape=(nrand,), dtype=np.float32)

    def randomize(self, randomized_values=[-1]):
        """Sets the parameter values such that a call to`update_randomized_params()`
        will generate an environment with those settings.

        Passing a list of 'default' strings will give the default value
        Passing a list of 'random' strings will give a purely random value for that dimension
        Passing a list of -1 integers will have the same effect.
        """
        for dimension, randomized_value in enumerate(randomized_values):
            if randomized_value == 'default':
                self.unwrapped.dimensions[dimension].current_value = \
                    self.unwrapped.dimensions[dimension].default_value
            elif randomized_value != 'random' and randomized_value != -1:
                assert 0.0 <= randomized_value <= 1.0, "using incorrect: {}".format(randomized_value)
                self.unwrapped.dimensions[dimension].current_value = \
                    self.unwrapped.dimensions[dimension].rescale(randomized_value)
            else:  # random
                self.unwrapped.dimensions[dimension].randomize()

        self.env.update_randomized_params()

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class RandomActionDelayWrapper(gym.Wrapper):
    """Add a random action delay to any env.

    The wrapper uses two randomized dimensions "action_delay_mean" and
    "action_delay_std" in the JSON config file. Each time step() is called,
    the specified action is inserted into an action queue with a random delay
    defined as floor(randn(action_delay_mean, action_delay_std)).
    """

    def __init__(self, env):
        super(RandomActionDelayWrapper, self).__init__(env)
        self.config_file = self.unwrapped.config_file
        self.action_delay_mean = None
        self.action_delay_std = None
        self.action_queue = [self.unwrapped.action_space.sample()]

    def update_randomized_params(self):
        self.action_delay_mean, self.action_delay_std = None, None
        for dimension in self.unwrapped.dimensions:
            if dimension.name == "action_delay_mean":
                self.action_delay_mean = dimension.current_value
            elif dimension.name == "action_delay_std":
                self.action_delay_std = dimension.current_value
            if self.action_delay_mean is not None and self.action_delay_std \
                    is not None:
                break
        self.env.update_randomized_params()

    def step(self, action):
        delay = max(int(np.floor(self.action_delay_mean
                                 + np.random.randn() * self.action_delay_std)),
                    0)
        while len(self.action_queue) < delay:
            self.action_queue.append(self.action_queue[-1])
        self.action_queue = self.action_queue[:delay] + [action]
        if len(self.action_queue) > 1:
            return self.env.step(self.action_queue.pop(0))
        else:
            return self.env.step(self.action_queue[0])

    def reset(self, **kwargs):
        self.action_queue = [self.unwrapped.action_space.sample()]
        return self.env.reset(**kwargs)


class ObsSystematicErrorWrapper(gym.Wrapper):
    """Apply a multiplicative bias to the observations.

    The bias is kept constant through one randomization, and varies from
    randomization to randomization. The wrapper uses randomized dimensions
    "obs_sys_err_multiplier_0", "obs_sys_err_multiplier_1", etc. in the JSON
    file, where each randomized dimension stands for the multiplicative factor
    applied to one dimension of the observation.
    """

    def __init__(self, env):
        super(ObsSystematicErrorWrapper, self).__init__(env)
        self.config_file = self.unwrapped.config_file
        self.mult_bias = np.ones(self.unwrapped.observation_space.shape)
        self.obs_dim = self.mult_bias.size

    def update_randomized_params(self):
        for dimension in self.unwrapped.dimensions:
            try:
                idx = int(re.findall(r'obs_sys_err_multiplier_(\d+)$',
                                     dimension.name)[0])
                assert 0 <= idx < self.obs_dim
                self.mult_bias.flat[idx] = dimension.current_value
            except (IndexError, AssertionError):
                pass
        self.env.update_randomized_params()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.clip(obs * self.mult_bias,
                      self.unwrapped.observation_space.low,
                      self.unwrapped.observation_space.high)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ObsRandomErrorWrapper(gym.Wrapper):
    """Emulate a random multiplicative error applied to the observations.

    The error varies from step to step. The wrapper uses randomized dimensions
    "obs_rand_err_std_0", "obs_rand_err_std_1", etc. in the JSON file, where
    each randomized dimension stands for the standard deviation of the
    multiplicative factor applied to one dimension of the observation.
    For example, the multiplicative factor applied to obs[0] is drawn from
    Normal(1.0, obs_rand_err_std_0).
    """

    def __init__(self, env):
        super(ObsRandomErrorWrapper, self).__init__(env)
        self.config_file = self.unwrapped.config_file
        self.std = np.zeros(self.unwrapped.observation_space.shape)
        self.obs_dim = self.std.size

    def update_randomized_params(self):
        for dimension in self.unwrapped.dimensions:
            try:
                idx = int(re.findall(r'obs_rand_err_std_(\d+)$',
                                     dimension.name)[0])
                assert 0 <= idx < self.obs_dim
                self.std.flat[idx] = dimension.current_value
            except (IndexError, AssertionError):
                pass
        self.env.update_randomized_params()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        factor = np.ones_like(self.std) + self.std * \
            np.random.randn(*self.std.shape)
        obs = np.clip(obs * factor,
                      self.unwrapped.observation_space.low,
                      self.unwrapped.observation_space.high)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
