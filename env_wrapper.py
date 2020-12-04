import numpy as np
import os
import gym
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack


def make_atari_default(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = DummyVecEnv,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = AtariWrapper(env, **wrapper_kwargs)
        return env

    return VecFrameStack(make_vec_env_fix(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=atari_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    ), n_stack = 4)


def make_atari_continuous(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = SubprocVecEnv,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = ContWrapper(env, **wrapper_kwargs)
        return env

    return VecFrameStack(make_vec_env_fix(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=atari_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    ), n_stack = 4)




def make_vec_env_fix(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    # This is mostly a copy paste from the stable_baselines3 funciton with 
    # monitor wrapper applyed at the end for consistent logging in tensrboard
    Also, the HideScore wrapper is applied here

    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    
    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)

            # Hide the score
            env = HideScore(env)

            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed            
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


class ContWrapper(AtariWrapper):

    ##TODO mask score!

    """
    :param env_name: (str) name of the Atari environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """
    def __init__(self, env, max_steps=10**8, **kwargs):
        # Call the parent constructor, so we can access self.env later
        super(ContWrapper, self).__init__(env, **kwargs)

        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        #TODO remove score from observation
        self.current_step += 1
        obs, reward, done, info = super(ContWrapper, self).step(action)
        
        # Replace done with negative reward and keep the episode going
        if done:
            reward = -2
            done = False
            self.env.reset()
        # Overwrite the done signal when 
        if self.current_step >= self.max_steps:
            done = True
            # Update the info dict to signal that the limit was exceeded
            info['time_limit_reached'] = True
            self.current_step = 0
            self.reset()

        obs = np.array(obs, dtype = np.uint8)

        return obs, np.float(reward), np.bool(done), info

class Vec_reward_wrapper(VecEnvWrapper):
    """
    This wrapper changes the reward of the provided environment to some function
    of its observations

    r_model must be a callable function that takes batch of obervations
    and returns batch of rewards

    """

    def __init__(self, venv, r_model):
        VecEnvWrapper.__init__(self, venv)
        assert callable(r_model)
        self.r_model = r_model

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        proxy_rew = self.r_model(obs)
        return obs, proxy_rew, dones, infos


class Reward_wrapper(gym.Wrapper):

    def __init__(self, env, r_model):
        super(Reward_wrapper, self).__init__(env)
        assert callable(r_model)
        self.r_model = r_model


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.r_model(observation), done, info

from stable_baselines3.common.vec_env import VecEnvWrapper
class Vec_reward_wrapper(VecEnvWrapper):
    """
    This wrapper changes the reward of the provided environment to some function
    of its observations

    r_model must be a callable function that takes batch of obervations
    and returns batch of rewards

    """

    def __init__(self, venv, r_model):
        VecEnvWrapper.__init__(self, venv)
        assert callable(r_model)
        self.r_model = r_model

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        proxy_rew = self.r_model(obs)
        return obs, proxy_rew, dones, infos



class HideScore(gym.ObservationWrapper):

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        Hides the score in the unwrapped gym environment
        """
        if self.spec.id == 'BeamRiderNoFrameskip-v4':
            frame[183:190, 32:46] = [0,0,0]
            frame[10:18, 61:107] = [0,0,0]
            frame[32:40, 20:31] = [0,0,0]
        elif self.spec.id == 'PongNoFrameskip-v4':
            frame[:24,:] = [236,236,236]
        elif self.spec.id == 'SeaquestNoFrameskip-v4':
            frame[:20,:] = [0,0,0]
            
        return frame
