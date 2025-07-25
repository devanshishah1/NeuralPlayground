import os
import pickle

import numpy as np
import pandas as pd
from deepdiff import DeepDiff #comparing two different python objects
from gymnasium import Env, spaces #Env is a class in gymnasium that we inherit from

# importing neccessary modules and libraries
# gymnasium is  a library important for "reinforcement learning" 
# Env: https://gymnasium.farama.org/api/env/


class Environment(Env):
    #description of the class
    """Abstract parent environment class

    Attributes
    ----------
    environment_name : str
        Name of the specific instantiation of the environment class
    state: array
        Empty array for this abstract class
    history: list
        list containing transition history
    time_step_size: float
        time step in second traverse when calling step method
    metadata: dict
        dictionary with extra metadata that might be available in other classes
    env_kwags: dict
        Arguments given to the init method
    global_steps: int
        number of calls to step method, set to 0 when calling reset
    global_time: float
        time simulating environment through step method, then global_time = time_step_size * global_steps
    observation_space: gym.spaces
        specify the range of observations as in openai gym
    action_space: gym.spaces
        specify the range of actions as in openai gym

    Methods
    ----------
    __init__(self, environment_name: str = "Environment", **env_kwargs):
        Initialize the class. env_kwargs arguments are specific for each of the child environments and
        described in their respective class
    make_observation(self):
        Returns the current state of the environment
    step(self, action):
        Runs the environment dynamics. Given some action, return observation, new state and reward.
    reset(self):
        Re-initialize state and global counters. Returns observation and re-setted state
    save_environment(self, save_path: str):
        Save current variables of the object to re-instantiate the environment later
    restore_environment(self, save_path: str):
        Restore environment saved using save_environment method
    get_trajectory_data(self):
        Returns interaction history
    reward_function(self, action, state):
        Reward curriculum as a function of action, state
        and attributes of the environment
    """

    def __init__(self, environment_name: str = "Environment", time_step_size: float = 1.0, **env_kwargs):
        """Initialisation of Environment class

        Parameters
        ----------
        environment_name: str
            environment name for the specific instantiation of the object
        env_kwargs: dict
            Define within each subclass for specific environments
            time_step_size: float
               Size of the time step in seconds

        """
        #**env_kwargs collects any keyword arguments that don't already have a specific parameter and packs them into a dictionary.

        self.environment_name = environment_name
        self.time_step_size = time_step_size
        self.env_kwargs = env_kwargs  # Parameters of the environment
        self.metadata = {"env_kwargs": env_kwargs}  # Define within each subclass for specific environments
        self.state = np.array([])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float64)
        self.history = []
        # Initializing global counts
        self.global_steps = 0
        self.global_time = 0

    def make_observation(self):
        """Just take the state and returns an array of sensory information
        In more complex cases, the observation might be different from the internal state of the environment

        Returns
        -------
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the environment (eg. position in the environment)
        """
        return self.state

    def step(self, action=None):
        """Runs the environment dynamics. Increasing global counters.
        Given some action, return observation, new state and reward.

        Parameters
        ----------
        action:
            Abstract argument representing the action from the agent

        Returns
        -------
        observation:
            Define within each subclass for specific environments
            Any set of observation that is made by the agent in the environment (position, visual features,...)
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the environment (eg.position in the environment)
        reward: float
            The reward that the animal receives in this state transition
        """
        observation = self.make_observation()  # Build sensory info from current state
        reward = self.reward_function(action, self.state)  # If you get reward, it should be coded here
        self._increase_global_step()
        new_state = self.state  # Define within each subclass for specific environments
        transition = {
            "action": action,
            "state": self.state,
            "next_state": new_state,
            "reward": reward,
            "step": self.global_steps,
        }
        self.history.append(transition)
        # state should be updated as well

        return observation, self.state, reward

    def _increase_global_step(self):
        self.global_steps += 1
        self.global_time += self.time_step_size

    def reset(self):
        """Re-initialize state. Returns observation and re-setted state

        Returns
        -------
        observation:
            Define within each subclass for specific environments.
            Any set of observation that is made by the agent in the environment (position, visual features,...)
        self.state:
            Define within each subclass for specific environments
            Variable containing the state of the environment (eg.position in the environment)
        """
        self.global_steps = 0
        self.global_time = 0
        observation = self.make_observation()
        return observation, self.state

    def save_environment(self, save_path: str, raw_object: bool = True):
        """Save current variables of the object to re-instantiate the environment later

        Parameters
        ----------
        save_path: str
            Path to save the environment
        raw_object: bool
            If True, save the raw object, otherwise save the dictionary of attributes
            If True, you can load the object by using env = pd.read_pickle(save_path)
            if False, you can load the object by using env.restore_environment(save_path)
        """
        if raw_object:
            pickle.dump(self, open(os.path.join(save_path), "wb"), pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(self.__dict__, open(os.path.join(save_path), "wb"), pickle.HIGHEST_PROTOCOL)

    def restore_environment(self, restore_path: str):
        """Restore environment saved using save_environment method

        Parameters
        ----------
        restore_path: str
            Path to retrieve the environment saved using save_environment method (raw_object=False)
        """
        self.__dict__ = pd.read_pickle(restore_path)

    def __eq__(self, other):
        """Check if two environments are equal by comparing all of its attributes

        Parameters:
        ----------
        other: Environment
            Another instantiation of the environment
        return: bool
            True if self and other are the same exact environment
        """
        diff = DeepDiff(self.__dict__, other.__dict__)
        if len(diff) == 0:
            return True
        else:
            return False

    def get_trajectory_data(self):
        """Returns interaction history"""
        return self.history

    def reward_function(self, action, state):
        """Reward curriculum as a function of action, state
        and attributes of the environment

        Parameters
        ----------
        action:
            same as step method argument
        state:
            same as state attribute of the class

        Returns
        -------
        reward: float
            reward given the parameters
        """
        reward = 0
        return reward


if __name__ == "__main__":
    env = Environment(environment_name="test_environment", time_step_size=0.5, one_kwarg_argument=10)
    print(env.__dict__)
    print(env.__dir__())
