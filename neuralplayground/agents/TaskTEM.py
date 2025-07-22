
import numpy as np
import random

from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters


class TaskAgent(Whittington2020):

    """
    This TEM agent follows a custom batch_act policy where it navigates to a series of target points and
    then performs a specific sequence of moves (up and right) before
    selecting a new target and repeating.
    """

    def __init__(self, **agent_params):
    
        # A list of target points set in simulation script.
        self.points = [np.array(p) for p in agent_params.pop("points", [])]
        if len(self.points) < 2:
            raise ValueError("Please provide at least 2 points for the agent to navigate between.")

        # Call the parent class constructor
        super().__init__(**agent_params)

        # Initialize targets for each agent in the batch
        self.current_targets = [random.choice(self.points) for _ in range(self.batch_size)]

        # This counter tracks the agent's current goal in the task.
        # 0: MOVING_TO_POINT - Agent is navigating to its current_target.
        # 1: MOVING_UP - Agent will now move up.
        # 2: MOVING_RIGHT - Agentwill now move right.
        self.special_move_counter = [0] * self.batch_size

    def batch_act(self, observations):
  
        new_actions = []
        for i in range(self.batch_size):
            current_pos = np.array(observations[i][2])
            state = self.special_move_counter[i]
            action = [0, 0]  # default action is to stay still

            # [0]

            if state == 0:  # MOVING_TO_POINT
                target_pos = self.current_targets[i]
                # check if the agent has reached the target (or close enough)
                if np.linalg.norm(current_pos - target_pos) < 1.0:
                    # if it has, set up for state = [1]
                    action = [0, 1]
                    self.special_move_counter[i] = 1
                else:
                    # if not at the target, move to it.
                    delta = target_pos - current_pos
                    if abs(delta[0]) > abs(delta[1]):
                        action = [np.sign(delta[0]), 0]  # horizontally
                    elif abs(delta[1]) > 0:
                        action = [0, np.sign(delta[1])]  # vertically

            elif state == 1:  
                #check that moving up worked
                prev_pos = np.array(self.prev_observations[i][2])
                if np.array_equal(current_pos, prev_pos) and self.prev_actions[i] == [0, -1]:
                    print(f"Agent {i} out of range: 'UP' move failed.")

                # next action for the next state is 'RIGHT'.
                action = [1, 0]
                self.special_move_counter[i] = 2

            elif state == 2:  
                #check moving right worked
                prev_pos = np.array(self.prev_observations[i][2])
                if np.array_equal(current_pos, prev_pos) and self.prev_actions[i] == [1, 0]:
                    print(f"Agent {i} out of range: 'RIGHT' move failed.")

                # reset state and find a new target.
                self.special_move_counter[i] = 0
                
                # exclude the current target from the list of possible next targets.
                current_target_tuple = tuple(self.current_targets[i])
                possible_next_targets = [p for p in self.points if tuple(p) != current_target_tuple]
                new_target = random.choice(possible_next_targets)
                self.current_targets[i] = new_target
                
                # The action for the current step is to start moving towards the new target.
                delta = new_target - current_pos
                if abs(delta[0]) > abs(delta[1]):
                    action = [np.sign(delta[0]), 0]  # Move horizontally
                elif abs(delta[1]) > 0:
                    action = [0, np.sign(delta[1])]  # Move vertically

            

            # The DiscreteObjectEnvironment flips the y-axis for actions.
            # We need to counteract this for the action to have the intended effect.
            action_to_send = list(action)  # Make a copy
            if action_to_send[0] == 0:
                action_to_send[1] = -action_to_send[1]

            new_actions.append(action_to_send)

        # This part is crucial for the TEM model to learn. It's inherited
        # from the base Whittington2020 agent's random walk policy. We must
        # maintain the history of observations and actions for the model's update step.
        self.walk_actions.append(self.prev_actions.copy())
        self.obs_history.append(self.prev_observations.copy())
        self.prev_actions = new_actions
        self.prev_observations = observations
        self.n_walk += 1

        return new_actions
