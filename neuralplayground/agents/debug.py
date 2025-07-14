import numpy as np
from neuralplayground.agents.whittington_2020 import Whittington2020

class debug(Whittington2020):
    
    def __init__(self, **agent_params):
        """
        This is the constructor for the class.
        """
        # Safely remove our custom parameters before calling the parent
        self.point_a = agent_params.pop("point_a", (-4, -3))
        self.point_b = agent_params.pop("point_b", (4, 3))
        
        # Now call the parent __init__ with only the parameters it understands
        super().__init__(**agent_params)
        
        # Set up a list of targets, one for each agent in the batch
        self.current_targets = [self.point_b] * self.batch_size
    
    # UN-INDENTED: batch_act is now a method of the debug class
    def batch_act(self, observations):
        """
        This is our test method to see if we can force the agent to move.
        """
        # We will ignore the target and force every agent to try to move right.
        action_to_test = [0, 1] 
        new_actions = [action_to_test] * self.batch_size

        # Manage the state variables correctly
        self.walk_actions.append(self.prev_actions.copy())
        self.obs_history.append(self.prev_observations.copy())
        self.prev_actions = new_actions
        self.prev_observations = observations
        self.n_walk += 1

        # Write to our debug file to see if the position changes
        with open("debug_output2.txt", "a") as f:
            # Log the first agent to see what's happening
            f.write(f"Step: {self.n_walk} | Pos: {observations[0][2]} | Sent Action: {new_actions[0]}\n")

        return new_actions