
import numpy as np
import random

from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters


class AnyRandom3Points(Whittington2020):
    
    def __init__(self, **agent_params):
        """
        Custom agent that navigates back and forth between two points.
        We define the target points and call the parent agent's constructor.
        """
        # Pop custom parameters before the parent class sees them
        #self.point_a = np.array(agent_params.pop("point_a", (-4, -3)))
        #self.point_b = np.array(agent_params.pop("point_b", (4, 3)))
        #self.point_c = np.array(agent_params.pop("point_c", (-4, 3)))

        #need a list of list of points 
        self.choice_points = [np.array(p) for p in agent_params.pop("points",[])]
        self.points = random.choice(self.choice_points)

        if len(self.points) < 2:
            print("please provide atleast 2 points")
    
        #allows user to set points in simulation script when creating agent params dict
        
        super().__init__(**agent_params) # calls the __init__ method of the parent class (Whittington2020).
        
        # random starting target set:
        self.current_targets = [random.choice(self.points) for _ in range(self.batch_size)]

    def batch_act(self, observations):
        """
        Generates a batch of actions to navigate between point_a and point_b,
        accounting for the environment's specific action format.
        """
        new_actions = []
        for i in range(self.batch_size):
            current_pos = np.array(observations[i][2])
            target_pos = self.current_targets[i]

            if np.linalg.norm(current_pos - target_pos) < 1.0:
                #i.e. if distance to target pos close enough to swap targets
    
                possible_next_targets = [x for x in self.points if not np.array_equal(x, target_pos)]
                
                
                # Randomly choose one target from our new list of possibilities.
                new_target = random.choice(possible_next_targets)
                
                # Update this agent's target to the newly chosen point.
                self.current_targets[i] = new_target
                
                # We also update target_pos here so the action for this step is towards the new target.
                target_pos = new_target
        

        
            # Determine the action towards the CURRENT target.
            # (The target will be updated on the next call to batch_act)
            delta = target_pos - current_pos
            
            # Choose the best discrete action (up, down, left, or right).
            if abs(delta[0]) > abs(delta[1]):
                action = [np.sign(delta[0]), 0]  # Move horizontally
            elif abs(delta[1]) > 0:
                action = [0, np.sign(delta[1])]  # Move vertically
            else:
                # This case should rarely happen if the arrival radius is > 0
                action = [0, 0] # Stay still if exactly at target
            
             # "Translate" the desired move into the format the environment understands
             # i.e. (0,-1) means move up etc 

            action_to_send = list(action) # Make a copy
            if action_to_send[0] == 0:
                action_to_send[1] = -action_to_send[1] # Counteract the environment's flip

            new_actions.append(action_to_send)

        # 4. Manage history for the TEM model (this is from the base class and is crucial)
        self.walk_actions.append(self.prev_actions.copy())
        self.obs_history.append(self.prev_observations.copy())
        self.prev_actions = new_actions
        self.prev_observations = observations
        self.n_walk += 1

        return new_actions

