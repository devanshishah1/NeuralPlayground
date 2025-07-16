
import numpy as np
import random

from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters


class randompoints(Whittington2020):
    
    def __init__(self, **agent_params):
        """
        Custom agent that navigates back and forth between two points.
        We define the target points and call the parent agent's constructor.
        """
        # Pop custom parameters before the parent class sees them
        #self.point_a = np.array(agent_params.pop("point_a", (-4, -3)))
        #self.point_b = np.array(agent_params.pop("point_b", (4, 3)))
        #self.point_c = np.array(agent_params.pop("point_c", (-4, 3)))

        self.points = [np.array(p) for p in agent_params.pop("points",[])]
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






    
    """
    def git
        self.iter = int((len(self.obs_history) / self.pars["n_rollout"])) - 1
        history = self.obs_history[-self.pars["n_rollout"]:]
        locations = [[{"id": env_step[0], "shiny": None} for env_step in step] for step in history]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.pars["n_rollout"]:]
        
        # Reset n_walk to start a new rollout in the training loop
        self.n_walk = 0
        
        action_values = self.step_to_actions(actions)
        self.walk_action_values.append(action_values)

        (
            self.eta_new,
            self.lambda_new,
            self.p2g_scale_offset,
            self.lr,
            self.walk_length_center,
            loss_weights,
        ) = parameters.parameter_iteration(self.iter, self.pars)

        self.tem.hyper["eta"] = self.eta_new
        self.tem.hyper["lambda"] = self.lambda_new
        self.tem.hyper["p2g_scale_offset"] = self.p2g_scale_offset
        for param_group in self.adam.param_groups:
            param_group["lr"] = self.lr
        
        model_input = [
            [
                locations[i],
                torch.from_numpy(np.reshape(observations, (self.pars["n_rollout"], self.pars["batch_size"], self.pars["n_x"]))[i]).type(torch.float32),
                np.reshape(action_values, (self.pars["n_rollout"], self.pars["batch_size"]))[i].tolist(),
            ]
            for i in range(self.pars["n_rollout"])
        ]
        
        self.final_model_input = model_input
        forward = self.tem(model_input, self.prev_iter)
        loss = torch.tensor(0.0)

        for _, step in enumerate(forward):
            step_loss_list = []
            for env_i, env_visited in enumerate(self.visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss_list.append(loss_weights * torch.stack([i[env_i] for i in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
            
            if step_loss_list:
                step_loss = torch.mean(torch.stack(step_loss_list, dim=0), dim=0)
                loss = loss + torch.sum(step_loss)

        self.adam.zero_grad()
        if loss.requires_grad:
            loss.backward(retain_graph=True)
            self.adam.step()
        self.prev_iter = [forward[-1].detach()]




#import TEM agent class
from neuralplayground.agents.whittington_2020 import Whittington2020

class backnforth(Whittington2020):
    def __init__ (self, model_name: str = "backnforth", point_a=(0, 0), point_b=(9, 9), **mod_kwargs): #adding a point a and b so if user doesnt specify, it still works
        super().__init__(**mod_kwargs) #allows to add to the originial init (specific attributes?)
        self.point_a = point_a
        self.point_b = point_b
        self.current_target = point_b
        #agent starts at point a and aims for point b (starting settings)
        
    

    #custom policy_func
    def action_policy(self):

        #return super().action_policy() we want to overwrite original action_policy 
    
        #if self.current_target == self.point_b:
            #self.current_target = self.point_a 
       #else:
            #self.current_target = self.point_b
        
        current_pos = self.prev_observations[0][2]
        if np.array_equal(current_pos, self.current_target):
        #if current_pos == list(self.current_target):
            self.current_target = self.point_a if self.current_target == self.point_b else self.point_b
        
        #calculating distance to target: 
        delta_x = self.current_target[0] - current_pos[0]
        delta_y = self.current_target[1] - current_pos[1]

        if abs(delta_x) > abs(delta_y):
            return [int(np.sign(delta_x)), 0]
        else:
            return [0, int(np.sign(delta_y))]
            """