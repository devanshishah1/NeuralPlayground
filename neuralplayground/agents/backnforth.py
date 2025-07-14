import os
import pickle
import numpy as np
import torch
import numpy as np
import time

from neuralplayground.agents.whittington_2020 import Whittington2020
import neuralplayground.agents.whittington_2020_extras.whittington_2020_parameters as parameters

class backnforth(Whittington2020):
    
    def __init__(self, **agent_params):
        # Safely remove our custom parameters before calling the parent
        self.point_a = agent_params.pop("point_a", (-4, -3))
        self.point_b = agent_params.pop("point_b", (4, 3))
        
        # Now call the parent __init__ with only the parameters it understands
        super().__init__(**agent_params)
        
        # Set up a list of targets, one for each agent in the batch
        self.current_targets = [self.point_b] * self.batch_size

    import numpy as np
from neuralplayground.agents.whittington_2020 import Whittington2020

class backnforth(Whittington2020):
    
    def __init__(self, **agent_params):
        """
        This is the constructor for the class.
        """
        self.point_a = agent_params.pop("point_a", (-4, -3))
        self.point_b = agent_params.pop("point_b", (4, 3))
        
        super().__init__(**agent_params)
        
        self.current_targets = [self.point_b] * self.batch_size

    def batch_act(self, observations):
        """
        Final corrected batch_act method with the action translator.
        """
        new_actions = []
        for i in range(self.batch_size):
            # Part 1: Standard logic to decide on the desired move
            current_pos = observations[i][2]
            target = self.current_targets[i]

            if np.linalg.norm(np.array(current_pos) - np.array(target)) < 0.5:
                new_target = self.point_a if np.array_equal(target, self.point_b) else self.point_b
                self.current_targets[i] = new_target
                target = new_target

            delta_x = target[0] - current_pos[0]
            delta_y = target[1] - current_pos[1]
            
            # Decide on our desired move in the standard [dy, dx] format
            desired_move = [0, 0]
            if abs(delta_x) > abs(delta_y):
                desired_move = [0, int(np.sign(delta_x))]  # Move horizontally
            else:
                desired_move = [int(np.sign(delta_y)), 0]  # Move vertically
            
            # Part 2: The "Translator"
            # We transform our desired move into the strange format the environment expects.
            # Based on our test, the rule is: expected_action = [desired_dx, -desired_dy]
            dy_desired, dx_desired = desired_move[0], desired_move[1]
            action_to_send = [dx_desired, -dy_desired]

            new_actions.append(action_to_send)

        # Part 3: Correctly manage state and history
        self.walk_actions.append(self.prev_actions.copy())
        self.obs_history.append(self.prev_observations.copy())
        self.prev_actions = new_actions
        self.prev_observations = observations
        self.n_walk += 1

        return new_actions
    
    def update(self):
        """
        This is the corrected version of the original update method. 
        It can handle any walk length and includes logging and saving.
        """
        self.iter = int((len(self.obs_history) / 20) - 1)
        history = self.obs_history[-self.pars["n_rollout"] :]
        locations = [[{"id": env_step[0], "shiny": None} for env_step in step] for step in history]
        observations = [[env_step[1] for env_step in step] for step in history]
        actions = self.walk_actions[-self.pars["n_rollout"] :]
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

        # This is the fixed part: using parameters instead of hardcoded numbers
        model_input = [
            [
                locations[i],
                torch.from_numpy(
                    np.reshape(
                        observations,
                        (
                            self.pars["n_rollout"],
                            self.pars["batch_size"],
                            self.pars["n_x"],
                        ),
                    )[i]
                ).type(torch.float32),
                np.reshape(action_values, (self.pars["n_rollout"], self.pars["batch_size"]))[i].tolist(),
            ]
            for i in range(self.pars["n_rollout"])
        ]
        
        self.final_model_input = model_input
        forward = self.tem(model_input, self.prev_iter)
        loss = torch.tensor(0.0)
        plot_loss = 0

        for ind, step in enumerate(forward):
            step_loss = []
            for env_i, env_visited in enumerate(self.visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(loss_weights * torch.stack([i[env_i] for i in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss = torch.tensor(0) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            plot_loss = plot_loss + step_loss.detach().numpy()
            loss = loss + torch.sum(step_loss)

        self.adam.zero_grad()
        loss.backward(retain_graph=True)
        self.adam.step()
        self.prev_iter = [forward[-1].detach()]

        
        
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]
        if self.iter % 10 == 0:
            self.logger.info(f"Loss: {loss.detach().numpy():.2f}. <p_g> {plot_loss[0]:.2f} ...")
        if self.iter % self.pars["save_interval"] == 0:
            torch.save(self.tem.state_dict(), self.model_path + "/tem_" + str(self.iter) + ".pt")
        if self.iter == self.pars["train_it"] - 1:
            torch.save(self.tem.state_dict(), self.model_path + "/tem_" + str(self.iter) + ".pt")



"""
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