from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change
        hidden_space3 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
    
class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-2  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        # Assuming self.probs and deltas are lists of log probabilities and deltas respectively
        loss = 0

        # Minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            if isinstance(log_prob, np.ndarray):
                log_prob = torch.tensor(log_prob, requires_grad=True)
            if isinstance(delta, np.ndarray):
                delta = torch.tensor(delta, requires_grad=False)
                
            loss += torch.mean(log_prob) * delta * (-1)

        # Ensure loss is a tensor
        if isinstance(loss, float):
            loss = torch.tensor(loss, requires_grad=True)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

# Create and wrap the environment
from physics_sim import System2D
env = System2D()

total_num_episodes = int(3.5e4)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (5)
obs_space_dims = len(env.get_observation_space())
# Action-space of InvertedPendulum-v4 (5)
action_space_dims = env.get_action_space().shape[0]
rewards_over_seeds = []

for seed in [42]:  # Fibonacci seed/s
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    first = True
    reached_1000 = False
    reached_5000 = False

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        env.reset_system_rand()
        obs, info = (env.get_observation_space(), [])

        # set time and max time
        t = 0
        t_max = 10 # (seconds)

        # set max theta
        theta_max = np.pi / 4

        # initialize total_reward
        total_reward = 0

        # get initial theta
        theta_prev = obs[0]
        phi_dot_prev = obs[3]

        done = False

        # initialize reward
        reward = 0
        saved = [[],[],[],[]]
        max_total_reward = 0

        while not done:
            obs = env.get_observation_space()
            action = agent.sample_action(obs)
            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            # step
            env.step(action)

            # update t
            t += env.dt

            # get obs
            obs = env.get_observation_space()
            theta = obs[0]
            theta_dot = obs[2]
            phi = obs[1]
            phi_dot = obs[3]

            # danger!!!!
            theta = theta + phi

            # reward = 1.0

            # calculate reward
            if np.abs(theta) < np.pi / 100:
                reward += 10
            elif np.abs(theta) < np.abs(theta_prev):
                reward += 1

            ## complex rewards to make the ball still while balanced
            # if np.abs(theta) < np.pi / 100 and np.abs(phi_dot) < np.abs(phi_dot_prev):
            #     reward += 3
            # if np.abs(theta) < np.pi / 100 and np.abs(phi_dot) < .01:
            #     reward += 10
            # if np.abs(phi_dot) < np.abs(phi_dot_prev):
            #     reward += 1

            # update terminated (max theta reached)
            terminated = np.abs(theta) > theta_max

            # update truncated (max time reached)
            truncated = t >= t_max

            agent.rewards.append(reward)

            # update total reward
            total_reward += reward

            # update theta_prev and phi_dot_prev
            theta_prev = theta
            phi_dot_prev = phi_dot

            # reset reward
            reward = 0
            saved[0].append(theta)
            saved[1].append(phi)
            saved[2].append(theta_dot)
            saved[3].append(phi_dot)
            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        # append total_reward to return_queue
        env.return_queue.append(total_reward)

        if total_reward > max_total_reward:
            max_total_reward = total_reward
            np.savetxt("best_model_system_states.csv", saved, delimiter=",")
            torch.save(agent.net.state_dict(), "best_model.pt")
            torch.save(agent.optimizer.state_dict(), "best_optimizer.pt")

        if first:
            first = False
            np.savetxt("first_model_system_states.csv", saved, delimiter=",")
            torch.save(agent.net.state_dict(), "first_model.pt")
            torch.save(agent.optimizer.state_dict(), "first_optimizer.pt")
        
        if not reached_1000 and total_reward > 1000:
            reached_1000 = True
            np.savetxt("thousand_model_system_states.csv", saved, delimiter=",")
            torch.save(agent.net.state_dict(), "thousand_model.pt")
            torch.save(agent.optimizer.state_dict(), "thousand_optimizer.pt")

        if not reached_5000 and total_reward > 5000:
            reached_5000 = True
            np.savetxt("five_thousand_model_system_states.csv", saved, delimiter=",")
            torch.save(agent.net.state_dict(), "five_thousand_model.pt")
            torch.save(agent.optimizer.state_dict(), "five_thousand_optimizer.pt")


        reward_over_episodes.append(env.return_queue[-1])
        agent.update()
        # for param in agent.net.parameters():
        #     print(param)

        if episode % 1000 == 0:
            avg_reward = int(np.mean(env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

rewards_to_plot = [
    [reward[0] if isinstance(reward, (list, tuple)) else reward for reward in rewards]
    for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "Episodes", "value": "Reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="Episodes", y="Reward", data=df1).set(
    title="Total Reward Obtained by Agent"
)
plt.show()