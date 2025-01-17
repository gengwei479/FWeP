import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging
from log.log_process import Logger
from envs.env_wrapper import NavigateEnv, AvoidCollisionEnv
from utils.loadcsv import trans_from_zjenv_to_mrad

# The network of the actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)#tanh softmax(self.l2(s), dim=1)
        return a_prob


# The network of the critic
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s))
        v_s = self.l2(s)
        return v_s


class A2C(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 64  # The number of neurons in hidden layers of the neural network
        self.lr = 5e-4  # learning rate
        self.GAMMA = 0.99  # discount factor
        self.I = 1
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, self.hidden_width)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, self.hidden_width)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = (2 * self.max_action * self.actor(s) - self.max_action).detach().numpy().flatten()  # probability distribution(numpy)
        return a
        # if deterministic:  # We use the deterministic policy during the evaluating
        #     a = np.argmax(prob_weights)  # Select the action with the highest probability
        #     return a
        # else:  # We use the stochastic policy during the training
        #     a = np.random.choice(range(self.action_dim), p=prob_weights)  # Sample the action according to the probability distribution
        #     return a

    def learn(self, s, a, r, s_, dw):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float), 0)
        v_s = self.critic(s).flatten()  # v(s)
        v_s_ = self.critic(s_).flatten()  # v(s')

        with torch.no_grad():  # td_target has no gradient
            td_target = r + self.GAMMA * (1 - dw) * v_s_

        # Update actor
        log_pi = torch.log(self.actor(s).flatten()[a])  # log pi(a|s)
        actor_loss = -(self.I * ((td_target - v_s).detach()) * log_pi).mean()  # Only calculate the derivative of log_pi
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        critic_loss = (td_target - v_s) ** 2  # Only calculate the derivative of v(s)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.I *= self.GAMMA  # Represent the gamma^t in th policy gradient theorem


def evaluate_policy(args, env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset(destination=[0, 0, 1000])
        s = trans_from_zjenv_to_mrad(list(s['player1'].values())[0 : args.obs_dim])
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            s_ = trans_from_zjenv_to_mrad(list(s_['player1'].values())[0 : args.obs_dim])
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


# if __name__ == '__main__':
def a2c_main(args):
    # env = gym.make(env_name[env_index])
    # env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment

    env = NavigateEnv(args, allow_flightgear_output = True)
    env_evaluate = NavigateEnv(args, allow_flightgear_output = True)
    logger = Logger("ppo_log_info.log", str(os.getcwd()) + str(args.log_dir) + '/')

    # Set random seed
    seed = args.seed
    # env.seed(seed)
    env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = args.obs_dim
    action_dim = args.action_dim
    max_action = float(1.0)
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = A2C(state_dim, action_dim, max_action)

    max_train_steps = args.A2C_max_train_steps  # Maximum number of training steps
    evaluate_freq = args.A2C_evaluate_freq  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_rewards = []  # Record the rewards during the evaluating
    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        episode_steps = 0
        s = env.reset(destination=[0, 0, 1000])
        s = trans_from_zjenv_to_mrad(list(s['player1'].values())[0 : args.obs_dim])
        done = False
        agent.I = 1
        while not done:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)
            s_ = trans_from_zjenv_to_mrad(list(s_['player1'].values())[0 : args.obs_dim])

            # When dead or win or reaching the max_epsiode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            agent.learn(s, a, r, s_, dw)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                logger.log_insert("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward), logging.INFO)
            total_steps += 1
    np.save('./data_train/A2C_env_{}_seed_{}.npy'.format(args.env_name, seed), np.array(evaluate_rewards))