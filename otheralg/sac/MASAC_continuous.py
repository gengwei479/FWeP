import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal

import logging
from log.log_process import Logger
from envs.env_wrapper import NavigateEnv, AvoidCollisionEnv
from utils.loadcsv import trans_from_zjenv_to_mrad
from otheralg.ppo.normalization import Normalization, RewardScaling

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.numpy().flatten()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(args, env, agents, state_norm):
    times = args.evaluate_times  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    total_evaluate_reward = []
    for _ in range(times):
        s = env.reset()

        for agent_id, agent_name in enumerate(args.agent_keys):
            s[agent_name] = trans_from_zjenv_to_mrad(list(s[agent_name].values())[0 : args.obs_dim])
            if args.PPO_use_state_norm:
                s[agent_name] = state_norm(s[agent_name], update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        total_episode_reward = []
        while not done:
            a = {}
            for agent_id, agent_name in enumerate(args.agent_keys):
                a_v = agents[agent_name].choose_action(s[agent_name], deterministic=True)  # We use the deterministic policy during the evaluating
                a[agent_name] = env.rule_act[agent_name].goto_point(dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 0.25 * a_v[2] + 0.75])), a_v[0], a_v[1])
            s_, r, done, _ = env.step(a)

            for agent_id, agent_name in enumerate(args.agent_keys):
                s_[agent_name] = trans_from_zjenv_to_mrad(list(s_[agent_name].values())[0 : args.obs_dim])
                if args.PPO_use_state_norm:
                    s_[agent_name] = state_norm(s_[agent_name], update=False)
            episode_reward += r
            total_episode_reward.append(r)
            s = s_
        evaluate_reward += episode_reward
        total_evaluate_reward.append(total_episode_reward)

    return evaluate_reward / times, total_evaluate_reward


def masac_main(args, env, env_evaluate):
    
    env = env
    env_evaluate = env_evaluate
    logger = Logger("sac_" + args.aircraft_type + "_" + env.env_name + "_log_info.log", os.path.join(os.getcwd(), args.log_dir))
    
    seed = args.seed
    # Set random seed
    # env.seed(seed)
    env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = args.obs_dim
    action_dim = args.static_action_dim
    max_action = torch.tensor([180, 1000, 1], dtype=torch.float32)
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.env_name))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    # agent = SAC(state_dim, action_dim, max_action)
    # replay_buffer = ReplayBuffer(state_dim, action_dim)
    agents = {}
    replay_buffers = {}
    for agent_id, agent_name in enumerate(args.agent_keys):
        agents[agent_name] = SAC(state_dim, action_dim, max_action)

        replay_buffers[agent_name] = ReplayBuffer(state_dim, action_dim)

    max_train_steps = args.max_train_steps  # Maximum number of training steps
    random_steps = args.SAC_random_steps  # Take the random actions in the beginning for the better exploration
    evaluate_freq = args.SAC_evaluate_freq  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    state_norm = Normalization(shape=args.obs_dim)  # Trick 2:state normalization
    if args.PPO_use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.PPO_use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.PPO_gamma)

    if os.path.exists(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_agent_{}_type_{}.pkl'.format(env.env_name, args.aircraft_type))):
        with open(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_agent_{}_type_{}.pkl'.format(env.env_name, args.aircraft_type)), 'rb') as f:
            agents = pickle.load(f)
        evaluate_reward, total_evaluate_reward = evaluate_policy(args, env_evaluate, agents, state_norm)
        np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_eval_{}_type_{}.npy'.format(env_evaluate.env_name, args.aircraft_type)), np.array(total_evaluate_reward))
        return

    while total_steps < max_train_steps:
        s = env.reset()

        for agent_id, agent_name in enumerate(args.agent_keys):
            s[agent_name] = trans_from_zjenv_to_mrad(list(s[agent_name].values())[0 : args.obs_dim])

            if args.PPO_use_state_norm:
                s[agent_name] = state_norm(s[agent_name])

        if args.PPO_use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1

            a = {}
            a_v_dict = {}
            for agent_id, agent_name in enumerate(args.agent_keys):
                if total_steps < random_steps: 
                    a_v = [random.randint(-max_action[0], max_action[0]), random.randint(-max_action[1], max_action[1]), random.random() * 0.5 + 0.5]
                else:
                    a_v = agents[agent_name].choose_action(s[agent_name])

                a_v_dict[agent_name] = a_v
                a[agent_name] = env.rule_act[agent_name].goto_point(dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 0.25 * a_v[2] + 0.75])), a_v[0], a_v[1])
            s_, r, done, _ = env.step(a)

            for agent_id, agent_name in enumerate(args.agent_keys):
                s_[agent_name] = trans_from_zjenv_to_mrad(list(s_[agent_name].values())[0 : args.obs_dim])

                if args.PPO_use_state_norm:
                    s_[agent_name] = state_norm(s_[agent_name])

            if args.PPO_use_reward_norm:
                r = reward_norm(r)
            elif args.PPO_use_reward_scaling:
                r = reward_scaling(r)

            # r = reward_adapter(r, env_index)  # Adjust rewards for better performance
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            for agent_id, agent_name in enumerate(args.agent_keys):
                replay_buffers[agent_name].store(s[agent_name], a_v_dict[agent_name], r, s_[agent_name], dw)  # Store the transition
            s = s_

            if total_steps >= random_steps:
                for agent_id, agent_name in enumerate(args.agent_keys):
                    agents[agent_name].learn(replay_buffers[agent_name])

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, total_evaluate_reward = evaluate_policy(args, env_evaluate, agents, state_norm)
                evaluate_rewards.append(total_evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t avg_step_num:{}".format(evaluate_num, evaluate_reward, sum([len(item) for item in total_evaluate_reward]) / 3))
                logger.log_insert("evaluate_num:{} \t evaluate_reward:{} \t avg_step_num:{}".format(evaluate_num, evaluate_reward, sum([len(item) for item in total_evaluate_reward]) / 3), logging.INFO)
            total_steps += 1

            if total_steps >= args.max_train_steps:
                break

    if not os.path.exists(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg')):
        os.mkdir(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg'))
    np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_eval_{}_type_{}.npy'.format(env_evaluate.env_name, args.aircraft_type)), np.array(total_evaluate_reward))
    np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_training_{}_type_{}.npy'.format(env_evaluate.env_name, args.aircraft_type)), np.array(evaluate_rewards))
    with open(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_agent_{}_type_{}.pkl'.format(env.env_name, args.aircraft_type)), 'wb') as f:
        pickle.dump(agents, f)