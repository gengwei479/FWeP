import torch
import numpy as np
import os
from replay_buffer import *
from rainbow_dqn import DQN

import logging
from log.log_process import Logger
from envs.env_wrapper import NavigateEnv, AvoidCollisionEnv
from utils.loadcsv import trans_from_zjenv_to_mrad

class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed

        # self.env = gym.make(env_name)
        # self.env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
        
        self.env = NavigateEnv(args, allow_flightgear_output = True)
        self.env_evaluate = NavigateEnv(args, allow_flightgear_output = True)
        self.logger = Logger("ppo_log_info.log", str(os.getcwd()) + str(args.log_dir) + '/')
        
        # self.env.seed(seed)
        # self.env.action_space.seed(seed)
        # self.env_evaluate.seed(seed)
        # self.env_evaluate.action_space.seed(seed)
        
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.args.DQN_state_dim = self.args.obs_dims
        self.args.DQN_action_dim = self.env.action_space.n
        self.args.DQN_episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.DQN_state_dim))
        print("action_dim={}".format(self.args.DQN_action_dim))
        print("episode_limit={}".format(self.args.DQN_episode_limit))

        if args.DQN_use_per and args.DQN_use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.DQN_use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.DQN_use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.DQN_use_double and args.DQN_use_dueling and args.DQN_use_noisy and args.DQN_use_per and args.DQN_use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.DQN_use_double:
                self.algorithm += '_Double'
            if args.DQN_use_dueling:
                self.algorithm += '_Dueling'
            if args.DQN_use_noisy:
                self.algorithm += '_Noisy'
            if args.DQN_use_per:
                self.algorithm += '_PER'
            if args.DQN_use_n_steps:
                self.algorithm += "_N_steps"

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_num = 0
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.DQN_use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.DQN_epsilon_init
            self.epsilon_min = self.args.DQN_epsilon_min
            self.epsilon_decay = (self.args.DQN_epsilon_init - self.args.DQN_epsilon_min) / self.args.DQN_epsilon_decay_steps

    def run(self, ):
        self.evaluate_policy()
        while self.total_steps < self.args.DQN_max_train_steps:
            state = self.env.reset()
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                episode_steps += 1
                self.total_steps += 1

                if not self.args.DQN_use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.args.DQN_episode_limit:
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, reward, next_state, terminal, done)  # Store the transition
                state = next_state

                if self.replay_buffer.current_size >= self.args.DQN_batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

                if self.total_steps % self.args.DQN_evaluate_freq == 0:
                    self.evaluate_policy()
        # Save reward
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.algorithm, self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))

    def evaluate_policy(self, ):
        evaluate_reward = 0
        self.evaluate_num += 1
        self.agent.net.eval()
        for _ in range(self.args.DQN_evaluate_times):
            state = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, done, _ = self.env_evaluate.step(action)
                episode_reward += reward
                state = next_state
            evaluate_reward += episode_reward
        self.agent.net.train()
        evaluate_reward /= self.args.DQN_evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.logger.log_insert("evaluate_num:{} \t evaluate_reward:{} \t".format(self.evaluate_num, evaluate_reward), logging.INFO)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
#     parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
#     parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
#     parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

#     parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
#     parser.add_argument("--batch_size", type=int, default=256, help="batch size")
#     parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
#     parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
#     parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
#     parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
#     parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
#     parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
#     parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
#     parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
#     parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
#     parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
#     parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
#     parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
#     parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
#     parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

#     parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
#     parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
#     parser.add_argument("--use_noisy", type=bool, default=True, help="Whether to use noisy network")
#     parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
#     parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

#     args = parser.parse_args()

#     env_names = ['CartPole-v1', 'LunarLander-v2']
#     env_index = 1
#     for seed in [0, 10, 100]:
#         runner = Runner(args=args, env_name=env_names[env_index], number=1, seed=seed)
#         runner.run()
