import torch
import numpy as np
import os
import pickle
from otheralg.ppo.normalization import Normalization, RewardScaling
from otheralg.ppo.replaybuffer import ReplayBuffer
from otheralg.ppo.ppo_continuous import PPO_continuous

import logging
from log.log_process import Logger
from envs.env_wrapper import NavigateEnv, AvoidCollisionEnv
from utils.loadcsv import trans_from_zjenv_to_mrad


def evaluate_policy(args, env, agent, state_norm):
    times = args.evaluate_times
    evaluate_reward = 0
    total_evaluate_reward = []
    for _ in range(times):
        s = env.reset()
        s = trans_from_zjenv_to_mrad(list(s['player1'].values())[0 : args.PPO_state_dim])
        if args.PPO_use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        total_episode_reward = []
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.PPO_policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.PPO_max_action  # [0,1]->[-max,max]
            else:
                action = a
                
            action = env.rule_act.goto_point(dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 0.25 * action[2] + 0.75])), action[0], action[1])
            s_, r, done, _ = env.step({'player1': action})
            s_ = trans_from_zjenv_to_mrad(list(s_['player1'].values())[0 : args.PPO_state_dim])
            if args.PPO_use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            total_episode_reward.append(r)
            s = s_
        evaluate_reward += episode_reward
        total_evaluate_reward.append(total_episode_reward)

    return evaluate_reward / times, total_evaluate_reward


def ppo_main(args, env, env_evaluate):
    
    env = env
    env_evaluate = env_evaluate
    logger = Logger("ppo_" + args.aircraft_type + "_" + env.env_name + "_log_info.log", os.path.join(os.getcwd(), args.log_dir))
    
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.PPO_state_dim = args.obs_dim
    args.PPO_action_dim = args.static_action_dim
    args.PPO_max_action = torch.tensor([180, 1000, 1], dtype=torch.float32)
    args.PPO_max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(args.PPO_state_dim))
    print("action_dim={}".format(args.PPO_action_dim))
    print("max_action={}".format(args.PPO_max_action))
    print("max_episode_steps={}".format(args.PPO_max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    state_norm = Normalization(shape=args.PPO_state_dim)  # Trick 2:state normalization
    if args.PPO_use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.PPO_use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.PPO_gamma)

    if os.path.exists(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_agent_{}_type_{}.pkl'.format(env.env_name, args.aircraft_type))):
        with open(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_agent_{}_type_{}.pkl'.format(env.env_name, args.aircraft_type)), 'rb') as f:
            agent = pickle.load(f)
        evaluate_reward, total_evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
        np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_eval_{}_type_{}.npy'.format(env_evaluate.env_name, args.aircraft_type)), np.array(total_evaluate_reward))
        return

    while total_steps < args.max_train_steps:
        s = env.reset()
        s = trans_from_zjenv_to_mrad(list(s['player1'].values())[0 : args.PPO_state_dim])
        if args.PPO_use_state_norm:
            s = state_norm(s)
        if args.PPO_use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.PPO_policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.PPO_max_action  # [0,1]->[-max,max]
            else:
                action = a
            action = env.rule_act.goto_point(dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 0.25 * action[2] + 0.75])), action[0], action[1])
            s_, r, done, _ = env.step({'player1': action})
            s_ = trans_from_zjenv_to_mrad(list(s_['player1'].values())[0 : args.PPO_state_dim])

            if args.PPO_use_state_norm:
                s_ = state_norm(s_)
            if args.PPO_use_reward_norm:
                r = reward_norm(r)
            elif args.PPO_use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.PPO_max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.PPO_batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.PPO_evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, total_evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(total_evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t avg_step_num:{}".format(evaluate_num, evaluate_reward, sum([len(item) for item in total_evaluate_reward]) / 3))
                logger.log_insert("evaluate_num:{} \t evaluate_reward:{} \t avg_step_num:{}".format(evaluate_num, evaluate_reward, sum([len(item) for item in total_evaluate_reward]) / 3), logging.INFO)
                # Save the rewards
            
            if total_steps >= args.max_train_steps:
                break

    if not os.path.exists(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg')):
        os.mkdir(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg'))
    np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_eval_{}_type_{}.npy'.format(env_evaluate.env_name, args.aircraft_type)), np.array(total_evaluate_reward))
    np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_training_{}_type_{}.npy'.format(env_evaluate.env_name, args.aircraft_type)), np.array(evaluate_rewards))
    with open(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_agent_{}_type_{}.pkl'.format(env.env_name, args.aircraft_type)), 'wb') as f:
        pickle.dump(agent, f)