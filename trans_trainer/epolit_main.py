from utils.loadcsv import trans_from_zjenv_to_mrad
from trans_trainer.e_pilot import PILOT
from trans_trainer.multi_epilot import MULPILOT
import os
import numpy as np
import logging
from log.log_process import Logger
from log.json_process import Jsoner

def evaluate_policy(args, env, agent, state_norm):
    times = 3
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

def epolit_main(args, env_evaluate):
    prompt_logger = Jsoner("epilot_" + args.sty_llm + "_" + args.aircraft_type + "_" + args.env_name + "_prompt_log_info.txt", os.path.join(os.getcwd(), args.log_dir))

    # if os.path.exists(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type))):
    #     evaluate_rewards = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type))).tolist()
    # else:
    #     evaluate_rewards = []

    times = 1#args.evaluate_times
    
    for _ in range(times):
        base_logger = Logger("epilot_" + args.sty_llm + "_" + args.aircraft_type + "_" + args.env_name + "_log_info.log", os.path.join(os.getcwd(), args.log_dir))
        if args.num_agents == 1:
            pilot = PILOT(args, env_evaluate, logger = base_logger, prompt_logger = prompt_logger, save_csv = True)
        else:
            pilot = MULPILOT(args, env_evaluate, logger = base_logger, prompt_logger = prompt_logger, save_csv = True)
        total_episode_reward = pilot.pilot_traj()

        if os.path.exists(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type))):
            evaluate_rewards = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type))).tolist()
        else:
            evaluate_rewards = []

        evaluate_rewards.append(total_episode_reward)
        np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type)), np.array(evaluate_rewards))


    # times = 1
    # evaluate_rewards = []
    # nest_res = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type)))
    # for _ in range(times):
    #     base_logger = Logger("epilot_" + args.aircraft_type + "_" + args.env_name + "_log_info.log", os.path.join(os.getcwd(), args.log_dir))
    #     if args.num_agents == 1:
    #         pilot = PILOT(args, env_evaluate, logger = base_logger, prompt_logger = prompt_logger, save_csv = True)
    #     else:
    #         pilot = MULPILOT(args, env_evaluate, logger = base_logger, prompt_logger = prompt_logger, save_csv = True)
    #     total_episode_reward = pilot.pilot_traj()
    #     evaluate_rewards.append(total_episode_reward)
    # np.save(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_{}_eval_{}_type_{}.npy'.format(args.sty_llm, args.env_name, args.aircraft_type)), np.concatenate((nest_res)))
    