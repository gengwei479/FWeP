from arg_parses import create_parses, pre_parses
from utils.viaual_fun import line_graphs_01, line_multi_graphs_01
# from trans_trainer.trans_gpt2 import TRANSGPT

args = create_parses()
args = pre_parses(args)
# translate = TRANSGPT(args)
# translate.generate_acc_test()

import os
import numpy as np
import matplotlib

#--------------------------------------------------------------------------------------------------------------------------------
task_names = ['navigate', 'nav_change', 'nav_collision', 'formation3']#'navigate', 'nav_change', 'nav_collision', 'formation3'
air_crafts = ['f16', '787-8']# 'f16', '787-8'
# algorithm = ['SAC', 'TD3', 'PPO', 'DDPG']
step_num = 10
inputsYs = []
desc_info = []

for air_craft in air_crafts:
    for task_name in task_names:
        fwep_qwen_max_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_qwen-max_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)
        fwep_qwen_plus_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_qwen-plus_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)
        fwep_qwen_turbo_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_qwen-turbo_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)
        # fwep_dolly_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_dolly_12b_v2_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)
        ppo_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'PPO_eval_{}_type_{}.npy'.format(task_name, air_craft)))
        ddpg_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'DDPG_eval_{}_type_{}.npy'.format(task_name, air_craft)))
        td3_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'TD3_eval_{}_type_{}.npy'.format(task_name, air_craft)))
        sac_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'SAC_eval_{}_type_{}.npy'.format(task_name, air_craft)))
        # # print(td3_reward)
        # print(ddpg_reward)

        if task_name == "formation3":
            fwep_qwen_max_reward = np.sum(fwep_qwen_max_reward, axis=2)
            fwep_qwen_plus_reward = np.sum(fwep_qwen_plus_reward, axis=2)
            fwep_qwen_turbo_reward = np.sum(fwep_qwen_turbo_reward, axis=2)
            ppo_reward = np.sum(ppo_reward, axis=2)
            ddpg_reward = np.sum(ddpg_reward, axis=2)
            td3_reward = np.sum(td3_reward, axis=2)
            sac_reward = np.sum(sac_reward, axis=2)

        # print(fwep_qwen_max_reward.shape)
        # print(fwep_qwen_plus_reward.shape)
        # print(fwep_qwen_turbo_reward.shape)

        desc_info.append({'task_name': task_name, 'air_craft': air_craft})
        inputsYs.append({"FWeP(qwen-max)": fwep_qwen_max_reward[1:, :3000:step_num], 
                                                    "FWeP(qwen-plus)": fwep_qwen_plus_reward[1:, :3000:step_num], #0:2:2
                                                    "FWeP(qwen-turbo)": fwep_qwen_turbo_reward[1:, :3000:step_num], 
                                                    # "FWeP(dolly)": fwep_dolly_reward[:][ :3000], 
                                                    "PPO + PID": ppo_reward[1:, :3000:step_num], 
                                                    "DDPG + PID": ddpg_reward[1:, :3000:step_num],
                                                    "TD3 + PID": td3_reward[1:, :3000:step_num], 
                                                    "SAC + PID": sac_reward[1:, :3000:step_num]})

line_multi_graphs_01(np.arange(0, 3000, step_num).data, inputsYs, ["Step", "Reward"], 
                os.path.join(os.getcwd(), args.result_data_dir, 'main_res_f16_787.pdf'),
                desc_info)


##--------------------------------------------------------------------------------------------------------------------------------
# air_crafts = ['f16', '787-8']# 'f16', '787-8'
# loss0 = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'loss_function', '{}loss_list_v50000.npy'.format(air_crafts[0])), allow_pickle=True)
# loss1 = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'loss_function', '{}loss_list_v50000.npy'.format(air_crafts[1])), allow_pickle=True)
# print(loss0)
# line_graphs_01(np.arange(0, 1500, 1).data, {"F16": [loss0], 
#                                             "Boeing 787-8": [loss1]}, 
#                                             ["Iteration", "Loss"], 
#                 os.path.join(os.getcwd(), args.result_data_dir, 'main_loss.pdf'))