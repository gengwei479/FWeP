import os
import numpy as np
from arg_parses import create_parses, pre_parses
from utils.viaual_fun import line_graphs_01, line_multi_graphs_01_c

args = create_parses()
args = pre_parses(args)
inputsYs = []
desc_info = []
step_num = 5

task_names = ['formation3']#'navigate', 'nav_change', 'nav_collision', 'formation3'
air_crafts = ['f16', '787-8']# 'f16', '787-8'
for air_craft in air_crafts:
    for task_name in task_names:
        fwep_qwen_max_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_qwen-max_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)
        fwep_qwen_plus_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_qwen-plus_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)
        fwep_qwen_turbo_reward = np.load(os.path.join(os.getcwd(), args.result_data_dir, 'otheralg', 'FWeP_qwen-turbo_eval_{}_type_{}.npy'.format(task_name, air_craft)), allow_pickle=True)

        # print(fwep_qwen_max_reward.shape)
        inputsYs.append({'follower 1': fwep_qwen_max_reward[1:, ::step_num, 1],
                        'follower 2': fwep_qwen_max_reward[1:, ::step_num, 2]})
        # inputsYs.append({'FWeP(qwen-plus)': fwep_qwen_max_reward[1:, ::step_num]})
        desc_info.append({'air_craft': air_craft})#'FWeP(qwen-max)'
line_multi_graphs_01_c(np.arange(0, 3000, step_num).data, inputsYs, ["Step", "Reward"], 
                os.path.join(os.getcwd(), args.result_data_dir, 'main_res_formation.pdf'),
                desc_info, fontsize = 60, fig_size=(50, 20))