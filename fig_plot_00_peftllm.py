from arg_parses import create_parses, pre_parses
from utils.viaual_fun import line_graphs_01, line_graphs_02, line_graphs_03, line_multi_graphs_02
# from trans_trainer.trans_gpt2 import TRANSGPT

args = create_parses()
args = pre_parses(args)
# translate = TRANSGPT(args)
# translate.generate_acc_test()

import os
import numpy as np
import matplotlib



aircraft_types = ['f16', '787-8']
file_dir = os.path.join(os.getcwd(), 'wandb')
files = os.listdir(file_dir)
desc_info = []
inputsYs = []
item_id = 0

for aircraft_type in aircraft_types:
    loss_res_dict = {}
    for fine_tune_mode in ['p-tuning_mlp', 'p-tuning_lstm', 'prefix-tuning', 'prompt-tuning', 'lora', 'adalora', 'ia3']:
        # print(files[item_id])
        file = open(os.path.join(file_dir, files[item_id], 'files', 'output.log')).readlines()
        tmp_loss_list = []
        for line in file:
            if 'loss' in line and 'grad_norm' in line:
                # print(eval(line)['loss'])
                tmp_loss_list.append(eval(line)['loss'])
        # print(tmp_loss_list)
        loss_res_dict[fine_tune_mode] = tmp_loss_list
        
        item_id += 1
    loss_res_dict['fine-tuning'] = np.load(os.path.join(os.getcwd(), args.finetuned_checkpoint, 'kine_llm_eval', 'gpt2', 'loss_function', '{}loss_list_v0.npy'.format(aircraft_type)), allow_pickle=True)[::10]
    # print(len(loss_res_dict['p-tuning_mlp']))
    inputsYs.append(loss_res_dict)
    desc_info.append({'air_craft':aircraft_type})
line_multi_graphs_02(np.arange(0, 1500, 10), inputsYs, ["Iteration", "Loss"], 
            os.path.join(os.getcwd(), args.result_data_dir, 'peftllm_loss.pdf'), y_label_list=list(loss_res_dict.keys()), 
            tiny_win = False, desc_info = desc_info, sub_just = [0.35, 0.085, 0.99, 0.93])

# for air_craft in air_crafts:
#     loss_kllm_dict = {}
#     for k_llm in k_llms:
#         loss = np.load(os.path.join(os.getcwd(), args.finetuned_checkpoint, 'kine_llm_eval', k_llm, 'loss_function', 
#                         '{}loss_list_v0.npy'.format(air_craft)), allow_pickle=True)
#         # print(loss)
#         loss_kllm_dict[k_llm] = loss
#     line_graphs_02(np.arange(0, 1500, 1), loss_kllm_dict, ["Iteration", "Loss"], 
#                     os.path.join(os.getcwd(), args.result_data_dir, 'kllm_{}_loss.pdf'.format(air_craft)), y_label_list=['gpt2', 't5_60m', 't5_220m', 'transformer'])