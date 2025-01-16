from arg_parses import create_parses, pre_parses
from utils.viaual_fun import line_graphs_01, line_graphs_02, line_multi_graphs_02
# from trans_trainer.trans_gpt2 import TRANSGPT

args = create_parses()
args = pre_parses(args)
# translate = TRANSGPT(args)
# translate.generate_acc_test()

import os
import numpy as np
import matplotlib


# k_llms = ['gpt2', 'qwen2.5_1.5b', 't5_60m', 't5_220m', 't5_3b', 'transformer']
# air_crafts = ['f16', '787-8']
# step_num = 1
# inputsYs = []
# desc_info = []
# for air_craft in air_crafts:
#     loss_kllm_dict = {}
#     for k_llm in k_llms:
#         loss = np.load(os.path.join(os.getcwd(), args.finetuned_checkpoint, 'kine_llm_eval', k_llm, 'loss_function', 
#                         '{}loss_list_v0.npy'.format(air_craft)), allow_pickle=True)
#         # print(loss.shape)
#         loss_kllm_dict[k_llm] = loss[::step_num]
#     inputsYs.append(loss_kllm_dict)
#     desc_info.append({'air_craft':air_craft})
# line_multi_graphs_02(np.arange(0, 1500, step_num), inputsYs, ["Iteration", "Loss"], 
#                 os.path.join(os.getcwd(), args.result_data_dir, 'kllm_loss.pdf'), y_label_list=['gpt2', 't5_60m', 't5_220m', 'transformer'],
#                 desc_info=desc_info)

k_llms = ['gpt2', 'qwen2.5_1.5b', 't5_60m', 't5_220m', 't5_3b', 'transformer']
air_crafts = ['f16', '787-8']
for air_craft in air_crafts:
    for k_llm in k_llms:
        time = np.load(os.path.join(os.getcwd(), args.finetuned_checkpoint, 'kine_llm_eval', k_llm, 'loss_function', 
                        '{}time_list_v0.npy'.format(air_craft)), allow_pickle=True)
        print(time[-1] - time[0])