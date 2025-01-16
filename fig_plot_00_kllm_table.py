from arg_parses import create_parses, pre_parses
# from utils.viaual_fun import line_graphs_01, line_graphs_02

from trans_trainer.trans_base_eval import TRANSBASE
from trans_trainer.trans_gpt2_eval import TRANSGPT
from trans_trainer.trans_qwen2_eval import TRANSQWEN2
from trans_trainer.trans_t5_eval import TRANST5

args = create_parses()
args = pre_parses(args)
# translate = TRANSGPT(args)
# translate.generate_acc_test()

import os
import numpy as np
import matplotlib


# k_llms = ['gpt2', 'qwen2.5_1.5b', 't5_60m', 't5_220m', 't5_3b', 'transformer']
air_crafts = ['f16']

def list_to_str(input_list):
    res = ''
    for item in input_list:
        res += str(item) + '\n'
    return res

for air_craft in air_crafts:
    file_name = os.path.join(os.getcwd(), args.result_data_dir, "kllm_train_test_table.txt")
    with open(file_name, "a") as f:
        # f.write("\nGPT2:\n")
        # args.kine_llm = "gpt2"
        # gpt_context = TRANSGPT(args).tester()
        # f.write(list_to_str(gpt_context))

        # f.write("\nTransformer:\n")
        # args.kine_llm = "transformer"
        # gpt_context = TRANSBASE(args).tester()
        # f.write(list_to_str(gpt_context))

        f.write("\nQwen2.5 (1.5B):\n")
        args.kine_llm = "qwen2.5_1.5b"
        gpt_context = TRANSQWEN2(args).tester()
        f.write(list_to_str(gpt_context))

        f.write("\nT5 (60M):\n")
        args.kine_llm = "t5_60m"
        gpt_context = TRANST5(args).tester()
        f.write(list_to_str(gpt_context))

        f.write("\nT5 (220M):\n")
        args.kine_llm = "t5_220m"
        gpt_context = TRANST5(args).tester()
        f.write(list_to_str(gpt_context))

        f.write("\nT5 (3B):\n")
        args.kine_llm = "t5_3b"
        gpt_context = TRANST5(args).tester()
        f.write(list_to_str(gpt_context))