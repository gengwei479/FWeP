import os
import numpy as np
import json
import dashscope
from utils.loadcsv import trans_from_zjenv_to_mrad
from trans_trainer.llm_utils.prompt import PromptData
from langchain_core.utils.function_calling import convert_to_openai_tool
from dashscope import Generation
import re
import logging

class QWEN():
    def __init__(self, args) -> None:
        self.args = args
        dashscope.api_key = self.args.qwen_api_key
        self.master_data = self._load_data(os.path.join(os.getcwd(), self.args.master_data_dir, self.args.sear_version))
        self.prompt_data = PromptData(self.master_data)

    def _load_data(self, data_file):
        dirs = os.listdir(data_file)
        total_obs_withouttoken_mrad = []
        for file in dirs:
            if '_' + self.args.aircraft[0] + '_traj.npy' in file:
                trajectory = np.load(os.path.join(data_file, file), allow_pickle=True)#[:100]                    
                obs_data = trans_from_zjenv_to_mrad([trajectory[it]['cur_observation'] for it in range(min(self.args.max_seq_length, len(trajectory)))])
                total_obs_withouttoken_mrad.append(obs_data)
        
        master_data = np.array(total_obs_withouttoken_mrad).reshape([-1, self.args.max_seq_length, 6])
        return master_data
    
    def get_prompt_res(self, message):
        response = Generation.call(
            model=self.args.sty_llm,
            messages=message,
        )
        return response

def auto_prompt(prompt_res_dict):
    prompt = ''
    for key, value in prompt_res_dict.items():
        prompt += 'Your {prop} is {stat}, '.format(prop = str(key), stat = str(value))
    return prompt

def task_prompt_main(args, task_desc = 'You need to fly to Samsun. ', cur_obs_history = [trans_from_zjenv_to_mrad([41.0, 38.0, 5000.0, 0.0, 0.0, 0.0])], prompt_logger = None):
    qwen = QWEN(args)
    prompt_res_dict = {}
    #______________________________global data______________________________
    master_data = qwen.master_data
    scratch = cur_obs_history
    executed_traj = []
    global_val_names = locals()
    print('observation history is: ' + str(scratch))
    invaild_info = None
    is_finished = False
    while True:

        if len(prompt_res_dict) == 0:
            message = qwen.prompt_data.get_prompt_template(task_desc)
        else:
            message = qwen.prompt_data.get_prompt_template(query = auto_prompt(prompt_res_dict))

        res = qwen.get_prompt_res(message)
        try:
            res_json_list = []
            for item in res['output']['text'].split("```"):
                item = item.replace('(json)', 'json')

                item = re.sub("\.[\s]{0,}]", ".00]", item)
                item = re.sub("\.[\s]{0,},", ".00,", item)

                reinfo = re.compile('\/\/.*\\n')
                item = reinfo.sub('\n', item)

                reinfo = re.compile('\/\/.*\.')
                item = reinfo.sub('\n', item)

                print('json:\n ' + str(item))
                prompt_logger.json_insert("\n{}".format(item))

                item_list = item.split('json')
                for it_item in item_list:
                    if it_item != '' and it_item != '\n':
                        res_json_list.append(json.loads(it_item.strip('json')))

                # if 'json' in item:
                #     res_json_list.append(json.loads(item.strip('json')))
            print('------------------------Length is: ' + str(len(res_json_list)) + '------------------------')
            print('------------------------Prompt is ' + str(prompt_res_dict))

            for res_json_item in res_json_list:
                action_info = res_json_item.get("action")

                if type(action_info) == dict:
                    action_name = action_info.get("name")
                    action_args = action_info.get("args")
                    action_args_dec = {}
                    for key, value in action_args.items():
                        if isinstance(value, str):
                            try:
                                action_args_dec[key] = eval(value)
                            except NameError:
                                action_args_dec[key] = value
                        else:
                            action_args_dec[key] = value
                    func = qwen.prompt_data.tools_map.get(action_name)
                    call_function_result = func(**action_args_dec)

                    if action_name == 'CalRelAngle' or action_name == 'CalRelHei':
                        prompt_res_dict[action_info.get("return")] = call_function_result * 100
                    else:
                        prompt_res_dict[action_info.get("return")] = call_function_result
                    if action_info.get("return") == "trajectory":
                        executed_traj.append(call_function_result)
                        is_finished = True
                        break
                    elif action_info.get("return") == 'finished':
                        is_finished = True
                        break
                elif type(action_info) == list:
                    for action_info_item in action_info:
                        action_name = action_info_item.get("name")
                        action_args = action_info_item.get("args")
                        action_args_dec = {}
                        for key, value in action_args.items():
                            if isinstance(value, str):
                                try:
                                    action_args_dec[key] = eval(value)
                                except NameError:
                                    action_args_dec[key] = value
                            else:
                                action_args_dec[key] = value
                        func = qwen.prompt_data.tools_map.get(action_name)
                        call_function_result = func(**action_args_dec)

                        if action_name == 'CalRelAngle' or action_name == 'CalRelHei':
                            prompt_res_dict[action_info_item.get("return")] = call_function_result * 100
                        else:
                            prompt_res_dict[action_info_item.get("return")] = call_function_result
                        # prompt_res_dict[action_info_item.get("return")] = call_function_result
                        if action_info_item.get("return") == "trajectory":
                            executed_traj.append(call_function_result)
                            is_finished = True
                            break
                        elif action_info_item.get("return") == 'finished':
                            is_finished = True
                            break 
                            

            if is_finished:
                break
        except TypeError as e:
            print(e)
            invaild_info = res
            break
    prompt_res_dict.clear()
    return executed_traj, invaild_info