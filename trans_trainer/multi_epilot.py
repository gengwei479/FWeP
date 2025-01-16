import os
import numpy as np
import json
from copy import deepcopy
from utils.loadcsv import trans_from_zjenv_to_mrad, write_result_csv_data, trans_from_zjenv_to_csv
from envs.env_wrapper import jsbsimSingleENV
from trans_trainer.trans_gpt2 import TRANSGPT
from trans_trainer.QWEN_planner import task_prompt_main
import dashscope
import time
import logging
from log.log_process import Logger
from trans_trainer.QWEN_planner import QWEN

class MULPILOT():
    def __init__(self, args, env_evaluate, save_csv = False, logger = None, prompt_logger = None) -> None:
        self.args = args
        dashscope.api_key = self.args.qwen_api_key
        
        self.env = env_evaluate
        main_condition = self.env.reset()
        self.cur_obs = main_condition
        if self.env.fig is not None:
            self.env.fig.auxiliary_points = []
        self.save_csv = save_csv
        
        self.cur_traj_mrad = None
        self.logger = logger
        self.prompt_logger = prompt_logger
        self.kinetic_solver = TRANSGPT(self.args)
        pass

    def pilot_traj(self):# Rize Samsun
        cur_obs_history = {item: [] for item in self.args.agent_keys}
        total_episode_reward = []
        prompt_time = 0
        done = False

        qwen = {}
        message = {}
        for agent_id, agent_name in enumerate(self.args.agent_keys):
            qwen[agent_name] = QWEN(self.args)
            if agent_id == self.env.leader_id:
                task_prompt = "You need to fly to " + self.env.dest_name + "."
            else:
                task_prompt = "Your leader's position is " + str(list(self.cur_obs[self.args.agent_keys[self.env.leader_id]].values())[0 : 3]) + ", which you need to follow it." + "You need to form a triangular formation with your leader where each side of the triangle is between 50 to 100 meters long."
            message[agent_name] = qwen[agent_name].prompt_data.get_prompt_template(task_prompt)

        # print(f"TASK PROMPT:\n{}".format(str(message)))
        self.prompt_logger.json_insert("TASK PROMPT:\n")
        self.prompt_logger.json_insert_head_mul(message)
        while not done:
            if prompt_time >= self.args.prompt_epoch:
                break
            prompt_time += 1
            # if self.cur_obs['player1']['position_h_sl_ft'] < 50:
            #     break 
            response_list_dict = {}

            print('current observation is: ' + str(self.cur_obs))
            for agent_id, agent_name in enumerate(self.args.agent_keys):
                cur_obs_history[agent_name].append(trans_from_zjenv_to_mrad(list(self.cur_obs[agent_name].values())[0 : self.args.obs_dim]))
                if agent_id == self.env.leader_id:
                    task_prompt = "You need to fly to " + self.env.dest_name + "."
                else:
                    task_prompt = "Your leader's position is " + str(list(self.cur_obs[self.args.agent_keys[self.env.leader_id]].values())[0 : self.args.obs_dim]) + ", which you need to follow it."
                response_list, invaild_info = task_prompt_main(self.args, task_desc = task_prompt, cur_obs_history = cur_obs_history[agent_name], prompt_logger=self.prompt_logger)
                if invaild_info is not None:
                    print(invaild_info)
                    break

                print('------------res:-----------')
                print(response_list)
                response_list_dict[agent_name] = response_list
            self.prompt_logger.json_insert("\nThe recommended trajectory from {} is:\n{}".format(self.args.sty_llm, response_list_dict))
            act_list_dict = {}

            if len(response_list_dict) != len(self.args.agent_keys):
                continue

            for response_id, _ in enumerate(response_list_dict[self.args.agent_keys[self.env.leader_id]]):
                self.cur_obs, rew_list, done = self.adjust_plan(100, self.cur_obs, False)
                total_episode_reward += rew_list

                for agent_id, agent_name in enumerate(self.args.agent_keys):
                    act_list = self.kinetic_solver.generate_from_rmad(response_list_dict[agent_name][response_id])
                    act_list_dict[agent_name] = act_list
                    self.prompt_logger.json_insert("\nThe recommended actions of {} from {} are:\n{}".format(agent_name, self.args.kine_llm, act_list))
                    if self.env.fig is not None and agent_id == self.env.leader_id:
                        self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(self.cur_obs['player1'].values())[0 : self.args.obs_dim])[:3], 'color': 'blue'})
                        self.env.fig.auxiliary_line = response_list_dict[self.args.aircraft[agent_id]][response_id]

                for act_id, _ in enumerate(act_list_dict[self.args.agent_keys[self.env.leader_id]]):
                    act_dict = {}
                    for agent_id, agent_name in enumerate(self.args.agent_keys):
                        act_dict[agent_name] = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], act_list_dict[agent_name][act_id]))
                    self.cur_obs, reward, done, _ = self.env.step(act_dict)
                    total_episode_reward.append(reward)
                    self.logger.log_insert("evaluate_step:{} \t evaluate_reward:{}".format(self.env.step_num, reward), logging.INFO)

                    break_down = False
                    for agent_id, agent_name in enumerate(self.args.agent_keys):
                        if abs(self.cur_obs[agent_name]['attitude_roll_rad']) >= 1.5 and self.cur_obs[agent_name]['velocities_v_down_fps'] >= 120:#200
                            print('attitude_roll_rad is: ' + str(self.cur_obs[agent_name]['attitude_roll_rad']) + ' and velocities_v_down_fps is: ' + str(self.cur_obs[agent_name]['velocities_v_down_fps']))
                            break_down = True
                        elif self.cur_obs[agent_name]['position_h_sl_ft'] <= 3000 and self.cur_obs[agent_name]['velocities_v_down_fps'] >= 120:#120
                            print('position_h_sl_ft is: ' + str(self.cur_obs[agent_name]['position_h_sl_ft']) + ' and velocities_v_down_fps is: ' + str(self.cur_obs[agent_name]['velocities_v_down_fps']))#3000
                            break_down = True
                    if break_down:
                        break

        
        self.env.complete()
        obs_traj = self.env.obs_traj
        
        if self.save_csv:
            for agent_id, agent_name in enumerate(self.args.agent_keys):
                write_result_csv_data(trans_from_zjenv_to_csv(obs_traj[agent_name]), os.path.join(os.getcwd(), self.args.result_data_dir, 'result_traj_' + self.args.env_name + '_' + agent_name + '_' + self.args.aircraft[0] + '_' + self.args.sty_llm + ' ' + self.args.kine_llm + '_' + time.ctime().replace(' ', '_').replace(':', '_') + '.csv'))

        return total_episode_reward[:self.env._max_episode_steps]

    def adjust_plan(self, step_num, cur_obs, can_break = False):
        for agent_id, agent_name in enumerate(self.args.agent_keys):
            if self.env.fig is not None and agent_id == self.env.leader_id:
                self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(cur_obs[agent_name].values())[0 : self.args.obs_dim])[:3], 'color': 'red'})
        is_stable = [False for _ in range(self.args.num_agents)]
        init_action = [0, 0, 0, 1]
        rew_list = []
        act_dict = {}
        done = False
        for id in range(step_num):
            if can_break and all(is_stable):
                break
            for agent_id, agent_name in enumerate(self.args.agent_keys):
                if abs(cur_obs[agent_name]['velocities_v_down_fps']) < 1 and cur_obs[agent_name]['velocities_v_down_fps'] < 0:
                    is_stable[agent_id] = True
                act_list, is_stable[agent_id] = self.env.rule_act[agent_name].idle(init_action = init_action, is_stable = is_stable)
                act_dict[agent_name] = act_list
            cur_obs, reward, done, _ = self.env.step(act_dict)
            rew_list.append(reward)
            self.logger.log_insert("evaluate_step:{} \t evaluate_reward:{}".format(self.env.step_num, reward), logging.INFO)
        return cur_obs, rew_list, done