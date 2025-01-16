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

class PILOT():
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
        cur_obs_history = []
        total_episode_reward = []
        prompt_time = 0
        done = False

        qwen = QWEN(self.args)
        task_prompt = "You need to fly to " + self.env.dest_name + "."
        message = qwen.prompt_data.get_prompt_template(task_prompt)

        # print(f"TASK PROMPT:\n{}".format(str(message)))
        self.prompt_logger.json_insert("TASK PROMPT:\n")
        self.prompt_logger.json_insert_head(message)
        while not done:
            if prompt_time >= self.args.prompt_epoch:
                break
            prompt_time += 1
            if self.cur_obs['player1']['position_h_sl_ft'] < 50:
                break 
            
            print('current observation is: ' + str(self.cur_obs))
            cur_obs_history.append(trans_from_zjenv_to_mrad(list(self.cur_obs['player1'].values())[0 : self.args.obs_dim]))
            task_prompt = "You need to fly to " + self.env.dest_name + "."
            response_list, invaild_info = task_prompt_main(self.args, task_desc = task_prompt, cur_obs_history = cur_obs_history, prompt_logger=self.prompt_logger)
            if invaild_info is not None:
                print(invaild_info)
                self.prompt_logger.json_insert("Now invaild_info is:\n" + invaild_info)
                break

            print('------------res:-----------')
            print(response_list)
            self.prompt_logger.json_insert("\nThe recommended trajectory from {} is:\n{}".format(self.args.sty_llm, response_list))
            
            for response in response_list:
                self.cur_obs, rew_list, done = self.adjust_plan(100, self.cur_obs, False)
                total_episode_reward += rew_list
                act_list = self.kinetic_solver.generate_from_rmad(response)
                self.prompt_logger.json_insert("\nThe recommended actions from {} are:\n{}".format(self.args.kine_llm, act_list))
                if self.env.fig is not None:
                    self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(self.cur_obs['player1'].values())[0 : self.args.obs_dim])[:3], 'color': 'blue'})
                    self.env.fig.auxiliary_line = response
                for act in act_list:
                    self.cur_obs, reward, done, _ = self.env.step({'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], act))})
                    total_episode_reward.append(reward)
                    self.logger.log_insert("evaluate_step:{} \t evaluate_reward:{}".format(self.env.step_num, reward), logging.INFO)
                    if abs(self.cur_obs['player1']['attitude_roll_rad']) >= 1.5 and self.cur_obs['player1']['velocities_v_down_fps'] >= 120:#200
                        print('attitude_roll_rad is: ' + str(self.cur_obs['player1']['attitude_roll_rad']) + ' and velocities_v_down_fps is: ' + str(self.cur_obs['player1']['velocities_v_down_fps']))
                        break
                    elif self.cur_obs['player1']['position_h_sl_ft'] <= 3000 and self.cur_obs['player1']['velocities_v_down_fps'] >= 120:#120
                        print('position_h_sl_ft is: ' + str(self.cur_obs['player1']['position_h_sl_ft']) + ' and velocities_v_down_fps is: ' + str(self.cur_obs['player1']['velocities_v_down_fps']))#3000
                        break

        
        self.env.complete()
        obs_traj = self.env.obs_traj
        
        if self.save_csv:
            write_result_csv_data(trans_from_zjenv_to_csv(obs_traj), os.path.join(os.getcwd(), self.args.result_data_dir, 'result_traj_' + self.args.aircraft[0] + '_' + self.args.sty_llm + ' ' + self.args.kine_llm + '_' + time.ctime().replace(' ', '_').replace(':', '_') + '.csv'))

        return total_episode_reward[:self.env._max_episode_steps]
    
    def adjust_plan(self, step_num, cur_obs, can_break = False):
        if self.env.fig is not None:
            self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : self.args.obs_dim])[:3], 'color': 'red'})
        is_stable = False
        init_action = [0, 0, 0, 1]
        rew_list = []
        done = False
        for id in range(step_num):
            if can_break and is_stable:
                break
            if abs(cur_obs['player1']['velocities_v_down_fps']) < 1 and cur_obs['player1']['velocities_v_down_fps'] < 0:
                is_stable = True
            act_list, is_stable = self.env.rule_act.idle(init_action = init_action, is_stable = is_stable)
            act_dict = {'player1': act_list}
            cur_obs, reward, done, _ = self.env.step(act_dict)
            rew_list.append(reward)
            self.logger.log_insert("evaluate_step:{} \t evaluate_reward:{}".format(self.env.step_num, reward), logging.INFO)
        return cur_obs, rew_list, done