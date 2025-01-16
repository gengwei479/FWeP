import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, GPT2LMHeadModel
from tokenizer.AutoTokenizerForGPT2 import AutoTokenizerForGPT2v1, AutoTokenizerForGPT2v2
from tqdm.auto import tqdm
from copy import deepcopy
from utils.gpu_alloc import use_gpu

from envs.env_wrapper import jsbsimSingleENV
from visual.plot_utils import mainWin
from utils.loadcsv import trans_from_zjenv_to_mrad, trans_from_zjenv_to_csv, write_result_csv_data

class TRANSDATA(Dataset):
    def __init__(self, data_file, max_size, max_seq_length, obs_tokenizer_tool, act_tokenizer_tool, aircraft, llm, vab_size):
        self.max_size = max_size
        self.max_seq_length = max_seq_length
        self.obs_tokenizer_tool = obs_tokenizer_tool
        self.act_tokenizer_tool = act_tokenizer_tool
        self.aircraft = aircraft
        self.llm = llm
        self.obs_data, self.act_data, self.total_obs_withouttoken_mrad, self.total_act_withouttoken_mrad = self.load_data(data_file, self.obs_tokenizer_tool, self.act_tokenizer_tool, vab_size)
        self.attention_mask = torch.ones(torch.concat([self.obs_data, self.act_data], dim=-1).shape, dtype=torch.long)
    
    def load_data(self, data_file, obs_tokenizer_tool, act_tokenizer_tool, vab_size):            
        dirs = os.listdir(data_file)
        total_obs_data = torch.tensor([], dtype=torch.long)
        total_act_data = torch.tensor([], dtype=torch.long)
        total_obs_withouttoken_mrad = []
        total_act_withouttoken_mrad = []
        for file in dirs:
            if '_' + self.aircraft + '_traj.npy' in file:
                trajectory = np.load(os.path.join(data_file, file), allow_pickle=True)#[:100]                  
                obs_data, act_data = [], []
                for id, item in enumerate([0]):               
                    obs_data = trans_from_zjenv_to_mrad([trajectory[id + it]['cur_observation'] for it in range(min(self.max_seq_length, len(trajectory)))])
                    act_data = [trajectory[id + it]['action'] for it in range(min(self.max_seq_length, len(trajectory)))]
                    total_obs_withouttoken_mrad.append(deepcopy(obs_data))
                    total_act_withouttoken_mrad.append(deepcopy(act_data))
                    obs_data = obs_tokenizer_tool.obs_tokenize(obs_data - obs_data[0]).transpose(0, 1)    
                    act_data = act_tokenizer_tool.act_tokenize(act_data).reshape(1, -1)
                    
                    total_obs_data = torch.concat((total_obs_data, obs_data), dim=0)
                    total_act_data = torch.concat((total_act_data, act_data), dim=0)
                
        assert torch.lt(total_obs_data, vab_size).all()
        assert torch.ge(total_obs_data, 0).all()
        assert torch.lt(total_act_data, vab_size).all()
        assert torch.ge(total_act_data, 0).all()
        return total_obs_data, total_act_data, total_obs_withouttoken_mrad, total_act_withouttoken_mrad
    
    def __len__(self):
        return len(self.obs_data)

    def __getitem__(self, idx):
        return self.obs_data[idx], self.act_data[idx], self.attention_mask[idx]

class TRANSGPT():
    def __init__(self, args) -> None:
        self.args = args
        self.vab_size = 50256
        self.obs_token_num = [0, 0, 0, 36, 36, 36]
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.5)
        
        self.obs_tokenizer_tool = AutoTokenizerForGPT2v1(self.vab_size, obs_token_num=self.obs_token_num)
        self.act_tokenizer_tool = AutoTokenizerForGPT2v2(self.vab_size)
        
        self.train_data = TRANSDATA(os.path.join(os.getcwd(), self.args.master_data_dir, self.args.sear_version), max_size = self.args.max_dataset_size, max_seq_length = self.args.max_seq_length, 
                               obs_tokenizer_tool = self.obs_tokenizer_tool, act_tokenizer_tool = self.act_tokenizer_tool,
                               aircraft = self.args.aircraft[0], llm = self.args.kine_llm, vab_size = self.vab_size)
        self.dataloader = DataLoader(self.train_data, batch_size = self.args.batch_size, shuffle=True)

        self.checkpoint = os.path.join(self.args.pretrained_checkpoint, self.args.kine_llm)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = AutoConfig.from_pretrained(self.checkpoint)
        self.model = GPT2LMHeadModel(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        
        self.finetuned_model = None

        self.load_finetuned_model()

    def load_finetuned_model(self):
        if os.path.exists(os.path.join(self.args.finetuned_checkpoint, self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)):
            finetuned_checkpoint = os.path.join(self.args.finetuned_checkpoint, self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)
            self.finetuned_model = GPT2LMHeadModel.from_pretrained(finetuned_checkpoint).to(self.device)

    def trainer(self):
        print('-----{} trains start!!-----'.format(self.args.kine_llm))
        # np.save(os.path.join(os.getcwd(), self.args.result_data_dir, self.args.aircraft_type + 'loss_list.npy'), np.array([1,2,3]))
        loss_values = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                obs_seq, act_seq, attention_seq = batch
                attention_seq = attention_seq.to(self.device)
                
                input_ids = torch.concat([obs_seq, act_seq], dim=-1).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_seq, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.dataloader)
            loss_values.append(avg_loss)

            if (epoch+1) % 10 == 0:
                avg_loss = total_loss / len(self.dataloader)
                perplexity = math.exp(avg_loss)
                print(f'Epoch [{epoch+1}/{self.args.num_epochs}], Loss: {avg_loss:.7f}, Perplexity: {perplexity:.5f}')

        np.save(os.path.join(os.getcwd(), self.args.result_data_dir, 'loss_function', self.args.aircraft_type + 'loss_list_' + self.args.sear_version + '.npy'), np.array(loss_values))
        finetuned_model_path_dir = os.path.join(os.getcwd(), self.args.finetuned_checkpoint, self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)
        if not os.path.exists(finetuned_model_path_dir):
            os.makedirs(finetuned_model_path_dir)
        self.model.save_pretrained(finetuned_model_path_dir)

    def generate(self, input_obs):
        input_token = input_obs.unsqueeze(dim=0).to(self.device)
        output = self.finetuned_model.generate(input_token, max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
        act_list = self.act_tokenizer_tool.act_detokenize(output[:, self.args.max_seq_length:].reshape(-1, 4))
        return act_list.tolist()
    
    def generate_from_rmad(self, input_obs):
        obs_token = self.obs_tokenizer_tool.obs_tokenize(input_obs - input_obs[0]).transpose(0, 1)
        input_token = obs_token.to(self.device)
        output = self.finetuned_model.generate(input_token, max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
        act_list = self.act_tokenizer_tool.act_detokenize(output[:, self.args.max_seq_length:].reshape(-1, 4))
        return act_list.tolist()
    
    def generate_acc_test(self):        
        self.load_finetuned_model()        
        obs_data = self.train_data.obs_data
        for id, item in enumerate(obs_data):
            input_token = item.unsqueeze(dim=0)
            output = self.finetuned_model.generate(input_token.to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
            loss = sum(self.train_data.act_data[id].to(self.device).eq(output.squeeze()[self.args.max_seq_length:]))
            print(str(id) + ' ' + str(self.train_data.act_data[id][:5]) + ' ' + str(output.squeeze()[self.args.max_seq_length:][:5]))
            print(loss)
        return output
    
    def generate_env_test(self, test_id = 8, fig_show = True, save_csv = False):
        self.load_finetuned_model()
        
        initial_condition = {}
        initial_condition['player1'] = {
            'position_lat_geod_deg': 41, 'position_long_gc_deg': 38, 'position_h_sl_ft': 5000, 'initial_heading_degree': 0, 
            'velocities_v_east_fps': 707, 'velocities_v_north_fps': 707, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }
        env = jsbsimSingleENV(self.args, contain_rule=False, isConstistShow=fig_show, isEndShow=fig_show)
        cur_obs = env.reset(initial_condition)
        env.fig.auxiliary_line = self.train_data.total_obs_withouttoken_mrad[test_id]
        act_list = self.generate(self.train_data.obs_data[test_id])#0
        
        stable_id = np.power(np.array(self.train_data.total_obs_withouttoken_mrad[test_id][int(self.args.max_seq_length / 2):, 3:5]) - np.array([[0, 0]]), 2).argmin()
        print(self.train_data.total_obs_withouttoken_mrad[test_id][stable_id])
        
        id = 0
        total_id = 0
        for id in range(self.args.max_seq_length):
            act = act_list[id]
            id += 1
            total_id += 1
            act_dict = {'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], act))}
            cur_obs = env.step(act_dict)
            if stable_id == id:
                act_list.clear()
                id = 0
                act_list = self.generate(self.train_data.obs_data[test_id])#0

            if total_id >= 1000:
                break
            
        env.complete()
        obs_traj = env.obs_traj
        if save_csv:
            write_result_csv_data(trans_from_zjenv_to_csv(obs_traj), self.args.result_data_dir + 'result_traj_' +
                                  self.args.aircraft[0] + '_' + self.args.kine_llm +  '.csv')

    def master_env_test(self, test_id = 8, fig_show = True, save_csv = False):
        self.env = jsbsimSingleENV(self.args, contain_rule=True, isConstistShow=fig_show, isEndShow=fig_show, auxiliary_line=self.train_data.total_obs_withouttoken_mrad[test_id])
        self.env.fig.auxiliary_points = []
        main_condition = self.env.reset()
        cur_obs = main_condition
        cur_obs = self.adjust_plan(100, main_condition, cur_obs)
        
        act_list = self.train_data.total_act_withouttoken_mrad[test_id]
        for id in range(len(act_list)):#200
            act = act_list[id % len(act_list)]
            act_dict = {'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], act))}
            cur_obs = self.env.step(act_dict)
        self.env.complete()

    def test_env_gpt2_clip(self, test_id = 70, fig_show = True):
        self.env = jsbsimSingleENV(self.args, contain_rule=False, isConstistShow=fig_show, isEndShow=fig_show)
        initial_condition = {}
        initial_condition['player1'] = {
            'position_lat_geod_deg': 41, 'position_long_gc_deg': 38, 'position_h_sl_ft': 5000, 'initial_heading_degree': 0, 
            'velocities_v_east_fps': 0, 'velocities_v_north_fps': 1000, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }
        main_condition = self.env.reset(initial_condition)
        cur_obs = main_condition
        self.env.fig.auxiliary_points = []
        self.load_finetuned_model()
        
        cur_obs = self.adjust_plan(100, main_condition, cur_obs)
        for _ in range(3):
            act_list = self.generate(self.train_data.obs_data[test_id])
            self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : self.args.obs_dim])[:3], 'color': 'blue'})
            self.env.fig.auxiliary_line = self.train_data.total_obs_withouttoken_mrad[test_id]
            for act in act_list:
                # if cur_obs['player1']['attitude_roll_rad'] >= 1 and cur_obs['player1']['velocities_v_down_fps'] >= 50:
                #     print(str(cur_obs['player1']['attitude_roll_rad']) + ' ' + str(cur_obs['player1']['velocities_v_down_fps']))
                #     act[0] = -1#'aileron'
                
                cur_obs, _ = self.env.step({'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], act))})
            cur_obs = self.adjust_plan(100, main_condition, cur_obs)
        self.env.complete()
        
    def test_env_env_north_pg(self):
        self.env = jsbsimSingleENV(self.args, contain_rule=False, isConstistShow=True, isEndShow=True)
        initial_condition = {}
        initial_condition['player1'] = {
            'position_lat_geod_deg': 41, 'position_long_gc_deg': 38, 'position_h_sl_ft': 5000, 'initial_heading_degree': 45, 
            'velocities_v_east_fps': 707, 'velocities_v_north_fps': 707, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }
        main_condition = self.env.reset(initial_condition)
        cur_obs = main_condition
        self.env.fig.auxiliary_points = []
        for id in range(200):
            action = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 1]))
            action = {'player1': self.env.rule_act.adjust_to_north(action, -45 - trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[5] + trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[5],
                                                                        0 + trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[2] - trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[2])}
            # action = {'player1': self.env.rule_act.adjust_to_north(action, 0 + trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[2] - trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[2])}
            cur_obs, _ = self.env.step(action)
            print(cur_obs)
            print('\n')
        self.env.complete()
        

    def traj_plan(self, step_num, main_condition, cur_obs, act_list = None):
        print('----------------------------------traj---------------------------------')
        self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : self.args.obs_dim])[:3], 'color': 'blue'})
        new_act_list = []
        for id in range(step_num):
            action = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 1]))
            if act_list is None:
                act_dict = {'player1': self.env.rule_act.goto_point(action, -90 + trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[5] - trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[5],
                                                                        0 + trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[2] - trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[2])}#
            else:
                act_dict = act_list[id]
            cur_obs, _ = self.env.step(act_dict)
            new_act_list.append(act_dict)
        return cur_obs, new_act_list
    
    
    def adjust_plan(self, step_num, main_condition, cur_obs, can_break = False):
        print('----------------------------------adjust---------------------------------')
        main_condition = deepcopy(cur_obs)
        self.env.fig.auxiliary_points.append({'pos': trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : self.args.obs_dim])[:3], 'color': 'red'})
        is_stable = False
        init_action = [0, 0, 0, 1]
        for id in range(step_num):
            # print(is_stable)
            if can_break and is_stable:
                break
            if abs(cur_obs['player1']['velocities_v_down_fps']) < 5:
                is_stable = True
            action = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], init_action))
            if not is_stable:
                act_dict = {'player1': self.env.rule_act.idle(init_action = init_action)[0]}
                # act_dict = {'player1': self.env.rule_act.goto_point(action, 0 + trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[5] - trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[5],
                #                                                 0 + trans_from_zjenv_to_mrad(list(main_condition['player1'].values())[0 : 6])[2] - trans_from_zjenv_to_mrad(list(cur_obs['player1'].values())[0 : 6])[2])}
            else:
                act_dict = {'player1': self.env.rule_act.idle(init_action = init_action)[0]}
            # print(act_dict)
            cur_obs, _ = self.env.step(act_dict)
        return cur_obs
    
    def visial_all_path(self):
        from matplotlib import pyplot as plt
        ax1 = plt.axes(projection='3d')
        # aaa = self.train_data.total_obs_withouttoken_mrad
        for traj in self.train_data.total_obs_withouttoken_mrad:
            # print(traj)
            ax1.plot([item[0] for item in traj], [item[1] for item in traj], [item[2] for item in traj])
        plt.show()
    
    def create_pds_about_total_obs_withouttoken_mrad(self):
        import pandas as pd
        
        # trajs = {}
        # for id, traj in enumerate(self.train_data.total_obs_withouttoken_mrad):
        #     trajs['trajectory' + str(id)] = [dict(zip(['x_position(latitude, unit meter)', 'y_position(longitude, unit meter)', 'z_position(altitude, unit meter)', 'roll(unit rad)', 'pitch(unit rad)', 'yaw(unit rad)'], item)) for item in traj]
        # trajs_dict_pd = pd.DataFrame(trajs)
        # print(trajs_dict_pd.info())
        # print(trajs_dict_pd.at[0, 'trajectory0']['x_position(latitude, unit meter)'])
        
        traj_num = len(self.train_data.total_obs_withouttoken_mrad)
        traj_length_list = [len(item) for item in self.train_data.total_obs_withouttoken_mrad]
        # print(['trajectory' + str(id) for id in range(traj_num) for i in range(traj_length_list[id])])#
        # print([i for id in range(traj_num) for i in range(traj_length_list[id])])
        
        # print(self.train_data.total_obs_withouttoken_mrad[:5])
        # print(np.array(self.train_data.total_obs_withouttoken_mrad).reshape([-1, 6])[:5])
        
        trajs_dict_pd = pd.DataFrame(np.array(self.train_data.total_obs_withouttoken_mrad).reshape([-1,6]), 
                                     index=[['trajectory' + str(id) for id in range(traj_num) for i in range(traj_length_list[id])], [i for id in range(traj_num) for i in range(traj_length_list[id])]], 
                                     columns=['x_position(latitude, unit meter)', 'y_position(longitude, unit meter)', 'z_position(altitude, unit meter)', 'roll(unit rad)', 'pitch(unit rad)', 'yaw(unit rad)'])
        trajs_dict_pd.to_json(self.args.master_data_dir + 'master_data.json')
        print(trajs_dict_pd.loc['trajectory0'])
        # print(trajs_dict_pd.info())
        # print(trajs_dict_pd.loc['trajectory0', 1]['x_position(latitude, unit meter)'])