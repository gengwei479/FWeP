import os
import time
import math
import numpy as np
import torch
from copy import deepcopy
from torch import nn
from torch.utils.data import DataLoader
from tokenizer.AutoTokenizerForGPT2 import AutoTokenizerForGPT2v1, AutoTokenizerForGPT2v2
from envs.gym_jsbsim.zj_jsbsim import zj_jsbsim
from envs.gym_jsbsim.tasks import BattleTask
from visual.plot_utils import mainWin
from utils.gpu_alloc import use_gpu
from utils.loadcsv import trans_from_zjenv_to_mrad, trans_from_zjenv_to_csv, write_result_csv_data

from log.json_process import Texter
# https://blog.csdn.net/zhaohongfei_358/article/details/126019181

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformBlock(nn.Module):
    def __init__(self, device, d_model=128, num_embeddings = 10) -> None:
        super(TransformBlock, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=128, device = self.device)
        self.transformer = nn.Transformer(d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True, device = self.device)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0).to(self.device)
        
        self.predictor = nn.Linear(128, num_embeddings).to(self.device)
        
    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(self.device)
        tgt_mask = torch.where(tgt_mask == float("-inf"), 1, tgt_mask).to(dtype=torch.bool)
        src_key_padding_mask = self.gets_key_padding_mask(src).to(self.device)
        tgt_key_padding_mask = self.gets_key_padding_mask(tgt).to(self.device)
        
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        out = self.transformer.forward(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return out
    
    def gets_key_padding_mask(self, tokens):
        key_padding_mask = torch.zeros(tokens.size(), dtype=torch.bool)
        # key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask

# def generate_data_batch(batch_size, max_length=16, num_embeddings = 10):
#     src = []
#     for i in range(batch_size):
#         random_len = random.randint(1, max_length - 2)
#         random_nums = [0] + [random.randint(3, num_embeddings - 1) for _ in range(random_len)] + [1]
#         random_nums += [2] * (max_length - random_len - 2)
#         src.append(random_nums)
#     src = torch.LongTensor(src)
#     tgt = src[:, :-1]
#     tgt_y = src[:, 1:]
#     n_tokens = (tgt_y != 2).sum()
#     return src, tgt, tgt_y, n_tokens

class TRANSBASE():
    def __init__(self, args) -> None:
        self.args = args
        self.max_seq_length = self.args.max_seq_length
        self.total_loss = 0
        self.training_epoch = self.args.num_epochs
        self.num_embeddings = 50256#self.args.lowp_t_num_embeddings
        # self.tokenizer_v = self.args.token_version
        self.obs_token_num = [0, 0, 0, 36, 36, 36]
        self.aircraft = self.args.aircraft[0]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.5)
        
        # if self.tokenizer_v == 1:
        #     self.tokenizer_tool = AutoTokenizerForGPT2v1()
        # elif self.tokenizer_v == 2:
        #     self.tokenizer_tool = AutoTokenizerForGPT2v2()
        self.obs_tokenizer_tool = AutoTokenizerForGPT2v1(self.num_embeddings, obs_token_num=self.obs_token_num)
        self.act_tokenizer_tool = AutoTokenizerForGPT2v2(self.num_embeddings)

        self.model = TransformBlock(num_embeddings = self.num_embeddings, device = self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        
        self.train_data = self.load_data(os.path.join(os.getcwd(), self.args.master_data_dir, self.args.sear_version), flag = "train", train_ratio = args.train_ratio)
        self.test_data = self.load_data(os.path.join(os.getcwd(), self.args.master_data_dir, self.args.sear_version), flag = "test", train_ratio = args.train_ratio)
        self.dataloader = DataLoader(self.train_data, batch_size = self.args.batch_size, shuffle=True)

        self.finetuned_model = None
    
    def load_finetuned_model(self):
        finetuned_checkpoint = os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version, 'tfbase_model.pt')
        if os.path.exists(finetuned_checkpoint):
            self.finetuned_model = torch.load(finetuned_checkpoint).to(self.device)
            
    def load_data(self, data_file, flag, train_ratio):            
        dirs = os.listdir(data_file)
        total_obs_data = torch.tensor([], dtype=torch.long)
        total_act_data = torch.tensor([], dtype=torch.long)
        data = torch.tensor([], dtype=torch.long)

        file_list = [item for item in dirs if '_' + self.aircraft + '_traj.npy' in item]
        if flag == "train":
            data_set = file_list[:int(len(file_list) * train_ratio)]
        elif flag == "test":
            data_set = file_list[int(len(file_list) * train_ratio):]

        for file in data_set:
                # trajectory = np.load(data_file + file, allow_pickle=True)[:100]
                # latent = self.master_config[os.path.splitext(file)[0].split('_' + self.aircraft + '_traj')[0]]
                trajectory = np.load(os.path.join(data_file, file), allow_pickle=True)
                
                obs_data, act_data = [], []
                for id, item in enumerate([0]):#trajectory[:-self.max_seq_length]

                    obs_data = trans_from_zjenv_to_mrad([trajectory[id + it]['cur_observation'] for it in range(min(self.max_seq_length, len(trajectory)))])
                    act_data = [trajectory[id + it]['action'] for it in range(min(self.max_seq_length, len(trajectory)))]
                    obs_data = self.obs_tokenizer_tool.obs_tokenize(obs_data - obs_data[0]).transpose(0, 1)    
                    act_data = self.act_tokenizer_tool.act_tokenize(act_data).reshape(1, -1)
                    
                    total_obs_data = torch.concat((total_obs_data, obs_data), dim=0)
                    total_act_data = torch.concat((total_act_data, act_data), dim=0)



                #     if id == 0:
                #         last_observation = trajectory[0]['main_observation']
                #     else:
                #         last_observation = trajectory[id - 1]['cur_observation']
                #     obs_data_it, act_data_it = [], []
                #     obs_data_it.append(deepcopy(last_observation)) # 'action' 'cur_observation'
                #     for it in range(self.max_seq_length - 1):
                #         obs_data_it.append(trajectory[id + it]['cur_observation'])
                #         act_data_it.append(trajectory[id + it]['action'])
                #     act_data_it.append(trajectory[id + self.max_seq_length]['action'])
                    
                #     obs_data.append(obs_data_it)
                #     act_data.append(act_data_it)
                # obs_data = tokenizer_tool.obs_tokenize(obs_data)        
                # act_data = tokenizer_tool.act_tokenize(act_data)
                
                # if tokenizer_v == 1:
                #     tmp_shape = obs_data.shape
                #     total_data = torch.stack((obs_data, act_data), dim = -1).view(tmp_shape[0], tmp_shape[1] * 2)
                # elif tokenizer_v == 2:
                #     tmp_shape = obs_data.shape
                #     total_data = torch.concat((obs_data, act_data), dim = -1).view(tmp_shape[0], -1)
                
                # latent_data = torch.tensor([[latent]] * total_data.shape[0] ,dtype=torch.long)
        data = torch.concat((total_obs_data, total_act_data), dim = -1)
        return data
    
    # def load_data(self, data_file, tokenizer_tool, tokenizer_v):            
    #     dirs = os.listdir(data_file)
    #     total_obs_data = torch.tensor([], dtype=torch.long)
    #     total_act_data = torch.tensor([], dtype=torch.long)
    #     for file in dirs:
    #         if '_' + self.aircraft + '_traj.npy' in file:
    #             trajectory = np.load(os.path.join(data_file, file), allow_pickle=True)
                
    #             # for id, item in enumerate(trajectory):
    #             #     if id == 0:
    #             #         trajectory[0].update({'brevity': 0})
    #             #     else:
    #             #         trajectory[id].update({'brevity': sum(abs(np.array(trajectory[id]['action']) - np.array(trajectory[id-1]['action'])))})
    #             # print([item['brevity'] for item in trajectory])
                            
    #             obs_data = []
    #             for id, item in enumerate([0]):#trajectory[:-self.max_seq_length]
    #                 obs_data_it = []
    #                 for it in range(self.max_seq_length):
    #                     obs_data_it.append(trajectory[id + it]['cur_observation'])
                    
    #                 obs_data.append(obs_data_it)
                    
    #             obs_data = tokenizer_tool.obs_tokenize(obs_data)        
    #             total_obs_data = torch.concat((total_obs_data, obs_data), dim=0)
    #     return total_obs_data
    
    def trainer(self):
        print('-----{} trains start!!-----'.format(self.args.kine_llm))
        loss_values = []
        time_list = []
        time_list.append(time.time())
        for step in range(self.training_epoch):
            total_loss = 0
            for data in self.dataloader:
            # src, tgt, tgt_y, n_tokens = generate_data_batch(batch_size=2, max_length=self.max_length, num_embeddings = self.num_embeddings)
            # data = self.load_data(self.args.student_data_dir, self.tokenizer_tool, self.tokenizer_v)
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data, data)
                out = self.model.predictor(out)
                
                loss = self.criteria(out.contiguous().view(-1, out.size(-1)), data.contiguous().view(-1)) 
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss

            avg_loss = total_loss / len(self.dataloader)
            loss_values.append(avg_loss.detach().cpu())
            time_list.append(time.time())

            if (step + 1) % 10 == 0:
                avg_loss = total_loss / len(self.dataloader)
                perplexity = math.exp(avg_loss)
                print(f'Epoch [{step+1}/{self.args.num_epochs}], Loss: {avg_loss:.7f}, Perplexity: {perplexity:.5f}')
        
        np.save(os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, 'loss_function', self.args.aircraft_type + 'loss_list_' + self.args.sear_version + '.npy'), np.array(loss_values))
        np.save(os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, 'loss_function', self.args.aircraft_type + 'time_list_' + self.args.sear_version + '.npy'), np.array(time_list))
        finetuned_checkpoint = os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)
        if not os.path.exists(finetuned_checkpoint):
            os.makedirs(finetuned_checkpoint)
        torch.save(self.model, os.path.join(finetuned_checkpoint, 'tfbase_model.pt'))

    def eval(self, src_test, tgt_test, max_length):
        eval_model = self.finetuned_model
        for i in range(max_length):# 
            out = eval_model(src_test, tgt_test).to(self.device)
            predict = eval_model.predictor(out[:, -1])
            y = torch.argmax(predict, dim=1)
            tgt_test = torch.concat([tgt_test, y.unsqueeze(0)], dim=1)
        return tgt_test

    def tester(self, data_labels = ['training data', 'test data']):        
        self.load_finetuned_model()
        log_info_res = []
        for data_label in data_labels:
            if data_label == 'training data':
                obs_data = self.train_data[:, :self.args.max_seq_length]
            elif data_label == 'test data':
                obs_data = self.test_data[:, :self.args.max_seq_length]

            total_acc_num = []
            consume_time = []
            for id, item in enumerate(obs_data):
                input_token = item.unsqueeze(dim=0)
                pre_time = time.time()
                output = self.generate(input_token.to(self.device), max_length = self.args.max_seq_length * 4)
                after_time = time.time()
                consume_time.append(after_time - pre_time)
                if data_label == 'training data':
                    acc_num = sum(self.train_data[id, self.args.max_seq_length:].to(self.device).eq(output.squeeze()[self.args.max_seq_length:])) / len(self.train_data[id, self.args.max_seq_length:])
                elif data_label == 'test data':
                    acc_num = sum(self.test_data[id, self.args.max_seq_length:].to(self.device).eq(output.squeeze()[self.args.max_seq_length:])) / len(self.test_data[id, self.args.max_seq_length:])
                # print(acc_num)
                total_acc_num.append(acc_num / self.args.max_seq_length)
            log_info = "Accurate rate:{}, Inference time:{} in {}\n".format(sum(total_acc_num) / len(total_acc_num), sum(consume_time) / len(consume_time), data_label)
            print(log_info)
            log_info_res.append(log_info)
        return log_info_res

    # def tester_fast(self):
    #     self.load_finetuned_model()
    #     # loss_type = nn.CrossEntropyLoss()
    #     loss_type = nn.MSELoss()
    #     dest_data = torch.tensor(self.test_data)
    #     # output = []
    #     for id, item in enumerate(dest_data):
    #         if id == 0:
    #             output = self.generate(item[:self.args.max_seq_length].unsqueeze(dim=0).to(self.device), max_length = self.args.max_seq_length * 4)
    #         else:
    #             output = torch.cat((output, self.generate(item[:self.args.max_seq_length].unsqueeze(dim=0).to(self.device), max_length = self.args.max_seq_length * 4)))
    #     loss = loss_type(torch.tensor(dest_data, device=self.device)[:, self.args.max_seq_length:].to(dtype=torch.float), output[:, self.args.max_seq_length:].to(dtype=torch.float)) / self.num_embeddings
    #     print(loss)

    def tester_fast(self, data_labels = ['test data']):#'training data', 

        eval_txt = Texter("peft_eval.txt", os.path.join(os.getcwd(), self.args.log_dir))
        eval_txt.text_insert("\n-------------------------------" + self.args.fine_tune_mode + "-----" + self.args.aircraft_type + "----------------------------\n")

        self.load_finetuned_model()
        loss_type = nn.MSELoss()

        for data_label in data_labels:
            if data_label == 'training data':
                obs_data = self.train_data
                dest_data = self.train_data
            elif data_label == 'test data':
                obs_data = self.test_data
                dest_data = self.test_data

            dest_data = torch.tensor(dest_data)
            pre_time = time.time()

            for id, item in enumerate(dest_data):
                if id == 0:
                    output = self.generate(item[:self.args.max_seq_length].unsqueeze(dim=0).to(self.device), max_length = self.args.max_seq_length * 4)
                else:
                    output = torch.cat((output, self.generate(item[:self.args.max_seq_length].unsqueeze(dim=0).to(self.device), max_length = self.args.max_seq_length * 4)))
            loss = loss_type(torch.tensor(dest_data, device=self.device)[:, self.args.max_seq_length:].to(dtype=torch.float), output[:, self.args.max_seq_length:].to(dtype=torch.float)) / self.num_embeddings

            # output = self.finetuned_model.generate(inputs = obs_data.to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
            after_time = time.time()
            consume_time = after_time - pre_time
            loss = loss_type(dest_data.to(dtype=torch.float, device=self.device), output[:, self.args.max_seq_length:].to(dtype=torch.float)) / (self.vab_size ** 2)
            log_info = "Loss :{}, Inference time:{} in {}".format(loss, consume_time, data_label)
            print(log_info)
            eval_txt.text_insert(log_info)
    
    def generate(self, input_obs, max_length):
        obs_token = input_obs#self.tokenizer_tool.obs_tokenize([input_obs])
        input_token = obs_token.to(self.device)#torch.concat((torch.tensor([[latent]], dtype = torch.long), obs_token), dim = -1)
        output = self.eval(input_token, input_token, max_length)
        # act_list = self.tokenizer_tool.act_detokenize(output[:, 2::2]).squeeze()
        return output
    
    def generate_env_test(self, latent = 10, fig_show = True, save_csv = False):
        self.load_finetuned_model()
        initial_condition = {}
        initial_condition['player1'] = {
            'position_lat_geod_deg': 41, 'position_long_gc_deg': 38, 'position_h_sl_ft': 10000, 'initial_heading_degree': 0, #10000
            'velocities_v_east_fps': 500, 'velocities_v_north_fps': 500, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }
        env = zj_jsbsim(self.args, BattleTask, self.args.aircraft)
        fig = mainWin()
        cur_obs = env.reset(initial_condition)
        obs_traj = [list(cur_obs['player1'].values())[0 : self.args.obs_dim]]
        act_list = self.generate(latent, list(cur_obs['player1'].values())[0 : self.args.obs_dim])#0
        for _ in range(1000):
            if fig_show:
                fig.draw(np.array(trans_from_zjenv_to_mrad(obs_traj)))
            act = act_list.pop(0)
            # act = list(env.action_space.sample())
            # act = [0, 0, 0, 0]
            act_dict = {'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], act))}
            cur_obs = env.step(act_dict)
            obs_traj.append(list(cur_obs['player1'].values())[0 : self.args.obs_dim])
            if len(act_list) == 0:
                act_list = self.generate(latent, list(cur_obs['player1'].values())[0 : self.args.obs_dim])
        
            if cur_obs['player1']['position_h_sl_ft'] < 10:
                break
        if save_csv:
            write_result_csv_data(trans_from_zjenv_to_csv(obs_traj), self.args.result_data_dir + 'result_traj_' +
                                  self.args.aircraft[0] + '_' + self.args.llm_model + '_latent' + str(latent) +  '.csv')
        
        if fig_show:
            fig.consist_update = False
            fig.draw(np.array(trans_from_zjenv_to_mrad(obs_traj)))