import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, Qwen2ForCausalLM
from tokenizer.AutoTokenizerForGPT2 import AutoTokenizerForGPT2v1, AutoTokenizerForGPT2v2
from tqdm.auto import tqdm
from copy import deepcopy
from utils.gpu_alloc import use_gpu
from log.json_process import Texter

from envs.env_wrapper import jsbsimSingleENV
from visual.plot_utils import mainWin
from utils.loadcsv import trans_from_zjenv_to_mrad, trans_from_zjenv_to_csv, write_result_csv_data

class TRANSDATA(Dataset):
    def __init__(self, data_file, max_size, max_seq_length, obs_tokenizer_tool, act_tokenizer_tool, aircraft, llm, vab_size, train_ratio, flag = "train"):
        self.max_size = max_size
        self.max_seq_length = max_seq_length
        self.obs_tokenizer_tool = obs_tokenizer_tool
        self.act_tokenizer_tool = act_tokenizer_tool
        self.aircraft = aircraft
        self.llm = llm
        self.flag = flag
        self.obs_data, self.act_data, self.total_obs_withouttoken_mrad, self.total_act_withouttoken_mrad = self.load_data(data_file, vab_size, train_ratio)
        self.attention_mask = torch.ones(torch.concat([self.obs_data, self.act_data], dim=-1).shape, dtype=torch.long)
    
    def load_data(self, data_file, vab_size, train_ratio):            
        dirs = os.listdir(data_file)
        total_obs_data = torch.tensor([], dtype=torch.long)
        total_act_data = torch.tensor([], dtype=torch.long)
        total_obs_withouttoken_mrad = []
        total_act_withouttoken_mrad = []

        file_list = [item for item in dirs if '_' + self.aircraft + '_traj.npy' in item]
        if self.flag == "train":
            data_set = file_list[:int(len(file_list) * train_ratio)]
        elif self.flag == "test":
            data_set = file_list[int(len(file_list) * train_ratio):]
        # print(len(data_set))

        for file in data_set:
            if '_' + self.aircraft + '_traj.npy' in file:
                trajectory = np.load(os.path.join(data_file, file), allow_pickle=True)
                
                obs_data, act_data = [], []
                for id, item in enumerate([0]):#enumerate(trajectory[:-self.max_seq_length]):                    
                    obs_data = trans_from_zjenv_to_mrad([trajectory[id + it]['cur_observation'] for it in range(self.max_seq_length)])
                    act_data = [trajectory[id + it]['action'] for it in range(self.max_seq_length)]

                    total_obs_withouttoken_mrad.append(deepcopy(obs_data))
                    total_act_withouttoken_mrad.append(deepcopy(act_data))
                    obs_data = self.obs_tokenizer_tool.obs_tokenize(obs_data - obs_data[0]).transpose(0, 1)    
                    act_data = self.act_tokenizer_tool.act_tokenize(act_data).reshape(1, -1)
                    
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

class TRANSQWEN2():
    def __init__(self, args) -> None:
        self.args = args
        self.vab_size = 151936
        # self.obs_token_num = [5, 5, 4, 6, 6, 6]
        self.obs_token_num = [0, 0, 0, 53, 53, 53]
        self.act_token_num = [20, 20, 20, 18]
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.5)
        
        # if self.args.token_version == 1:
        #     self.tokenizer_tool = AutoTokenizerForGPT2v1(self.vab_size, self.obs_token_num, self.act_token_num)
        # elif self.args.token_version == 2:
        #     self.tokenizer_tool = AutoTokenizerForGPT2v2()

        self.obs_tokenizer_tool = AutoTokenizerForGPT2v1(self.vab_size, obs_token_num=self.obs_token_num)
        self.act_tokenizer_tool = AutoTokenizerForGPT2v2(self.vab_size)
        
        self.train_data = TRANSDATA(os.path.join(self.args.master_data_dir, self.args.sear_version), max_size = self.args.max_dataset_size, max_seq_length = self.args.max_seq_length, 
                               obs_tokenizer_tool = self.obs_tokenizer_tool, act_tokenizer_tool = self.act_tokenizer_tool,
                               aircraft = self.args.aircraft[0], llm = self.args.kine_llm, vab_size = self.vab_size, train_ratio = args.train_ratio, flag = "train")
        self.test_data = TRANSDATA(os.path.join(self.args.master_data_dir, self.args.sear_version), max_size = self.args.max_dataset_size, max_seq_length = self.args.max_seq_length, 
                               obs_tokenizer_tool = self.obs_tokenizer_tool, act_tokenizer_tool = self.act_tokenizer_tool,
                               aircraft = self.args.aircraft[0], llm = self.args.kine_llm, vab_size = self.vab_size, train_ratio = args.train_ratio, flag = "test")
        self.dataloader = DataLoader(self.train_data, batch_size = self.args.batch_size, shuffle=True)
        self.dataloader_test = DataLoader(self.test_data, batch_size = self.args.batch_size, shuffle=True)

        self.checkpoint = os.path.join(self.args.pretrained_checkpoint, self.args.kine_llm)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = AutoConfig.from_pretrained(self.checkpoint)
        self.model = Qwen2ForCausalLM(config).to(self.device)
        # self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        
        self.finetuned_model = None

        self.load_finetuned_model()

    def load_finetuned_model(self):
        if os.path.exists(os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)):
            finetuned_checkpoint = os.path.join(self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)
            self.finetuned_model = Qwen2ForCausalLM.from_pretrained(finetuned_checkpoint).to(self.device)

    def trainer(self):
        print('-----{} trains start!!-----'.format(self.args.kine_llm))
        loss_values = []
        time_list = []
        time_list.append(time.time())
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                obs_seq, act_seq, attention_seq = batch
                attention_seq = attention_seq.to(self.device)
                
                input_ids = torch.concat([obs_seq, act_seq], dim=-1).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_seq, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item()

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.dataloader)
            loss_values.append(avg_loss)
            time_list.append(time.time())

            if (epoch+1) % 10 == 0:
                avg_loss = total_loss / len(self.dataloader)
                perplexity = math.exp(avg_loss)
                print(f'Epoch [{epoch+1}/{self.args.num_epochs}], Loss: {avg_loss:.7f}, Perplexity: {perplexity:.5f}')

        np.save(os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, 'loss_function', self.args.aircraft_type + 'loss_list_' + self.args.sear_version + '.npy'), np.array(loss_values))
        np.save(os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, 'loss_function', self.args.aircraft_type + 'time_list_' + self.args.sear_version + '.npy'), np.array(time_list))
        finetuned_model_path_dir = os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)
        os.path.join(os.getcwd(), self.args.finetuned_checkpoint, self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version)
        if not os.path.exists(finetuned_model_path_dir):
            os.makedirs(finetuned_model_path_dir)
        self.model.save_pretrained(finetuned_model_path_dir)
    
    def generate(self, input_obs):
        input_token = input_obs.unsqueeze(dim=0).to(self.device)
        output = self.finetuned_model.generate(input_token, max_length = self.args.max_seq_length, num_return_sequences = 1, pad_token_id = self.vab_size)
        act_list = self.tokenizer_tool.act_detokenize(output).squeeze()
        return act_list.tolist()

    def tester(self, data_labels = ['training data', 'test data']):        
        self.load_finetuned_model()
        log_info_res = []
        for data_label in data_labels:
            if data_label == 'training data':
                obs_data = self.train_data.obs_data
            elif data_label == 'test data':
                obs_data = self.test_data.obs_data
            total_acc_num = []
            consume_time = []
            for id, item in enumerate(obs_data):
                input_token = item.unsqueeze(dim=0)
                pre_time = time.time()
                output = self.finetuned_model.generate(input_token.to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
                after_time = time.time()
                consume_time.append(after_time - pre_time)
                if data_label == 'training data':
                    acc_num = sum(self.train_data.act_data[id].to(self.device).eq(output.squeeze()[self.args.max_seq_length:])) / len(self.train_data.act_data[id])
                elif data_label == 'test data':
                    acc_num = sum(self.test_data.act_data[id].to(self.device).eq(output.squeeze()[self.args.max_seq_length:])) / len(self.test_data.act_data[id])
                total_acc_num.append(acc_num)
            log_info = "Accurate rate:{}, Inference time:{} in {}\n".format(sum(total_acc_num) / len(total_acc_num), sum(consume_time) / len(consume_time), data_label)
            print(log_info)
            log_info_res.append(log_info)
        return log_info_res

    def tester_fast(self, data_labels = ['test data']):#'training data', 

        eval_txt = Texter("peft_eval.txt", os.path.join(os.getcwd(), self.args.log_dir))
        eval_txt.text_insert("\n-------------------------------" + self.args.fine_tune_mode + "-----" + self.args.aircraft_type + "----------------------------\n")

        self.load_finetuned_model()
        loss_type = nn.MSELoss()

        for data_label in data_labels:
            if data_label == 'training data':
                obs_data = self.train_data.obs_data
                dest_data = self.train_data.act_data
            elif data_label == 'test data':
                obs_data = self.test_data.obs_data
                dest_data = self.test_data.act_data

            dest_data = torch.tensor(dest_data)
            pre_time = time.time()
            output = self.finetuned_model.generate(inputs = obs_data.to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
            after_time = time.time()
            consume_time = after_time - pre_time
            loss = loss_type(dest_data.to(dtype=torch.float, device=self.device), output[:, self.args.max_seq_length:].to(dtype=torch.float)) / (self.vab_size ** 2)
            log_info = "Loss :{}, Inference time:{} in {}".format(loss, consume_time, data_label)
            print(log_info)
            eval_txt.text_insert(log_info)