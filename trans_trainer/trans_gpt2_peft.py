import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from peft.peft_model import PeftModel
from datasets import Dataset
from transformers import AutoConfig, GPT2LMHeadModel
from tokenizer.AutoTokenizerForGPT2 import AutoTokenizerForGPT2v1, AutoTokenizerForGPT2v2
from utils.gpu_alloc import use_gpu
from copy import deepcopy
from utils.loadcsv import trans_from_zjenv_to_mrad
import logging
from log.log_process import Logger
from log.json_process import Texter

class TRANSDATA():
    def __init__(self, data_file, max_size, max_seq_length, obs_tokenizer_tool, act_tokenizer_tool, aircraft, llm, vab_size, train_ratio, flag = "train"):
        self.max_size = max_size
        self.max_seq_length = max_seq_length
        self.obs_tokenizer_tool = obs_tokenizer_tool
        self.act_tokenizer_tool = act_tokenizer_tool
        self.aircraft = aircraft
        self.llm = llm
        self.flag = flag
        self.transdata_res = self.load_data(data_file, self.obs_tokenizer_tool, self.act_tokenizer_tool, vab_size, train_ratio)
    
    def load_data(self, data_file, obs_tokenizer_tool, act_tokenizer_tool, vab_size, train_ratio):            
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

        datasets_dict = {}
        for file in data_set:
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

        # sentence1 sentence2 label idx input_ids token_type_ids attention_mask
        datasets_dict['input_ids'] = torch.concat([total_obs_data, total_act_data], dim=-1)#.to(torch.long)
        datasets_dict['labels'] = torch.concat([total_obs_data, total_act_data], dim=-1)#.to(torch.long)
        datasets_dict['attention_mask'] = torch.ones(torch.concat([total_obs_data, total_act_data], dim=-1).shape, dtype=torch.long)
        return Dataset.from_dict(datasets_dict)
    

class TRANSGPT():
    def __init__(self, args) -> None:
        self.args = args
        self.vab_size = 50256
        self.obs_token_num = [0, 0, 0, 36, 36, 36]
        os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.5)
        
        self.logger = Logger("Peft_" + args.aircraft_type + "_" + args.fine_tune_mode + "_log_info.log", os.path.join(os.getcwd(), args.log_dir))

        self.obs_tokenizer_tool = AutoTokenizerForGPT2v1(self.vab_size, obs_token_num=self.obs_token_num)
        self.act_tokenizer_tool = AutoTokenizerForGPT2v2(self.vab_size)

        self.train_data = TRANSDATA(os.path.join(os.getcwd(), self.args.master_data_dir, self.args.sear_version), max_size = self.args.max_dataset_size, max_seq_length = self.args.max_seq_length, 
                               obs_tokenizer_tool = self.obs_tokenizer_tool, act_tokenizer_tool = self.act_tokenizer_tool, flag = "train",
                               aircraft = self.args.aircraft[0], llm = self.args.kine_llm, vab_size = self.vab_size, train_ratio = args.train_ratio).transdata_res

        self.test_data = TRANSDATA(os.path.join(os.getcwd(), self.args.master_data_dir, self.args.sear_version), max_size = self.args.max_dataset_size, max_seq_length = self.args.max_seq_length, 
                               obs_tokenizer_tool = self.obs_tokenizer_tool, act_tokenizer_tool = self.act_tokenizer_tool, flag = "test",
                               aircraft = self.args.aircraft[0], llm = self.args.kine_llm, vab_size = self.vab_size, train_ratio = args.train_ratio).transdata_res

        self.checkpoint = os.path.join(self.args.pretrained_checkpoint, self.args.kine_llm)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = AutoConfig.from_pretrained(self.checkpoint)
        self.model = GPT2LMHeadModel(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # from peft import PromptEncoderConfig, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit, LoraConfig, LoHaConfig, LoKrConfig, AdaLoraConfig, IA3Config, get_peft_model
        if self.args.fine_tune_mode == "p-tuning_mlp":
            from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
            peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=256, encoder_hidden_size=1024, 
                                                encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP)
            self.model = get_peft_model(self.model, peft_config)
            print(self.model.print_trainable_parameters())
        elif self.args.fine_tune_mode == "p-tuning_lstm":
            from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
            peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=256, encoder_hidden_size=1024, 
                                                encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM)
            self.model = get_peft_model(self.model, peft_config)
            print(self.model.print_trainable_parameters())
        elif self.args.fine_tune_mode == "prefix-tuning":
            from peft import PrefixTuningConfig, get_peft_model, TaskType
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=256, prefix_projection=True, encoder_hidden_size=1024)
            self.model = get_peft_model(self.model, peft_config)
            print(self.model.print_trainable_parameters())
        elif self.args.fine_tune_mode == "prompt-tuning":
            from peft import PromptTuningConfig, TaskType, get_peft_model
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=256)
            self.model = get_peft_model(self.model, peft_config)
            print(self.model.print_trainable_parameters())
        elif self.args.fine_tune_mode == "lora":
            from peft import LoraConfig, TaskType, get_peft_model
            config = LoraConfig(r=256, lora_alpha=256, task_type=TaskType.SEQ_2_SEQ_LM, lora_dropout=0.1, bias="lora_only", fan_in_fan_out=True)
            self.model = get_peft_model(self.model, config)
            # print(config)
            print(self.model.print_trainable_parameters())
        elif self.args.fine_tune_mode == "adalora":
            from peft import AdaLoraConfig, get_peft_model, TaskType
            config = AdaLoraConfig(r=256,init_r=256, tinit=512, tfinal=2048, deltaT=10, fan_in_fan_out=True, task_type=TaskType.SEQ_2_SEQ_LM)
            self.model = get_peft_model(self.model, config)
            print(self.model.print_trainable_parameters())
        elif self.args.fine_tune_mode == "ia3": 
            from peft import IA3Config, get_peft_model, TaskType
            peft_config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM)
            self.model = get_peft_model(self.model, peft_config)
            print(self.model.print_trainable_parameters())

        self.logger.log_insert(self.model.print_trainable_parameters(), logging.INFO)
        self.finetuned_model = None
        # self.load_finetuned_model()

    def load_finetuned_model(self):
        # if os.path.exists(os.path.join(self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version + '_' + self.args.fine_tune_mode)):
        #     finetuned_checkpoint = os.path.join(self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version + '_' + self.args.fine_tune_mode)
        #     self.finetuned_model = GPT2LMHeadModel.from_pretrained(finetuned_checkpoint).to(self.device)

        config = AutoConfig.from_pretrained(self.checkpoint)
        base_model = GPT2LMHeadModel(config).to(self.device)
        finetuned_model_path_dir = os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version + '_' + self.args.fine_tune_mode)
        self.finetuned_model = PeftModel.from_pretrained(model=base_model, model_id=os.path.join(finetuned_model_path_dir, 'checkpoint-'+str(self.args.num_epochs)))



    def trainer(self):
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        finetuned_model_path_dir = os.path.join(os.getcwd(), self.args.finetuned_checkpoint, 'kine_llm_eval', self.args.kine_llm, self.args.aircraft[0] + '_token_' + self.args.sear_version + '_' + self.args.fine_tune_mode)
        training_args = Seq2SeqTrainingArguments(finetuned_model_path_dir, max_steps = self.args.num_epochs, per_device_train_batch_size = self.args.batch_size, save_steps = self.args.num_epochs,
                                                 logging_steps = 10, logging_dir=os.path.join(os.getcwd(), self.args.log_dir, self.args.fine_tune_mode))
        peft_trainer = Seq2SeqTrainer(self.model, training_args, train_dataset=self.train_data.with_format("torch"), eval_dataset=self.test_data.with_format("torch"))
        peft_trainer.train()

    def tester(self):        
        self.load_finetuned_model()

        dest_data = self.test_data
        total_acc_num = []
        consume_time = []
        for id, item in enumerate(dest_data['input_ids']):
            input_token = torch.tensor(item[:self.args.max_seq_length]).unsqueeze(dim=0)
            pre_time = time.time()
            output = self.finetuned_model.generate(inputs = input_token.to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
            after_time = time.time()
            consume_time.append(after_time - pre_time)
            acc_num = sum(torch.tensor(dest_data['input_ids'][id][self.args.max_seq_length:], device=self.device).eq(output.squeeze()[self.args.max_seq_length:])) / len(dest_data['input_ids'][id][self.args.max_seq_length:])
            # print(dest_data['input_ids'][id][self.args.max_seq_length:])
            # print(output.squeeze()[self.args.max_seq_length:])
            print(acc_num)
            total_acc_num.append(acc_num)
        print("Accurate rate:{}, Inference time:{}".format(sum(total_acc_num) / len(total_acc_num), sum(consume_time) / len(consume_time)))
        # return sum(total_acc_num) / len(total_acc_num)
    
    def tester_fast(self, data_labels = ['training data', 'test data']):

        eval_txt = Texter("peft_eval.txt", os.path.join(os.getcwd(), self.args.log_dir))
        eval_txt.text_insert("-------------------------------" + self.args.fine_tune_mode + "----------------------------")

        self.load_finetuned_model()
        loss_type = nn.CrossEntropyLoss()

        for data_label in data_labels:
            if data_label == 'training data':
                obs_data = self.train_data['input_ids']
            elif data_label == 'test data':
                obs_data = self.test_data['input_ids']

            dest_data = torch.tensor(obs_data)
            pre_time = time.time()
            output = self.finetuned_model.generate(inputs = dest_data[:, self.args.max_seq_length:].to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
            after_time = time.time()
            consume_time = after_time - pre_time
            loss = loss_type(torch.tensor(dest_data[:, self.args.max_seq_length:], device=self.device).to(dtype=torch.float), output[:, self.args.max_seq_length:].to(dtype=torch.float))
            log_info = "Loss :{}, Inference time:{} in {}".format(loss, consume_time, data_label)
            print(log_info)
            eval_txt.text_insert(log_info)

            # input_token = []
            # imput_tmp = []
            # for id, item in enumerate(dest_data['input_ids']):
            #     if (id + 1) % batch_size == 0:
            #         input_token.append(deepcopy(imput_tmp))
            #         imput_tmp.clear()
            #     imput_tmp.append(item[:self.args.max_seq_length])

            # for input_token_item in input_token:
            #     input_ = torch.tensor(input_token_item)#.reshape([batch_size, -1 , self.args.max_seq_length])
            #     output = self.finetuned_model.generate(inputs = input_.to(self.device), max_length = self.args.max_seq_length * 5, num_return_sequences = 1, pad_token_id = self.vab_size)
            #     # acc_num = torch.sum(torch.tensor(dest_data['input_ids'][id][self.args.max_seq_length:], device=self.device).eq(output[:, self.args.max_seq_length:])) / (output.numel())
            #     loss = loss_type(torch.tensor(dest_data['input_ids'][id][self.args.max_seq_length:], device=self.device).to(dtype=torch.float), output[:, self.args.max_seq_length:].to(dtype=torch.float))
            #     print(loss)