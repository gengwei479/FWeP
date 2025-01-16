import math
import numpy as np
import torch

# from AutoTokenizerForGPT2Base import AutoTokenizerForGPT2Base
from tokenizer.AutoTokenizerForGPT2Base import AutoTokenizerForGPT2Base


class AutoTokenizerConfig():
    def __init__(self, vab_size = 50256, obs_token_num = [6, 6, 6, 6, 6, 6], act_token_num = [16, 16, 16, 12]) -> None:
        PI = math.pi + 0.001
        self.vab_size = vab_size
        self.obs_dim = 6
        self.act_dim = 4
        self.obs_token_num = obs_token_num
        self.act_token_num = act_token_num
        # self.obs_range = [[-5000, 5000], [-5000, 5000], [-500, 500], [0, 2 * PI], [0, 2 * PI], [0, 2 * PI]]
        # self.obs_range = [[-90, 90], [-90, 90], [0, 20000], [-PI, PI], [-PI, PI], [0, 360]]
        self.obs_range = [[-4000.001, 4000.001], [-4000.001, 4000.001], [-4000.001, 4000.001], [-PI, PI], [-PI, PI], [-PI, PI]]
        self.act_range = [[-1.001, 1.001], [-1.001, 1.001], [-1.001, 1.001], [0, 1.001]]

class AutoTokenizerForGPT2v1():
    def __init__(self, vab_size = 50256, obs_token_num = [6, 6, 6, 6, 6, 6], act_token_num = [16, 16, 16, 12]) -> None:
        self.config = AutoTokenizerConfig(vab_size, obs_token_num, act_token_num)
        
        self.obs_group_size = len(self.config.obs_token_num)
        self.act_group_size = len(self.config.act_token_num)
        prod_lambda_fun = lambda n, input_list: 1 if n == len(input_list) - 1 else prod_lambda_fun(n + 1, input_list) * max(input_list[n + 1], 1)        
        self.obs_token_axis = torch.tensor([[prod_lambda_fun(i, self.config.obs_token_num) * (self.config.obs_token_num[i] != 0) for i in range(len(self.config.obs_token_num))]]).transpose(0, 1)
        self.act_token_axis = torch.tensor([[prod_lambda_fun(i, self.config.act_token_num) * (self.config.act_token_num[i] != 0) for i in range(len(self.config.act_token_num))]]).transpose(0, 1)
        self.token_base = AutoTokenizerForGPT2Base(obs_token_num = self.config.obs_token_num, 
                                                   act_token_num = self.config.act_token_num,
                                                   obs_range = self.config.obs_range, act_range = self.config.act_range,
                                                   obs_dim = self.config.obs_dim, act_dim = self.config.act_dim)
        pass

    # input_dim : N * 6obs_dim
    # output_sim : N * obs_dim
    def obs_tokenize(self, obs_batch):
        if not torch.is_tensor(obs_batch):
            obs_batch = torch.tensor(obs_batch)
        #################
        input_dim = obs_batch.shape
        # if int(input_dim[-1] / self.obs_group_size) == 0:
        #     print(input_dim)
        obs_batch_reshape = obs_batch.reshape((-1, int(input_dim[-1] / self.obs_group_size), self.obs_group_size))
        obs_batch_res = self.token_base.obs_tokenize(obs_batch_reshape)
        return torch.matmul(obs_batch_res, self.obs_token_axis).squeeze().reshape((input_dim[0], -1))#input_dim[1]
    
    # input_sim : N * obs_dim
    # output_dim : N * 6obs_dim
    def obs_detokenize(self, obs_batch):
        if not torch.is_tensor(obs_batch):
            obs_batch = torch.tensor(obs_batch)
        
        obs_batch_res = obs_batch.unsqueeze(dim = -1).repeat(1, 1, self.config.obs_dim)
        
        primary_part = torch.zeros_like(obs_batch_res[:, :, 0], dtype = torch.float)
        for id in range(self.obs_token_axis.shape[0]):
            if self.obs_token_axis[id, 0] != 0:
                obs_batch_res[:, :, id] = (obs_batch_res[:, :, id] - primary_part) / self.obs_token_axis[id, 0]
            else:
                obs_batch_res[:, :, id] = torch.zeros_like(obs_batch_res[:, :, id])
            primary_part += obs_batch_res[:, :, id] * self.obs_token_axis[id, 0]
        return self.token_base.obs_detokenize(obs_batch_res)
    
    def act_tokenize(self, act_batch):
        if not torch.is_tensor(act_batch):
            act_batch = torch.tensor(act_batch)
        
        input_dim = act_batch.shape
        act_batch_reshape = act_batch.reshape((-1, int(input_dim[-1] / self.act_group_size), self.act_group_size))
        act_batch_res = self.token_base.act_tokenize(act_batch_reshape)
        return torch.matmul(act_batch_res, self.act_token_axis).squeeze().reshape((input_dim[0], -1))#input_dim[1]

    def act_detokenize(self, act_batch):
        if not torch.is_tensor(act_batch):
            act_batch = torch.tensor(act_batch)
        
        act_batch_res = act_batch.unsqueeze(dim = -1).repeat(1, 1, self.config.act_dim)
        
        primary_part = torch.zeros_like(act_batch_res[:, :, 0], dtype = torch.float)
        for id in range(self.act_token_axis.shape[0]):
            if self.act_token_axis[id, 0] != 0:
                act_batch_res[:, :, id] = (act_batch_res[:, :, id] - primary_part) / self.act_token_axis[id, 0]
            else:
                act_batch_res[:, :, id] = torch.zeros_like(act_batch_res[:, :, id])
            primary_part += act_batch_res[:, :, id] * self.act_token_axis[id, 0]
        return self.token_base.act_detokenize(act_batch_res)

class AutoTokenizerForGPT2v2():
    def __init__(self, vab_size = 50256) -> None:
        self.config = AutoTokenizerConfig(vab_size)
        self.config.obs_token_num = [self.config.vab_size - 1] * self.config.obs_dim
        self.config.act_token_num = [self.config.vab_size - 1] * self.config.act_dim
        self.token_base = AutoTokenizerForGPT2Base(obs_token_num = self.config.obs_token_num, 
                                            act_token_num = self.config.act_token_num,
                                            obs_range = self.config.obs_range, act_range = self.config.act_range,
                                            obs_dim = self.config.obs_dim, act_dim = self.config.act_dim)
    # input_sim : N * 6obs_dim
    # output_dim : N * 6obs_dim
    def obs_tokenize(self, obs_batch):
        return self.token_base.obs_tokenize(obs_batch)
    
    # input_sim : N * 6obs_dim
    # output_dim : N * 6obs_dim
    def obs_detokenize(self, obs_batch):
        return self.token_base.obs_detokenize(obs_batch)
    
    def act_tokenize(self, act_batch):
        return self.token_base.act_tokenize(act_batch)
    
    def act_detokenize(self, act_batch):
        return self.token_base.act_detokenize(act_batch)
    
# #input_dim: batch_num * seq_num * obs_dim(act_dim)
# #output_dim: batch_num * seq_num

# testtokenclass = AutoTokenizerForGPT2v1()
# # test_obs = torch.randint(low=0, high=100, size=[2,12])
# test_obs = torch.rand(size=[6, 12,4])
# # test_obs = torch.tensor([[0.6857, 0.3871, 0.6894, 0.1127], [0.6857, 0.3871, 0.6894, 0.1127]])
# # test_obs = torch.tensor([3000, 3140, 222, 0.1127, 1.1127, 4])

# print(test_obs)
# test_obs_token = testtokenclass.act_tokenize(test_obs)
# print(test_obs_token)
# # print(testtokenclass.act_detokenize(test_obs_token))