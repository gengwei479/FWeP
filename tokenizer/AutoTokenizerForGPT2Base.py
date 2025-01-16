import numpy as np
import torch

# vocabulary_size in gpt2 is 50256
# observation 6, 6, 6, 6, 6, 6
# action 16, 16, 16, 12


class AutoTokenizerForGPT2Base():
    def __init__(self, obs_token_num, act_token_num, 
                 obs_range, act_range, obs_dim = 6, act_dim = 4) -> None:
        # self.vab_size = 50256
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_range = obs_range
        self.act_range = act_range
        self.obs_token_num = obs_token_num
        self.act_token_num = act_token_num
        
        self.obs_pt = [(self.obs_range[id][1] - self.obs_range[id][0]) / (self.obs_token_num[id]) if (self.obs_token_num[id]) != 0 else 0 for id in range(self.obs_dim)]
        self.act_pt = [(self.act_range[id][1] - self.act_range[id][0]) / (self.act_token_num[id]) if (self.act_token_num[id]) != 0 else 0 for id in range(self.act_dim)]
    
    def obs_tokenize(self, obs_batch):
        if not torch.is_tensor(obs_batch):
            obs_batch = torch.tensor(obs_batch)
        assert obs_batch.shape[-1] == self.obs_dim
        
        obs_batch_res = torch.ones_like(obs_batch, dtype=torch.long)
        for obs_id in range(self.obs_dim):
            if len(obs_batch.shape) == 2:
                if self.obs_pt[obs_id] != 0:
                    obs_batch_res[:, obs_id] = (obs_batch[:, obs_id] - self.obs_range[obs_id][0]) / self.obs_pt[obs_id]
                else:
                    obs_batch_res[:, obs_id] = torch.zeros_like(obs_batch_res[:, obs_id])
            elif len(obs_batch.shape) == 3:
                if self.obs_pt[obs_id] != 0:
                    obs_batch_res[:, :, obs_id] = (obs_batch[:, :, obs_id] - self.obs_range[obs_id][0]) / self.obs_pt[obs_id]
                else:
                    obs_batch_res[:, :, obs_id] = torch.zeros_like(obs_batch_res[:, :, obs_id])
        return obs_batch_res
    
    def obs_detokenize(self, obs_batch):
        if not torch.is_tensor(obs_batch):
            obs_batch = torch.tensor(obs_batch)
        
        obs_batch_res = torch.ones_like(obs_batch, dtype=torch.float)
        for obs_id in range(self.obs_dim):
            if len(obs_batch.shape) == 2:
                obs_batch_res[:, obs_id] = (obs_batch[:, obs_id] + 0.5) * self.obs_pt[obs_id] + self.obs_range[obs_id][0] * (self.obs_token_num[obs_id] != 0)
            elif len(obs_batch.shape) == 3:
                obs_batch_res[:, :, obs_id] = (obs_batch[:, :, obs_id] + 0.5) * self.obs_pt[obs_id] + self.obs_range[obs_id][0] * (self.obs_token_num[obs_id] != 0)
        return obs_batch_res
    
    def act_tokenize(self, act_batch):
        if not torch.is_tensor(act_batch):
            act_batch = torch.tensor(act_batch)
        assert act_batch.shape[-1] == self.act_dim
        
        act_batch_res = torch.ones_like(act_batch, dtype=torch.long)
        for act_id in range(self.act_dim):
            if len(act_batch.shape) == 2:
                if self.act_pt[act_id] != 0:
                    act_batch_res[:, act_id] = (act_batch[:, act_id] - self.act_range[act_id][0]) / self.act_pt[act_id]
                else:
                    act_batch_res[:, act_id] = torch.zeros_like(act_batch_res[:, act_id])
            elif len(act_batch.shape) == 3:
                if self.act_pt[act_id] != 0:
                    act_batch_res[:, :, act_id] = (act_batch[:, :, act_id] - self.act_range[act_id][0]) / self.act_pt[act_id]
                else:
                    act_batch_res[:, :, act_id] = torch.zeros_like(act_batch_res[:, :, act_id])
        return act_batch_res
    
    def act_detokenize(self, act_batch):
        if not torch.is_tensor(act_batch):
            act_batch = torch.tensor(act_batch)
        
        act_batch_res = torch.ones_like(act_batch, dtype=torch.float)
        for act_id in range(self.act_dim):
            if len(act_batch.shape) == 2:
                act_batch_res[:, act_id] = (act_batch[:, act_id] + 0.5) * self.act_pt[act_id] + self.act_range[act_id][0] * (self.act_token_num[act_id] != 0)
            elif len(act_batch.shape) == 3:
                act_batch_res[:, :, act_id] = (act_batch[:, :, act_id] + 0.5) * self.act_pt[act_id] + self.act_range[act_id][0] * (self.act_token_num[act_id] != 0)
        return act_batch_res
    