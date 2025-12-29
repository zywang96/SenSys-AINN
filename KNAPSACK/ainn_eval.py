import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
import sys
import random
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import copy
from stage_helper.utils import main, run_inference
import argparse
import yaml

parser = argparse.ArgumentParser('Inference AINN - KNAPSACK')
parser.add_argument("--config", type=str, default = 'ainn', help="config")
args = parser.parse_args()

cfg = yaml.safe_load(open('config/{}.yaml'.format(args.config)))

num_sample = cfg['data']['num_samples']


torch.manual_seed(100)


def build_fractional_mask(k_mask, r):
    #print(k_mask)
    B, L = k_mask.shape

    one_index = torch.argmax(k_mask, dim=1)  # shape [B]

    mask = torch.arange(L).expand(B, L)
    full_mask = (mask <= one_index.unsqueeze(1)).float()

    return full_mask

class FractionalKnapsackTwoStage(nn.Module):
    def __init__(self, d_model=32*4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=1),
            num_layers=2
        )
        self.k_head = nn.Linear(d_model, 1)   # [B, L] - logits for K
        self.r_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L = x.shape
        x_input = x.unsqueeze(-1)  #[Batch, L, 1]
        h = self.embed(x_input)    #[Batch, L, D]
        h = h.permute(1, 0, 2)     #[L, Batch, D]
        h = self.encoder(h)        #[L, Batch, D]
        h = F.relu(h.permute(1, 0, 2))     #[Batch, L, D]

        k_logits = self.k_head(h).squeeze(-1)       #[B, L]
        k_idx = torch.argmax(k_logits, dim=1)       #[B]
        k_mask = F.one_hot(k_idx, num_classes=L).float()  #[B, L]

        # r prediction from mean pooled sequence
        h_pooled = h.mean(dim=1)                    # [B, D]
        r = self.r_head(h_pooled).squeeze(-1)       # [B]

        return k_mask, k_logits, r


class SimpleNN(nn.Module):
    def __init__(self, input_size = 10, hidden_size = 16*4, output_size = 5):
        super(SimpleNN, self).__init__()
        assert input_size == 10 and output_size == 5

        self.transforms = [
            lambda x: x,
            torch.log,
            torch.exp,
            torch.sin
        ]
        self.num_transforms = len(self.transforms)

        self.shared_gate_logits = nn.Parameter(torch.randn(self.num_transforms))

        self.shared_mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []

        gate_idx = torch.argmax(self.shared_gate_logits).view(1, 1)  # shape (1, 1)
        gate_idx = gate_idx.expand(batch_size, 1)  # shape (B, 1)

        for i in range(5):
            wi = x[:, i].unsqueeze(1)       # (B, 1)
            pi = x[:, i + 5].unsqueeze(1)   # (B, 1)

            wi_trans = torch.cat([tf(wi) for tf in self.transforms], dim=1)
            pi_trans = torch.cat([tf(pi) for tf in self.transforms], dim=1)

            wi_selected = torch.gather(wi_trans, dim=1, index=gate_idx)  # (B, 1)
            pi_selected = torch.gather(pi_trans, dim=1, index=gate_idx)  # (B, 1)

            inp = torch.cat([wi_selected, pi_selected], dim=1)  # (B, 2)
            ri = self.shared_mlp(inp)  # (B, 1)
            outputs.append(ri)

        return torch.cat(outputs, dim=1)  # (B, 5)




class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'train'):
        self.weight_dataset = np.load('dataset/feature_weight_{}.npy'.format(mode))[:]
        self.profit_dataset = np.load('dataset/feature_profit_{}.npy'.format(mode))[:]
        self.value = np.load('dataset/label_{}.npy'.format(mode))[:]
        self.ind_dataset = np.load('dataset/label_ind_{}.npy'.format(mode))[:]

    def __len__(self):
        return len(self.weight_dataset)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'weight': self.weight_dataset[idx], 'profit': self.profit_dataset[idx], 'label': self.value[idx], 'limit': np.array([15]), 'ind': self.ind_dataset[idx]}
        return sample
        
        




def fractional_knapsack(weights, values, sorted_indices, capacity = 15):
    # Compute value-to-weight ratio
    #value_weight_ratio = values / weights
    
    # Sort items by value-to-weight ratio in descending order using argsort
    #sorted_indices = np.argsort(value_weight_ratio)[::-1]
    #print(sorted_indices)
    total_value = 0.0
    remaining_capacity = capacity
    ind_list = [0, 0, 0, 0, 0]

    # Loop through items based on the sorted indices
    for i in sorted_indices:
        if remaining_capacity > 0:
            # If the item's weight is less than the remaining capacity, take the whole item
            if weights[i] <= remaining_capacity:
                total_value += values[i]
                remaining_capacity -= weights[i]
                ind_list[i] = 1
            # Otherwise, take a fraction of the item
            else:
                total_value += values[i] * (remaining_capacity / weights[i])
                ind_list[i] = remaining_capacity / weights[i]
                remaining_capacity = 0  # Knapsack is full

    return ind_list, total_value


def eval(model_ori, train_model, model2, dataset, rng_key):
    model_ori.eval()
    model2.eval()
    correct1, total1 = 0, 0
    cnt = 0
    for batch_idx, input_data in enumerate(dataset):
        weights, profits, limits, labels, inds = input_data['weight'].to('cuda', dtype=torch.float), input_data['profit'].to('cuda', dtype=torch.float), input_data['limit'].to('cuda', dtype=torch.float), input_data['label'].to('cuda', dtype=torch.float), input_data['ind'].to('cuda', dtype=torch.float)
        z = model_ori(torch.cat((weights, profits), -1))
        gt_ratio = profits/weights
        new_target = torch.tensor(np.zeros((weights.size()[0], 1)))
        for w, v, p, gt, l in zip(weights.cpu().detach().numpy().tolist(), profits.cpu().detach().numpy().tolist(), z.cpu().detach().numpy().tolist(), gt_ratio.cpu().detach().numpy().tolist(), inds.cpu().detach().numpy().tolist()):
            ret_ind, _ = run_inference(train_model, np.array(p)[None, :], rng_key)

            ret_ind = torch.tensor(ret_ind)

            reordered = torch.gather(torch.tensor(w)[None, :] / 15, dim=1, index=ret_ind)

            cumsum_order = torch.cumsum(reordered, dim = 1)

            k_mask, k_logits, _ = model2(cumsum_order.to('cuda'))
            k_mask = k_mask.cpu()
            B, L = k_mask.shape
            k1_index = torch.argmax(k_mask, dim = 1)
            kp1_index = k1_index + 1
            valid_k1 = kp1_index < L

            batch_indices = torch.arange(B, device=k_mask.device)


            cumsum_k1 = cumsum_order[batch_indices[valid_k1], k1_index[valid_k1]]
            reordered_k1 = reordered[batch_indices[valid_k1], kp1_index[valid_k1]]
            r = torch.zeros(B, device=k_mask.device)

            final_value = build_fractional_mask(k_mask, r)

            over_mask = cumsum_k1 > 1.0
            under_mask = ~over_mask

            r_over = 1.0 / cumsum_k1[over_mask]
            r[valid_k1.clone()] = 0  # clear all first to avoid leftovers
            r[batch_indices[valid_k1][over_mask]] = r_over
            final_value[batch_indices[valid_k1][over_mask], k1_index[valid_k1][over_mask]] = r_over

            r_under = (1.0 - cumsum_k1[under_mask]) / reordered_k1[under_mask]
            r[batch_indices[valid_k1][under_mask]] = r_under
            final_value[batch_indices[valid_k1][under_mask], kp1_index[valid_k1][under_mask]] = r_under

            inverse_indices = torch.zeros_like(ret_ind)
            inverse_indices.scatter_(1, ret_ind, torch.arange(ret_ind.size(1)).expand_as(ret_ind))

            final_value2 = torch.gather(final_value, dim=1, index=inverse_indices).cpu().detach().numpy()[0]

            if np.allclose(final_value2, l, rtol=1e-5, atol=1e-8):
                correct1 += 1
            cnt += 1
    print('Test acc: ', correct1 / cnt)
    return


num_sample = cfg['data']['num_samples']
config = args.config

model_ori = SimpleNN()
model_ori = model_ori.to('cuda')
state_dict = torch.load(
        'ainn_model/model_{}_stage1_{}.pth'.format(config, num_sample), map_location='cuda'
    )
model_ori.load_state_dict(state_dict['net'])


model2 = FractionalKnapsackTwoStage()
model2 = model2.to('cuda')
state_dict = torch.load(
        'ainn_model/model_{}_stage3_{}.pth'.format(config, num_sample), map_location='cuda'
    )
model2.load_state_dict(state_dict['net'])

train_model, feedback, rng_key = main()

train_model.restore_model('model_{}_stage2_{}.pkl'.format(config, num_sample))


test_dataset = FeatureDataset('eval')
test_dataloader = DataLoader(test_dataset, batch_size = 64*2, shuffle = False)

eval(model_ori, train_model, model2, test_dataloader, rng_key)
