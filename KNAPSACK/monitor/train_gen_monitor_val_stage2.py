import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
import numpy as np
import random
#from utils import progress_bar
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import copy
from monitor_helper import main, run_inference
from contextlib import contextmanager
import argparse
import yaml

parser = argparse.ArgumentParser('Train AINN - KNAPSACK')
parser.add_argument("--config", type=str, default = 'ainn_less_iter', help="config")
args = parser.parse_args()
cfg = yaml.safe_load(open('../config/{}.yaml'.format(args.config)))
num_sample = cfg['data']['num_samples']

best_acc = 0
torch.manual_seed(0)

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

        gate_idx = torch.argmax(self.shared_gate_logits).view(1, 1)  # (1, 1)
        gate_idx = gate_idx.expand(batch_size, 1)  # (B, 1)

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
        
        if mode == 'train':
            self.weight_dataset = np.load('../dataset/feature_weight_{}.npy'.format(mode))[:]
            self.profit_dataset = np.load('../dataset/feature_profit_{}.npy'.format(mode))[:]
            self.value = np.load('../dataset/label_{}.npy'.format(mode))[:]
            self.ind_dataset = np.load('../dataset/label_ind_{}.npy'.format(mode))[:]
        else:
            self.weight_dataset = np.load('../dataset/feature_weight_{}.npy'.format(mode)) #[2000:12000]
            self.profit_dataset = np.load('../dataset/feature_profit_{}.npy'.format(mode)) #[2000:12000]
            self.value = np.load('../dataset/label_{}.npy'.format(mode)) #[2000:12000]
            self.ind_dataset = np.load('../dataset/label_ind_{}.npy'.format(mode)) #[2000:12000]
        
    def __len__(self):
        return len(self.weight_dataset)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'weight': self.weight_dataset[idx], 'profit': self.profit_dataset[idx], 'label': self.value[idx], 'limit': np.array([15]), 'ind': self.ind_dataset[idx]}
        
        return sample
        
        


def compute_acc(pred, labels):
    pred = torch.round(pred).long()
    c = 0
    t = 0
    for i in range(pred.size()[0]):
        res = torch.equal(pred[i].type(torch.int64), labels[i].type(torch.int64))
        c += res
        t += 1
    return c, t


        

def fractional_knapsack(weights, values, sorted_indices, capacity = 15):
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


def gen(model_ori, train_model, dataset, rng_key, mode = 'train'):
    model_ori.eval()
    #model_ae.train()
    correct1, total1 = 0, 0
    correct2, total2 = 0, 0
    correct3, total3 = 0, 0
    train_loss = 0
    train_loss2 = 0
    p_list = []
    gt_list = []
    w_list = []
    v_list = []
    inf_list = []
    pinf_list = []

    _p_list = []
    _gt_list = []
    _w_list = []
    _v_list = []
    _inf_list = []
    _pinf_list = []

    cnt = 0
    for batch_idx, input_data in enumerate(dataset):
        weights, profits, limits, labels, inds = input_data['weight'].to('cuda', dtype=torch.float), input_data['profit'].to('cuda', dtype=torch.float), input_data['limit'].to('cuda', dtype=torch.float), input_data['label'].to('cuda', dtype=torch.float), input_data['ind'].to('cuda', dtype=torch.float)
        z = model_ori(torch.cat((weights, profits), -1))
        gt_ratio = profits/weights
        new_target = torch.tensor(np.zeros((weights.size()[0], 1)))
        for w, v, p, gt in zip(weights.cpu().detach().numpy().tolist(), profits.cpu().detach().numpy().tolist(), z.cpu().detach().numpy().tolist(), gt_ratio.cpu().detach().numpy().tolist()):
            ret_ind, preds_sorting = run_inference(train_model, np.array(p)[None, :], rng_key)
            ind_list_gt, value_gt = fractional_knapsack(w, v, np.argsort(gt)[::-1])
            ind_list_pred, value_pred = fractional_knapsack(w, v, ret_ind[0])
            if np.array_equal(np.argsort(p)[::-1], ret_ind[0]): 
                p_list.append(p) #.cpu().detach().numpy())
                gt_list.append(gt) #.cpu().detach().numpy())
                w_list.append(w)
                v_list.append(v)
                inf_list.append(preds_sorting['pred'].data[0])
                pinf_list.append(ret_ind[0])
                correct1 += 1
            else:
                _p_list.append(p)
                _gt_list.append(gt)
                _w_list.append(w)
                _v_list.append(v)
                _inf_list.append(preds_sorting['pred'].data[0])
                _pinf_list.append(ret_ind[0])
                
            cnt += 1
        
    os.makedirs('intermediate', exist_ok = True)
    #return
    with open('intermediate/_pinf_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(_pinf_list))

    with open('intermediate/_weight_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(_w_list))
    with open('intermediate/_profit_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(_v_list))
    with open('intermediate/_pred_ratio_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(_p_list))
    with open('intermediate/_gt_ratio_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(_gt_list))
    with open('intermediate/_inf_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(_inf_list))


    with open('intermediate/weight_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(w_list))
    with open('intermediate/profit_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(v_list))
    with open('intermediate/pred_ratio_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(p_list))
    with open('intermediate/gt_ratio_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(gt_list))
    with open('intermediate/inf_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(inf_list))
    with open('intermediate/pinf_list5_log_exp_{}_stage2.npy'.format(mode), 'wb') as f:
        np.save(f, np.array(pinf_list))


model_ori = SimpleNN()
model_ori = model_ori.to('cuda')
state_dict = torch.load(
        '../ainn_model/model_{}_stage1_{}.pth'.format(args.config, num_sample), map_location='cuda'
    )
model_ori.load_state_dict(state_dict['net'])

#with chdir(PARENT_DIR):
train_model, feedback, rng_key = main()
train_model.restore_model('model_{}_stage2_{}.pkl'.format(args.config, num_sample))

train_dataset = FeatureDataset('train')
train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = False)

val_dataset = FeatureDataset('val')
val_dataloader = DataLoader(val_dataset, batch_size = 128, shuffle = False)

test_dataset = FeatureDataset('eval')
test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

print('constructing training part...')
gen(model_ori, train_model, train_dataloader, rng_key, 'train')
print('constructing validation part...')
gen(model_ori, train_model, val_dataloader, rng_key, 'val')
print('constructing testing part...')
gen(model_ori, train_model, test_dataloader, rng_key, 'eval')
