import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import numpy as np
import sys
import random
#from utils import progress_bar
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import copy
import argparse

parser = argparse.ArgumentParser('Train Baseline - KNAPSACK')
parser.add_argument("--num_samples", type=int, default = 1000, help="number of training samples")
args = parser.parse_args()

best_acc = 0

torch.manual_seed(12345)
num_sample = args.num_samples



class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'train'):

        if mode == 'train':
            self.weight_dataset = np.load('../dataset/feature_weight_{}.npy'.format(mode))[:num_sample]
            self.profit_dataset = np.load('../dataset/feature_profit_{}.npy'.format(mode))[:num_sample]
            self.value = np.load('../dataset/label_{}.npy'.format(mode))[:num_sample]
            self._ind_dataset = np.load('../dataset/label_ind_{}.npy'.format(mode))[:num_sample]
            
        else:
            self.weight_dataset = np.load('../dataset/feature_weight_{}.npy'.format(mode))
            self.profit_dataset = np.load('../dataset/feature_profit_{}.npy'.format(mode))
            self.value = np.load('../dataset/label_{}.npy'.format(mode))
            self._ind_dataset = np.load('../dataset/label_ind_{}.npy'.format(mode))
            
        self.ind_dataset = []
        self.r_dataset = []
        for label_ind in self._ind_dataset:
            new_label_ind = []
            r_ind = 5
            empty_r = [0, 0, 0, 0, 0, 0]
            for j, k in enumerate(label_ind):
                if 0 < k < 1:
                    r_ind = j
                if k < 1:
                    new_label_ind.append(0)
                else:
                    new_label_ind.append(1)
            self.ind_dataset.append(new_label_ind)
            empty_r[r_ind] = 1
            self.r_dataset.append(empty_r)
        self.ind_dataset = np.array(self.ind_dataset)
            
    def __len__(self):
        return len(self.weight_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'weight': self.weight_dataset[idx] / 15, 'profit': self.profit_dataset[idx], 'label': self.value[idx], 'limit': np.array([15]), 'ind': self.ind_dataset[idx], 'r': np.array(self.r_dataset[idx])}
        return sample

class FractionalKnapsackMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(10, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU(),
            nn.Linear(64*4*2, 64*4*2),
            nn.ReLU()
        )
        self.pick_head = nn.Linear(64*4*2, 5)   # k_mask (0 or 1 per item)
        self.partial_head = nn.Linear(64*4*2, 6)  # r (one-hot)

    def forward(self, weights, profits):
        x = torch.cat([weights, profits], dim=1)  # shape (B, 10)
        h = self.hidden(x)

        k_mask_logits = self.pick_head(h)
        r_logits = self.partial_head(h)

        k_mask = torch.sigmoid(k_mask_logits)         # continuous ∈ (0,1)
        r_softmax = F.softmax(r_logits, dim=1)        # one-hot style
        return k_mask, r_softmax




def train(model_ori, criterion, criterion2, optim, dataset):
    model_ori.train()
    train_loss = 0

    cnt = 0
    for batch_idx, input_data in enumerate(dataset):
        weights, profits, limits, labels, inds, r_gt = input_data['weight'].to('cuda', dtype=torch.float), input_data['profit'].to('cuda', dtype=torch.float), input_data['limit'].to('cuda', dtype=torch.float), input_data['label'].to('cuda', dtype=torch.float), input_data['ind'].to('cuda', dtype=torch.float), input_data['r'].to('cuda', dtype=torch.float)
        k_logits, r_logits = model_ori(weights, profits)
        loss = criterion(k_logits, inds) + criterion2(r_logits, r_gt)
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        cnt += 1
    print('train loss: ', train_loss / cnt)

def test(model_ori, criterion, optim, dataset, mode = 'train'):
    global best_acc
    model_ori.eval()
    correct1, total1 = 0, 0   

    cnt = 0
    for batch_idx, input_data in enumerate(dataset):
        weights, profits, limits, labels, inds, r_gt = input_data['weight'].to('cuda', dtype=torch.float), input_data['profit'].to('cuda', dtype=torch.float), input_data['limit'].to('cuda', dtype=torch.float), input_data['label'].to('cuda', dtype=torch.float), input_data['ind'].to('cuda', dtype=torch.float), input_data['r'].to('cuda', dtype=torch.float)
        k_logits, r_logits = model_ori(weights, profits)

        for i in range(k_logits.size()[0]):
            pred = torch.round(k_logits[i])
            r_pred = torch.round(r_logits[i])
            res = torch.equal(pred.type(torch.int64), inds[i].type(torch.int64))
            res2 = torch.equal(r_gt[i].type(torch.int64), r_pred.type(torch.int64))
            correct1 += res * res2
            total1 += 1

    print(mode, 'acc: ', correct1/total1)
    if correct1 / total1 > best_acc and mode != 'eval': #train_loss < best_loss:
        print('saving...')
        best_acc = correct1 / total1 #train_loss
        state_dict = {'net': model_ori.state_dict()}
        os.makedirs('baseline_model', exist_ok = True)
        torch.save(state_dict, 'baseline_model/model_fractional_{}.pth'.format(num_sample))


model_ori = FractionalKnapsackMLP()
total_params = sum(p.numel() for p in model_ori.parameters())
print(f"Total parameters: {total_params}")

model_ori = model_ori.to('cuda')

train_dataset = FeatureDataset('train')
train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

test_dataset = FeatureDataset('val') #this is actually validation dataset
test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

opt = torch.optim.Adam(model_ori.parameters(), lr = 0.001) 

criterion = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss()

for i in range(200):
    print('Epoch: ', i)
    train(model_ori,  criterion, criterion2, opt, train_dataloader)
    test(model_ori,  criterion, opt, test_dataloader, 'val')

state_dict = torch.load(
        'baseline_model/model_fractional_{}.pth'.format(num_sample), map_location='cuda'
    )
model_ori.load_state_dict(state_dict['net'])

eval_dataset = FeatureDataset('eval')
eval_dataloader = DataLoader(eval_dataset, batch_size = 128, shuffle = False)
test(model_ori, criterion, opt, eval_dataloader, 'eval')
