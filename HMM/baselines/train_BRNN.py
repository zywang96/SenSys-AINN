import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pickle
import random
import argparse

parser = argparse.ArgumentParser('Train BRNN')
parser.add_argument("--num_samples", type=int, default = 50, help="number of training samples")
args = parser.parse_args()

torch.manual_seed(0)

num_samples = args.num_samples


# === Load Data ===
data0 = pickle.load(open('../dataset/train_fall_dataset_seq.pkl', 'rb'))
label0 = [1] * len(data0)
l0 = len(data0)

data1 = pickle.load(open('../dataset/_train_fall_dataset_seq.pkl', 'rb'))
label1 = [0] * len(data1)
l1 = len(data1)

tmp_data = data0[:num_samples//2] + data1[:num_samples//2] #data0 + data1
data = []
for _d in tmp_data:
    data.append(np.array(_d).flatten())

label = label0[:num_samples//2] + label1[:num_samples//2] #label0 + label1

data_test0 = pickle.load(open('../dataset/val_fall_dataset_seq.pkl', 'rb'))
label_test0 = [1] * len(data_test0)

data_test1 = pickle.load(open('../dataset/_val_fall_dataset_seq.pkl', 'rb'))
label_test1 = [0] * len(data_test1)

tmp_data_test = data_test0 + data_test1
data_test = []
for _d in tmp_data_test:
    data_test.append(np.array(_d).flatten())

label_test = label_test0 + label_test1


data_test_0 = pickle.load(open('../dataset/test_fall_dataset_seq.pkl', 'rb'))
label_test_0 = [1] * len(data_test_0)

data_test_1 = pickle.load(open('../dataset/_test_fall_dataset_seq.pkl', 'rb'))
label_test_1 = [0] * len(data_test_1)

tmp_data_test_ = data_test_0 + data_test_1
data_test_ = []
for _d in tmp_data_test_:
    data_test_.append(np.array(_d).flatten())

label_test_ = label_test_0 + label_test_1


def resample_sequence(seq, target_len=100):
    seq = np.array(seq, dtype=np.float32).squeeze()  # [T]
    T = len(seq)
    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_len)
    resampled = np.interp(new_indices, old_indices, seq)  # [target_len]
    
    return torch.tensor(resampled, dtype=torch.float32).unsqueeze(-1)  # [target_len, 1]


# === Dataset ===
class SensorDataset(Dataset):
    def __init__(self, data, labels, target_len):
        self.data = [resample_sequence(d, target_len) for d in data]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def collate_fn(batch):
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.tensor(labels)

class BRNNFallDetector(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_total: int = 768,
        num_layers: int = 1,
        dense_units: int = 64,
        dropout: float = 0.3,
        batch_first: bool = True,
    ):
        super().__init__()
        assert hidden_total % 2 == 0, "hidden_total must be even for bidirectional."
        hidden_per_dir = hidden_total // 2

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_per_dir,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.bn = nn.BatchNorm1d(hidden_total)
        self.fc1 = nn.Linear(hidden_total, dense_units)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dense_units, 1)  # single logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, h_n = self.gru(x)  
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  
        z = self.bn(h_last)
        z = self.act(self.fc1(z))
        z = self.drop(z)
        return F.sigmoid(self.out(z).squeeze(-1))  



# === Training Setup ===
target_len = 151
input_dim = data[0].shape[1] if len(data[0].shape) > 1 else 1
train_dataset = SensorDataset(data, label, target_len = target_len)
val_dataset = SensorDataset(data_test, label_test, target_len = target_len)
test_dataset = SensorDataset(data_test_, label_test_, target_len = target_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BRNNFallDetector().to(device)
#total = sum(p.numel() for p in model.parameters())
#print('Size: ', total)
l1_lambda = 1.7089e-6
optimizer = torch.optim.Adam(model.parameters(), lr=3.47012e-4, weight_decay=6.3877e-6)
criterion = nn.BCELoss()
best_acc = -1000

# === Training Loop ===
for epoch in range(200):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        l1 = sum(p.abs().sum() for p in model.parameters()) * l1_lambda
        optimizer.zero_grad()
        (loss + l1).backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}")

    # === Validation ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x) #, lengths)
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"           Val Accuracy = {acc:.4f}")
    if acc > best_acc:
        print('saving...')
        best_acc = acc
        state_dict = {'net': model.state_dict()}
        os.makedirs('baseline_model', exist_ok = True)
        torch.save(state_dict, f'baseline_model/baseline_brnn_{num_samples}.pth')

        _correct = 0
        _total = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                predicted = (pred > 0.5).float()
                _correct += (predicted == y).sum().item()
                _total += y.size(0)
        _acc = _correct / _total
        print(f"           Test Accuracy = {_acc:.4f}")

