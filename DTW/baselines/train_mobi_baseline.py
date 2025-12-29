import os
import random
from math import inf
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import argparse
from torchaudio.sox_effects import apply_effects_tensor

SEED = 12345
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class FeatureDataset(torch.utils.data.Dataset):

    def __init__(self, mode, portion = 1.0):
        if mode == 'training':
            self.X = np.load("../dataset/{}_X_{}.npy".format(mode, portion))
            self.y = np.load("../dataset/{}_y_{}.npy".format(mode, portion))
        else:
            self.X = np.load("../dataset/{}_X.npy".format(mode))
            self.y = np.load("../dataset/{}_y.npy".format(mode))
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        sample = {'X': np.asarray(x), 'y': np.asarray(y)}
        return sample

def _collate(batch):
    X = [torch.from_numpy(b['X']).float() for b in batch]
    y = [torch.from_numpy(b['y']) for b in batch]
    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)
    return {'X': X, 'y': y}

@torch.no_grad()
def compute_mean_std_after_resize_32(dataset, batch_size=256, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)

    sum_x  = torch.zeros(1, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(1, dtype=torch.float64, device=device)
    count  = torch.zeros(1, dtype=torch.float64, device=device)

    for batch in loader:
        x = batch['X'].to(device)                # (B, 61, 13)
        if x.dim() == 3:
            x = x.unsqueeze(1)                   # (B, 1, 61, 13)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)  # (B,1,32,32)

        B = x.size(0)
        x = x.reshape(B, 1, -1)                  # (B, 1, P)
        P = x.size(-1)

        xd = x.double()
        sum_x  += xd.sum(dim=(0, 2))             # (1,)
        sum_x2 += (xd * xd).sum(dim=(0, 2))      # (1,)
        count  += torch.as_tensor(B * P, dtype=torch.float64, device=device)

    mean = (sum_x / count).float()               # (1,)
    var  = (sum_x2 / count - (sum_x / count) ** 2).float()  # (1,)
    std  = torch.sqrt(torch.clamp(var, min=0.0))            # (1,)

    return mean.cpu(), std.cpu(), var.cpu()

class DatasetNorm(nn.Module):
    """Standardize with fixed mean/std (per-channel), like Keras Normalization after adapt()."""
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std  = torch.as_tensor(std,  dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-7)

class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=8, mean=(0.0,), std=(1.0,)):
        super().__init__()
        self.resize = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)
        self.norm   = DatasetNorm(mean, std)          # <- precompute from training set

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=True, padding=0)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=True, padding=0)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 14 * 14, 128)
        self.drop2 = nn.Dropout(0.5)
        self.out   = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 1, F, T]
        x = self.resize(x)         # [B,1,32,32]
        x = self.norm(x)           # dataset-standardize (Keras-like)
        x = F.relu(self.conv1(x))  # [B,32,30,30]
        x = F.relu(self.conv2(x))  # [B,64,28,28]
        x = self.pool(x)           # [B,64,14,14]
        x = self.drop1(x)
        x = torch.flatten(x, 1)    # [B, 64*14*14]
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.out(x)




# -------------------- Train / Eval --------------------
def run_sc(
    epochs=200,
    batch_size=64*1,
    lr=1e-3,
    device=None,
    portion = 0.1
):
    best_acc = 0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    tr = FeatureDataset('training', portion)
    va = FeatureDataset('validation')
    te = FeatureDataset('testing')
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2) #, collate_fn=collate)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2) #, collate_fn=collate)


    mean, std, var = compute_mean_std_after_resize_32(tr, batch_size=256, device="cuda")

    model = SimpleAudioCNN(mean = mean, std = std).to(device)
    
    #total = sum(p.numel() for p in model.parameters())
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07, weight_decay=0) #default settings
    
    def eval_loop(loader, proto_source_ds):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for _input in loader:
                x, y = _input['X'], _input['y']
                B = x.size(0)
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 2, 1).unsqueeze(1)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / total if total > 0 else 0.0

    for ep in range(1, epochs + 1):
        model.train()
        runloss = 0.0
        correct, total = 0, 0
        for _input in tqdm(train_loader):
            x, y = _input['X'], _input['y']
            B = x.size(0)
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1).unsqueeze(1)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            runloss += loss.item() * y.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        val_acc = eval_loop(val_loader, va)
        print(f"Epoch {ep}: train_loss={runloss/len(tr):.4f}  val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            print("Saving...")
            best_acc = val_acc
            state_dict = {'net': model.state_dict()}
            os.makedirs('baseline_model', exist_ok = True)
            torch.save(state_dict, f'baseline_model/mobicom_model_{portion:.1f}.pth')
    state_dict = torch.load(
        f'baseline_model/mobicom_model_{portion:.1f}.pth', map_location='cuda'
    )
    model.load_state_dict(state_dict['net'])
    test_acc = eval_loop(test_loader, te)
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train Baseline - DTW')
    parser.add_argument("--portion", type = float, default = 0.1, help = "portion")
    args = parser.parse_args()
    run_sc(portion = args.portion)
