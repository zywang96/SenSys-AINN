import random
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

def keep_leftmost_one(k: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(k, dim=1)
    mask_first_one = (cumsum == 1) & (k == 1)
    return mask_first_one.to(k.dtype)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

pos_output = np.load("intermediate/c_list5_log_exp_train_stage3.npy")
neg_output = np.load("intermediate/_c_list5_log_exp_train_stage3.npy")


pos = np.load("intermediate/k_list5_log_exp_train_stage3.npy") 
neg = np.load("intermediate/_k_list5_log_exp_train_stage3.npy")

pos_extra = np.load("intermediate/kl_list5_log_exp_train_stage3.npy")
neg_extra = np.load("intermediate/_kl_list5_log_exp_train_stage3.npy")

assert pos.shape[0] == pos_extra.shape[0]
assert neg.shape[0] == neg_extra.shape[0]

pos_flat = pos.reshape(len(pos), -1)
neg_flat = neg.reshape(len(neg), -1)

pos_combined = np.concatenate([pos_output > 1, pos, pos_extra], axis=1)
neg_combined = np.concatenate([neg_output > 1, neg, neg_extra], axis=1)

X = np.concatenate([pos_combined, neg_combined], axis=0)
y = np.array([1]*len(pos_combined) + [0]*len(neg_combined))

pos_output_val = np.load("intermediate/c_list5_log_exp_val_stage3.npy")
neg_output_val = np.load("intermediate/_c_list5_log_exp_val_stage3.npy")

pos_val = np.load("intermediate/k_list5_log_exp_val_stage3.npy")
neg_val = np.load("intermediate/_k_list5_log_exp_val_stage3.npy")

pos_extra_val = np.load("intermediate/kl_list5_log_exp_val_stage3.npy")
neg_extra_val = np.load("intermediate/_kl_list5_log_exp_val_stage3.npy")

assert pos_val.shape[0] == pos_extra_val.shape[0]
assert neg_val.shape[0] == neg_extra_val.shape[0]

pos_flat_val = pos_val.reshape(len(pos_val), -1)
neg_flat_val = neg_val.reshape(len(neg_val), -1)

pos_combined_val = np.concatenate([pos_output_val > 1, pos_val, pos_extra_val], axis=1)
neg_combined_val = np.concatenate([neg_output_val > 1, neg_val, neg_extra_val], axis=1)

X_val = np.concatenate([pos_combined_val, neg_combined_val], axis=0)
y_val = np.array([1]*len(pos_combined_val) + [0]*len(neg_combined_val))


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = FeatureDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

val_dataset = FeatureDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)


def shift_one_right_batchwise(k: torch.Tensor) -> torch.Tensor:
    B, N = k.shape
    device = k.device

    indices = torch.argmax(k, dim=1)
    not_at_end = indices < (N - 1)
    not_at_start = indices != 0

    shift_mask = not_at_end & not_at_start     

    k_new = torch.zeros_like(k)

    row_idx = torch.arange(B, device=device)[shift_mask]
    col_idx = indices[shift_mask] + 1
    k_new[row_idx, col_idx] = 1

    row_idx_static = torch.arange(B, device=device)[~shift_mask]
    col_idx_static = indices[~shift_mask]
    k_new[row_idx_static, col_idx_static] = 1

    return k_new



# ----------------- encoder -----------------
class Encoder(nn.Module):
    def __init__(self, in_dim=10*1 + 5, embed_dim=16*4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64*3),
            nn.LeakyReLU(),
            nn.Linear(64*3, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, 1)  # 2 classes

    def forward(self, x):
        x_half = x[:,:5]
        x_half = keep_leftmost_one(x_half)
        x_half2 = x[:,5:10]
        x_half2 = shift_one_right_batchwise(x_half2)
        new_input = torch.cat((x_half, x_half2, x[:, 10:]), dim = 1)
        z = self.encoder(new_input) 
        logits = self.classifier(z).squeeze(-1) 
        return z, logits



class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0, w_pos: float = 1.0, w_neg: float = 5.0):
        super().__init__()
        self.margin = margin
        self.w_pos = w_pos
        self.w_neg = w_neg

    def forward(self, z1, z2, label):
        d = torch.norm(z1 - z2, dim=1)
        pos_loss = self.w_pos * label * d.pow(2)
        neg_loss = self.w_neg * (1 - label) * torch.clamp(self.margin - d, min=0).pow(2)
        return (pos_loss + neg_loss).mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()), lr=1e-3)
loss_fn = WeightedContrastiveLoss(margin=50.0, w_pos=1.0, w_neg=20.0)
bce_loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 300
best_val_auroc = -1.0
best_ckpt_path = "model/encoder_stage3.pth"

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

pos_global = (y_t == 1).nonzero(as_tuple=True)[0].tolist()
neg_global = (y_t == 0).nonzero(as_tuple=True)[0].tolist()

pair_batch_size = 64

for epoch in range(EPOCHS):
    encoder.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pairs = []
    pair_lbls = []

    for _ in range(250):
        i, j = random.sample(pos_global, 2)
        pairs.append((i, j))
        pair_lbls.append(1.0)

    for _ in range(250):
        i, j = random.sample(neg_global, 2)
        pairs.append((i, j))
        pair_lbls.append(1.0)

    for _ in range(5000):
        i = random.choice(pos_global)
        j = random.choice(neg_global)
        pairs.append((i, j))
        pair_lbls.append(0.0)

    perm = list(range(len(pairs)))
    random.shuffle(perm)
    pairs = [pairs[k] for k in perm]
    pair_lbls = [pair_lbls[k] for k in perm]

    for s in range(0, len(pairs), pair_batch_size):
        batch_pairs = pairs[s : s + pair_batch_size]
        batch_pair_lbl = torch.tensor(
            pair_lbls[s : s + pair_batch_size],
            dtype=torch.float32,
            device=device,
        )

        idx_a, idx_b = zip(*batch_pairs)
        idx_a = torch.tensor(idx_a, dtype=torch.long)
        idx_b = torch.tensor(idx_b, dtype=torch.long)

        xa = X_t[idx_a].to(device)
        xb = X_t[idx_b].to(device)
        ya = y_t[idx_a].to(device)
        yb = y_t[idx_b].to(device)

        za, logit_a = encoder(xa)
        zb, logit_b = encoder(xb)

        loss_contrast = loss_fn(za, zb, batch_pair_lbl)

        loss_cls_a = bce_loss_fn(logit_a, ya)
        loss_cls_b = bce_loss_fn(logit_b, yb)
        loss_classification = 0.5 * (loss_cls_a + loss_cls_b)

        total_loss = 1.0 * loss_contrast + 100.0 * loss_classification

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        pred_a = (logit_a > 0).float()
        pred_b = (logit_b > 0).float()
        running_correct += (pred_a == ya).sum().item()
        running_correct += (pred_b == yb).sum().item()
        running_total += ya.numel() + yb.numel()

    acc = running_correct / max(1, running_total)
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  Total Loss: {running_loss:.4f}  Train Acc: {acc:.4f}")

    encoder.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            _, val_logits = encoder(val_x)
            all_scores.append(val_logits.detach().cpu().numpy())
            all_labels.append(val_y.detach().cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    val_auroc = roc_auc_score(all_labels, all_scores)
    print(f"    Val AUROC: {val_auroc:.6f}")

    if val_auroc > best_val_auroc:
        print("saving...")
        best_val_auroc = val_auroc
        os.makedirs("model", exist_ok=True)
        torch.save({"net": encoder.state_dict()}, best_ckpt_path)





pos_output = np.load("intermediate/c_list5_log_exp_eval_stage3.npy")
neg_output = np.load("intermediate/_c_list5_log_exp_eval_stage3.npy")

pos = np.load("intermediate/k_list5_log_exp_eval_stage3.npy")
neg = np.load("intermediate/_k_list5_log_exp_eval_stage3.npy")

pos_extra = np.load("intermediate/kl_list5_log_exp_eval_stage3.npy")
neg_extra = np.load("intermediate/_kl_list5_log_exp_eval_stage3.npy")
assert pos.shape[0] == pos_extra.shape[0]
assert neg.shape[0] == neg_extra.shape[0]

pos_flat = pos.reshape(len(pos), -1)
neg_flat = neg.reshape(len(neg), -1)

pos_combined = np.concatenate([pos_output > 1, pos, pos_extra], axis=1)
neg_combined = np.concatenate([neg_output > 1, neg, neg_extra], axis=1)

X = np.concatenate([pos_combined, neg_combined], axis=0)
y = np.array([1]*len(pos_combined) + [0]*len(neg_combined))


dataset = FeatureDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

state_dict = torch.load(
        'model/encoder_stage3.pth', map_location='cuda'
)
encoder.load_state_dict(state_dict['net'])

encoder.eval()
with torch.no_grad():
    _, logits = encoder(torch.tensor(X, dtype=torch.float32).to('cuda'))
    logits = logits.cpu().numpy()
logits_class0_on_neg = logits[y == 0]
logits_class1_on_pos = logits[y == 1]
correct = 0
for l in logits_class0_on_neg:
    if l < 0:
        correct += 1
#print(correct / len(logits_class0_on_neg))
for l in logits_class1_on_pos:
    if l >= 0:
        correct += 1

print('Test acc: ', correct / (len(logits_class0_on_neg) + len(logits_class1_on_pos)))

