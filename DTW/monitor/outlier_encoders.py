import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import random
import os

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)




def load_data(prefix="", mode = 'train'):
    if True:
        return (
        np.load('monitor_dataset/{}_{}_phi_int_list_stack.npy'.format(prefix, mode)),
        np.load('monitor_dataset/{}_{}_cs_list_stack.npy'.format(prefix, mode)),
        np.load('monitor_dataset/{}_{}_l2_list_stack.npy'.format(prefix, mode)),
        np.load('monitor_dataset/{}_{}_ref_enc_list_stack.npy'.format(prefix, mode))
        )

# ========== Dataset ==========
class FullDataset(Dataset):
    def __init__(self, phi_list, cs_list, l1_list, ref_enc_list):
        self.data = list(zip(phi_list, cs_list, l1_list, ref_enc_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phi, cs, l1, ref = self.data[idx]
        return (
            torch.tensor(phi, dtype=torch.float32),
            torch.tensor(cs, dtype=torch.float32),
            torch.tensor(l1, dtype=torch.float32),
            torch.tensor(ref, dtype=torch.float32)
            )

# ========== Collate Function ==========
def collate_fn(batch):
    phi, cs, l1, ref = zip(*batch)
    return torch.stack(phi), torch.stack(cs), torch.stack(l1), torch.stack(ref)

class BranchCNNCat(nn.Module):

    def __init__(self, in_ch: int = 1, hidden: int = 32):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Linear(1 * 8, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )
        

        self.conv2 = nn.Sequential(
            nn.Linear(1 * 8, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.proj = nn.Linear(hidden*2, hidden)
        self.proj2 = nn.Linear(hidden, 1)
        # Kaiming init for conv, xavier for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x3 = self.conv(torch.flatten(y, start_dim=1))
        x2 = self.conv2(torch.flatten(x, start_dim=1))
        x = self.proj(torch.cat([x3,x2], dim = -1))
        x = self.proj2(F.relu(x))
        return x


class BranchCNN(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             

            nn.Conv2d(32, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),             

            nn.Conv2d(32, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  
        self.proj = nn.Linear(hidden*2, 1)         # -> (B, 1)

        # Kaiming init for conv, xavier for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.conv(x[:,None,:,:])
        x2 = self.conv2(y)
        x = self.gap(x).flatten(1) 
        x2 = self.gap(x2).flatten(1)
        x = self.proj(torch.cat([x,x2], dim = -1))            # (B, 1)
        return x


# ========== Contrastive Loss ==========
def contrastive_loss(f1, f2, label, margin=1.0):
    dist = torch.norm(f1 - f2, dim=1)
    loss_pos = label * dist.pow(2)
    loss_neg = (1 - label) * F.relu(margin - dist).pow(2)
    return (loss_pos + loss_neg).mean()


class GaussianClusterLoss(nn.Module):
    def __init__(self, feat_dim=2, sigma=1.0, learn_sigma=False):
        super().__init__()
        self.means = nn.Parameter(torch.randn(2, feat_dim))
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('log_sigma', torch.tensor(np.log(sigma)))

    def forward(self, feat, labels):
        mu = self.means[labels]
        dist_sq = ((feat - mu) ** 2).sum(dim=1)
        loss = 0.5 * dist_sq / torch.exp(self.log_sigma * 2) + self.log_sigma
        return loss.mean()

def evaluate(model1, model2, classifier, test_dataset, flag, device):
    model1.eval()
    model2.eval()
    classifier.eval()
    with torch.no_grad():
        data = [test_dataset[i] for i in range(len(test_dataset))]
        if flag == 'pos':
            test_labels = [0 for i in range(len(test_dataset))]
        else:
            test_labels = [1 for i in range(len(test_dataset))]
        phi, cs, l1, ref = map(lambda x: x.to(device), collate_fn(data))
        f1 = model1(phi, ref)
        f2 = model2(l1, cs)
        feat = torch.cat([f1, f2], dim=-1)
        logits = classifier(feat).squeeze(1)
        preds = (logits > 0).long().cpu()
        labels = torch.tensor(test_labels).long()
        #acc = (preds == labels).float().mean().item()
        correct = (preds == labels).sum().item()
        total = labels.numel()
    #print(f"Test Accuracy: {acc:.4f}")
    return correct, total

# ========== Training ==========
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    pos_data = load_data('pos', 'train')
    neg_data = load_data('neg', 'train')

    pos_dataset = FullDataset(*pos_data)
    neg_dataset = FullDataset(*neg_data)


    pos_data_ = load_data('pos', 'validation')
    neg_data_ = load_data('neg', 'validation')
    test_dataset_pos = FullDataset(*pos_data_)
    test_dataset_neg = FullDataset(*neg_data_)


    stage1_model = BranchCNN().to(device)
    stage2_model = BranchCNNCat(in_ch = 1).to(device)
    classifier = nn.Linear(2, 1).to(device)
    cluster_loss_fn = GaussianClusterLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(stage1_model.parameters()) + list(stage2_model.parameters()) + list(classifier.parameters()) + list(cluster_loss_fn.parameters()),
        lr=5e-4
    )

    for epoch in range(1, 301):

        stage1_model.train()
        stage2_model.train()
        classifier.train()

        total_correct = 0
        total_samples = 0

        pairs = []
        labels = []
        pos_idx = list(range(len(pos_dataset)))
        neg_idx = list(range(len(neg_dataset)))
        single_feats = []
        single_label = []
        for _ in range(50):
            i, j = random.sample(pos_idx, 2)
            pairs.append((pos_dataset[i], pos_dataset[j]))
            labels.append(1)
            single_feats.append(pos_dataset[i])
            single_label.append(0)
            single_feats.append(pos_dataset[j])
            single_label.append(0)
        for _ in range(50):
            i, j = random.sample(neg_idx, 2)
            pairs.append((neg_dataset[i], neg_dataset[j]))
            labels.append(1)
            single_feats.append(neg_dataset[i])
            single_label.append(1)
            single_feats.append(neg_dataset[j])
            single_label.append(1)
        for _ in range(1000):
            i = random.choice(pos_idx)
            j = random.choice(neg_idx)
            pairs.append((pos_dataset[i], neg_dataset[j]))
            labels.append(0)
            single_feats.append(pos_dataset[i])
            single_label.append(0)
            single_feats.append(neg_dataset[j])
            single_label.append(1)

        #random.shuffle(pairs)
        perm = list(range(len(pairs)))
        random.shuffle(perm)
        pairs  = [pairs[k] for k in perm]
        labels = [labels[k] for k in perm]

        batch_size = 32
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.float32).to(device)

            a_list, b_list = zip(*batch_pairs)
            a = collate_fn(a_list)
            b = collate_fn(b_list)
            #print(a)

            phia, csa, l1a, refa = map(lambda x: x.to(device), a)
            phib, csb, l1b, refb = map(lambda x: x.to(device), b)

            f1a = stage1_model(phia, refa)
            f2a = stage2_model(l1a, csa)
            f1b = stage1_model(phib, refb)
            f2b = stage2_model(l1b, csb)

            feat_a = torch.cat([f1a, f2a], dim=-1)
            feat_b = torch.cat([f1b, f2b], dim=-1)

            loss_contrast = contrastive_loss(feat_a, feat_b, batch_labels)
            
            single_list = collate_fn(single_feats)
            cls_batch = [x.to(device) for x in single_list]
            cls_labels = torch.tensor(single_label, dtype=torch.float32).to(device)
            phi1, cs1, l11, ref1 = map(lambda x: x.to(device), cls_batch)
            f_out = stage1_model(phi1, ref1)
            f_out2 = stage2_model(l11, cs1)

            feat_ = torch.cat([f_out, f_out2], dim=-1)
            

            logits_all = classifier(feat_).squeeze(1)
            loss_cls = F.binary_cross_entropy_with_logits(logits_all, cls_labels)


            loss_cluster = cluster_loss_fn(feat_, cls_labels.long())

            total_loss = loss_cluster + 100 * loss_cls + 1 * loss_contrast

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            preds = (logits_all > 0).long()
            total_correct += (preds == cls_labels.long()).sum().item()
            total_samples += cls_labels.size(0)

        acc = total_correct / total_samples
        print(f"Epoch {epoch}: Accuracy = {acc:.4f}")
        c1, t1 = evaluate(stage1_model, stage2_model, classifier, test_dataset_pos, 'pos', device)
        c2, t2 = evaluate(stage1_model, stage2_model, classifier, test_dataset_neg, 'neg', device)
        print(f"	Val Accuracy = {(c1 + c2) / (t1 + t2):.4f}")
        if (c1 + c2) / (t1 + t2) > best_acc:
            print('saving...')
            os.makedirs('model', exist_ok = True)
            torch.save({
            'stage1': stage1_model.state_dict(),
            'stage2': stage2_model.state_dict(),
            'classifier': classifier.state_dict(),
            }, 'model/model_monitor.pth'.format(epoch))
            best_acc = (c1 + c2) / (t1 + t2)


if __name__ == "__main__":
    train()

    #==============Eval=================#
    device = 'cpu'
    stage1_model = BranchCNN().to(device)
    stage2_model = BranchCNNCat().to(device)
    classifier = nn.Linear(2, 1)

    state_dict = torch.load('model/model_monitor.pth', map_location='cpu')
    stage1_model.load_state_dict(state_dict['stage1'])
    stage2_model.load_state_dict(state_dict['stage2'])
    classifier.load_state_dict(state_dict['classifier'])

    pos_data = load_data('pos', 'test')
    neg_data = load_data('neg', 'test')
    test_dataset_pos = FullDataset(*pos_data)
    test_dataset_neg = FullDataset(*neg_data)


    c1, t1 = evaluate(stage1_model, stage2_model, classifier, test_dataset_pos, 'pos', device)
    c2, t2 = evaluate(stage1_model, stage2_model, classifier, test_dataset_neg, 'neg', device)
    print('Test Accuracy: ', (c1  + c2) / (t1 + t2))
