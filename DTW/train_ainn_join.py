import os
import random
import numpy as np
from math import inf
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
import torchsort
import argparse
import yaml

parser = argparse.ArgumentParser('Train AINN - DTW')
parser.add_argument("--config", type=str, default = 'ainn', help="config")
args = parser.parse_args()

cfg = yaml.safe_load(open('config/{}.yaml'.format(args.config)))

# -------------------- Repro --------------------
SEED = cfg['seed'] #12345
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

KNN_BREF = cfg['knn_bref'] #6   # references per class
KNN_K    = cfg['knn_k'] #2   # k neighbors to vote
prtn = cfg['portion']
epoch = cfg['epoch']

class FeatureDataset(torch.utils.data.Dataset):

    def __init__(self, mode, portion = 1.0):
        if mode == 'training':
            self.X = np.load("dataset/{}_X_{}.npy".format(mode, portion))
            self.y = np.load("dataset/{}_y_{}.npy".format(mode, portion))
        else:
            self.X = np.load("dataset/{}_X.npy".format(mode))
            self.y = np.load("dataset/{}_y.npy".format(mode))
        self.class_to_idxs = {}
        for _i in range(len(self.y)):
            if not self.y[_i] in self.class_to_idxs:
                self.class_to_idxs[self.y[_i]] = []
            self.class_to_idxs[self.y[_i]].append(_i)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'X': np.asarray(self.X[idx]), 'y': np.asarray(self.y[idx])}
        return sample



#For each neural block, training needs less iteration to converge than a single end-to-end
#network because the constraints are tailored to the block's specific subtask.


class FrameEncoder(nn.Module):
    def __init__(self, F, d=64, hidden=96, context=7, dropout=0.1):
        super().__init__()
        k = context          
        self.prj = nn.Linear(F, hidden) 

        def dw_pw(cin, cout, dilation=1):
            return nn.Sequential(
                nn.Conv1d(cin, cin, kernel_size=k, padding=dilation*(k//2), dilation=dilation, groups=cin),
                nn.Conv1d(cin, cout, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.block1 = dw_pw(hidden, hidden, dilation=1)
        self.block2 = dw_pw(hidden, hidden, dilation=2)
        self.block3 = dw_pw(hidden, d, dilation=4)
        self.ln = nn.LayerNorm(d)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.prj.weight)
        if self.prj.bias is not None:
            nn.init.zeros_(self.prj.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x):          # x: [B, T, F]
        B,T,F = x.shape
        h = self.prj(x)            # [B,T,hidden]
        h = h.transpose(1, 2)      # [B,hidden,T] for Conv1d over time
        r = h
        h = self.block1(h) + r     # residual
        r = h
        h = self.block2(h) + r
        h = self.block3(h)         # [B,d,T]
        h = h.transpose(1, 2)      # [B,T,d]
        return self.ln(h)



class NN1_LocalMatcher(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1, cosine_head=True):
        super().__init__()
        self.cosine_head = cosine_head
        self.scale = nn.Parameter(torch.tensor(10.0))  # learnable temperature
        self.W = nn.Parameter(torch.eye(d))

    @torch.no_grad()
    def _cosine(self, phi_t, psi_u):
        a = F.normalize(phi_t, dim=-1)
        b = F.normalize(psi_u, dim=-1)
        return (a * b).sum(dim=-1)  # [B]

    def reg_loss(self, phi_t, psi_u, c_tu, lambda_reg=0.1):
        cos_tgt = -1 * self._cosine(phi_t, psi_u)
        pred_cos = torch.tanh(c_tu)
        return lambda_reg * F.l1_loss(c_tu, cos_tgt)


class KNNLogitHead(nn.Module):
    def __init__(self, K: int=8, class_emb_dim: int = 16,
                 phi_hidden: int = 32, rho_hidden: int = 32,
                 use_counts: bool = True, residual_init: float = 0.0):
        super().__init__()
        self.K = K
        self.use_counts = use_counts

        # Class embeddings so the net can learn class-specific weighting if useful
        #self.class_emb = nn.Embedding(K, class_emb_dim)

        in_phi = 1
        self.phi = nn.Sequential(
            nn.Linear(in_phi, phi_hidden), nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden), nn.GELU()
        )

        in_rho = phi_hidden + (1 if use_counts else 0)
        self.rho = nn.Sequential(
            nn.Linear(in_rho, rho_hidden), nn.GELU(),
            nn.Linear(rho_hidden, 1)
        )
        self.alpha = nn.Parameter(torch.ones(K))
        self.bias  = nn.Parameter(torch.zeros(K))

        self.residual_scale = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))
        self.residual_scale2 = nn.Parameter(torch.tensor(residual_init, dtype=torch.float32))
        # Lightweight init
        for m in list(self.phi) + list(self.rho):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        #nn.init.normal_(self.class_emb.weight, std=0.02)

    def _soft_onehot_from_fractional(self, frac_ids: torch.Tensor) -> torch.Tensor:
        B, k = frac_ids.shape
        device = frac_ids.device
        K = self.K
        cls_idx = torch.arange(K, device=device, dtype=frac_ids.dtype).view(1,1,K)
        dist2 = (frac_ids.unsqueeze(-1) - cls_idx).abs()
        logits = - dist2 / 1e-5
        oh_soft = F.softmax(logits, dim=-1)
        return oh_soft


    def forward(self, topk_vals: torch.Tensor, topk_classes: torch.Tensor) -> torch.Tensor:
        B, k = topk_vals.shape
        K = self.K
        device = topk_vals.device
        dtype  = topk_vals.dtype

        oh = self._soft_onehot_from_fractional(topk_classes) 
        class_sum = torch.einsum('bkc,bk->bc', oh, topk_vals)   # [B, K]

        neigh_feat = torch.cat([topk_vals.unsqueeze(-1),], dim=-1)  # [B,k,1+E]
        phi_out = self.phi(neigh_feat)                          # [B,k,H]

        pooled = torch.einsum('bkc,bkh->bch', oh, phi_out)      # [B,K,H]

        if self.use_counts:
            counts = oh.sum(dim=1, keepdim=False)               # [B,K]
            pooled = torch.cat([pooled, counts.unsqueeze(-1)], dim=-1)  # [B,K,H+1]

        delta = self.rho(pooled).squeeze(-1)                    # [B, K]

        logits_like = class_sum * self.residual_scale2 + self.residual_scale * delta   # [B, K]
        return logits_like




class SoftArgsortTopK(nn.Module):
    def __init__(self, reg_strength: float = 0.01, temperature: float = 1e-1, largest: bool = True):
        super().__init__()
        self.reg_strength = reg_strength
        self.temperature  = temperature
        self.largest      = largest

    def forward(self, scores: torch.Tensor, ref_class: torch.Tensor, k: int = KNN_K, K: int = 8):
        B, R = scores.shape
        device, dtype = scores.device, scores.dtype

        # Broadcast ref_class to [B, R] if needed
        if ref_class.dim() == 1:
            ref_class = ref_class.view(1, R).expand(B, R)  # [B,R]
        ref_class_float = ref_class.to(dtype)
        s_in = scores
        sorted_vals = torchsort.soft_sort(
            s_in, regularization_strength=self.reg_strength
        )  # [B, R]
        yk = sorted_vals[:, -k:]      # [B,k]

        M = (s_in.unsqueeze(1) - yk.unsqueeze(-1)).abs()   # [B,k,R]

        logits = -M / self.temperature                     # [B,k,R]
        soft_mask = F.softmax(logits, dim=-1)              # [B,k,R]

        soft_label_expectation = torch.einsum('bkr,br->bk', soft_mask, ref_class_float)  # [B,k]
        return yk, soft_label_expectation



class NN2_SoftMin(nn.Module):
    def __init__(self, init_tau=0.2, tau_min=0.02, tau_max=2.0, learn_tau=True, add_gap=False, hidden = 16):
        super().__init__()
        self.learn_tau = learn_tau
        self.tau_min, self.tau_max = tau_min, tau_max
        self.net = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau))) if learn_tau else nn.Parameter(torch.log(torch.tensor(init_tau)), requires_grad=False)
        self.add_gap = add_gap
        if add_gap:
            self.gap = nn.Parameter(torch.zeros(3))  # bias moves if desired

    def forward(self, neigh3: torch.Tensor):
        if self.add_gap:
            neigh3 = neigh3 + self.gap.view(1,3)
        tau = torch.clamp(torch.exp(self.log_tau), self.tau_min, self.tau_max) #stablize training
        best_vals = -tau * torch.logsumexp(-neigh3 / tau, dim=-1)  # [B]
        best_vals = torch.sum(neigh3 * F.softmax(self.net(-torch.abs(neigh3 - best_vals.unsqueeze(-1))), dim = -1), dim = -1)
        return best_vals


class AINN_SC_Merged(nn.Module):
    def __init__(self, F=13, T=61, num_classes=8, d=64//2, hidden=64//2):
        super().__init__()
        self.F, self.T, self.K = F, T, num_classes
        self.enc = FrameEncoder(F, d)
        self.nn1 = NN1_LocalMatcher(d, hidden)
        self.nn2 = NN2_SoftMin(d, hidden)
        self.selector = SoftArgsortTopK()
        self.agg = KNNLogitHead()

    def _neigh3x3(self, DP: torch.Tensor, t: int, u: int) -> torch.Tensor:
        B = DP.size(0)
        rows = torch.tensor([t-1, t], device=DP.device).clamp(min=0)
        cols = torch.tensor([u-1, u], device=DP.device).clamp(min=0)
        neigh, mask = [], []
        for r in rows:
            for c in cols:
                if r == t and c == u:
                    continue
                neigh.append(DP[:, r, c])         
                mask.append((r == 0) | (c == 0))
        neigh = torch.stack(neigh, dim=-1)        
        mask = torch.tensor(mask, device=DP.device, dtype=DP.dtype).unsqueeze(0)
        return neigh
    
    def forward(self, x, refs):
        B, T, Fk = x.shape
        K, B_ref = refs.shape[0], refs.shape[1]
        phi = self.enc(x)
        all_scores = x.new_zeros(B, K, B_ref)   

        K, B_ref, T, Fk = refs.shape
        B = x.size(0)
        R = K * B_ref

        
        phi = self.enc(x)                            # [B, T, d]
        refs_flat = refs.reshape(R, T, Fk)           # [R, T, F]
        psi_all = self.enc(refs_flat)                # [R, T, d]

        
        q = F.normalize(phi, dim=-1)  # [B, T, d]
        k = F.normalize(psi_all, dim=-1)  # [R, T, d]
        
        W = self.nn1.W                              # (d, d)
        W_s = 0.5 * (W + W.t())
        qW = torch.einsum('btd,df->btf', q, W_s)     # [B, T, d]
        cross = torch.einsum('btf,ruf->brtu', qW, k)  # [B, R, T, T]
        C_all = -self.nn1.scale * cross


        
        DP = x.new_full((B, R, T+1, T+1), 0.0)
        pad_bias = 8.0 
        DP[:, :, 0, :] = pad_bias
        DP[:, :, :, 0] = pad_bias
        DP[:, :, 0, 0]  = 0.0
        
        for t in range(1, T+1):
            for u in range(1, T+1):
                up   = DP[:, :, t-1, u]      # [B,R]
                left = DP[:, :, t,   u-1]    # [B,R]
                diag = DP[:, :, t-1, u-1]    # [B,R]
                DP[:, :, t, u] = C_all[:, :, t-1, u-1] + self.nn2(torch.stack([up, left, diag], dim = -1))            # [B,R]

        """
        #The following approach speeds up DP computation during inference (deployment); feel free to switch to this version when you deploy them on the embedded devices.
        diags = []
        for k in range(2, 2*T + 1):
            t_vals = torch.arange(1, T+1)
            u_vals = k - t_vals
            mask = (u_vals >= 1) & (u_vals <= T)
            t_idx = t_vals[mask]
            u_idx = u_vals[mask]
            diags.append((t_idx, u_idx))

        for t_idx, u_idx in diags:
            t_e = t_idx.view(1, 1, -1).to(DP.device)
            u_e = u_idx.view(1, 1, -1).to(DP.device)

            up   = DP[:, :, t_e - 1, u_e]     # [B, R, L]
            left = DP[:, :, t_e,     u_e - 1] # [B, R, L]
            diag = DP[:, :, t_e - 1, u_e - 1] # [B, R, L]

            c_loc = C_all[:, :, t_e - 1, u_e - 1]  # [B, R, L]

            cand = torch.stack([up, left, diag], dim=-1)  # [B, R, L, 3]
            trans = self.nn2(cand)                        # [B, R, L]

            DP[:, :, t_e, u_e] = c_loc + trans
        """

        all_scores = DP[:, :, T, T]                 # [B, R]
        all_scores = all_scores.view(B, K, B_ref)   # [B, K, B_ref]

        

        flat_scores = all_scores.reshape(B, K * B_ref)            # lower = better
        ref_class = torch.arange(K, device=x.device).repeat_interleave(B_ref)  # [K*B_ref]

        soft_vals, soft_classes = self.selector(
            -flat_scores, ref_class
        )
        
        logits_like = self.agg(soft_vals, soft_classes)
        return torch.softmax(logits_like, dim = 1)


@torch.no_grad()
def build_knn_refs(dataset, K: int, T: int, F: int, B_ref: int, device=None, seed: int | None = None):
    device = device or "cpu"
    rng = np.random.default_rng(seed)
    refs = torch.zeros(K, B_ref, T, F, device=device, dtype=torch.float32)
    labels = torch.zeros(K, B_ref, dtype=torch.long, device=device)

    for k in range(K):
        pool = dataset.class_to_idxs.get(k, [])
        if len(pool) == 0:
            raise RuntimeError(f"No samples in class {k} for reference building.")
        if len(pool) >= B_ref:
            picks = rng.choice(pool, size=B_ref, replace=False)
        else:
            picks = rng.choice(pool, size=B_ref, replace=True)

        for r, idx in enumerate(picks):
            xk = dataset[idx]['X']
            refs[k, r] = torch.as_tensor(xk, dtype=torch.float32, device=device)
            labels[k, r] = k
    return refs, labels



# -------------------- Train / Eval --------------------
def run_sc(
    epochs=80,
    batch_size=64*1,
    lr=1e-3,
    device=None,
    portion = 0.1
):
    best_acc = -1
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    tr = FeatureDataset('training', portion)
    va = FeatureDataset('validation')
    te = FeatureDataset('testing')
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2) 
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AINN_SC_Merged(F=13, T=61, num_classes=8, d=64, hidden=64).to(device)
    
    #total = sum(p.numel() for p in model.parameters())
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-07, weight_decay=0)
    def eval_loop(loader, proto_source_ds):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for _input in loader:
                x, y = _input['X'], _input['y']
                B = x.size(0)
                x, y = x.to(device), y.to(device)
                refs, _ = build_knn_refs(proto_source_ds, K=8, T=61, F=13,
                                     B_ref=KNN_BREF, device=device, seed=SEED) #label will be reconstructed in ainn
                logits = model(x, refs)

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
            refs, _ = build_knn_refs(tr, K=8, T=61, F=13,
                         B_ref=KNN_BREF, device=device, seed=SEED) #label will be reconstructed in ainn
            logits = model(x, refs)
            W = model.nn1.W                              # (d, d)
            I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
            lambda_I = 1e-1                              # tunable

            loss = F.cross_entropy(logits, y) + lambda_I * torch.norm(W - I, p='fro')**2
            opt.zero_grad()
            loss.backward()
            opt.step()
            runloss += loss.item() * y.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

        val_acc = eval_loop(val_loader, tr)
        print(f"Epoch {ep}: train_loss={runloss/len(tr):.4f}  val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            print("Saving...")
            best_acc = val_acc
            state_dict = {'net': model.state_dict()}
            os.makedirs('ainn_model', exist_ok = True)
            torch.save(state_dict, f'ainn_model/dtw_model_{args.config}_{portion:.1f}.pth')

    state_dict = torch.load(
        f'ainn_model/dtw_model_{args.config}_{portion:.1f}.pth', map_location='cuda'
    )
    model.load_state_dict(state_dict['net'])
    test_acc = eval_loop(test_loader, tr)
    print(f"Test accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    run_sc(portion = prtn, epochs = epoch)
