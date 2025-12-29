import joblib
import torch
import pickle
import torch.nn as nn
import os
import sys
import numpy as np
import random
import torch.nn.functional as F
from hmmlearn.hmm import CategoricalHMM
from sklearn.preprocessing import KBinsDiscretizer
import warnings
import yaml
import argparse

parser = argparse.ArgumentParser('Get intermediate results of AINN - DTW')
parser.add_argument("--config", type=str, default = 'ainn_less_iter', help="config")
args = parser.parse_args()

cfg = yaml.safe_load(open('../config/{}.yaml'.format(args.config)))

torch.manual_seed(cfg['seed'])

num_sample = cfg['data']['num_samples']
hidden_size = cfg['stage1']['hidden_size']
hstage_hidden_size = cfg['stage2']['hstage_hidden_size']
num_particles = cfg['stage2']['num_particles']
num_iterations = cfg['stage2']['num_iterations']
num_component = cfg['hmm_component']


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
    module="torch.nn.modules.transformer"
)

torch.manual_seed(0)

def fitness_function(model1, model2, data, label, model_hmm, _model_hmm, mode = 'train'):
    model1.eval()
    model2.eval()

    z1, z_embed, inter = model1(data) 

    score = 0
    total = 0
    correct = 0
    stage1_z1 = []
    stage1_z_embed = []
    stage1_inter = []
    stage2_ret = []
    stage2_pred = []
    stage2_inter = []
    stage2_inter2 = []
    _stage1_z1 = []
    _stage1_z_embed = []
    _stage1_inter = []
    _stage2_ret = []
    _stage2_pred = []
    _stage2_inter = []
    _stage2_inter2 = []

    for i in range(len(data)):
        obs_seq1 = z_embed[i].detach().cpu().numpy()
        ret, pred_label, inter_h1, inter_h2 = model2(torch.tensor([obs_seq1]).to(torch.float32))

        score += (int(pred_label[0].detach().cpu().numpy()) == label[i]) 

        if int(pred_label[0].detach().cpu().numpy()) == int(label[i]): 
            correct += 1
            stage1_z1.append(z1[i])
            stage1_z_embed.append(obs_seq1)
            stage1_inter.append(inter[i].detach().cpu().numpy())

            stage2_ret.append(ret[0].detach().cpu().numpy())
            stage2_pred.append(pred_label[0].detach().cpu().numpy())
            stage2_inter.append(inter_h1[0].detach().cpu().numpy())
            stage2_inter2.append(inter_h2[0].detach().cpu().numpy())

        else:
            _stage1_z1.append(z1[i])
            _stage1_z_embed.append(obs_seq1)
            _stage1_inter.append(inter[i].detach().cpu().numpy())

            _stage2_ret.append(ret[0].detach().cpu().numpy())
            _stage2_pred.append(pred_label[0].detach().cpu().numpy())
            _stage2_inter.append(inter_h1[0].detach().cpu().numpy())
            _stage2_inter2.append(inter_h2[0].detach().cpu().numpy())

        total += 1
    start_prob_nn = model2.hmm_fall.start_logits.detach().cpu().numpy()
    trans_prob_nn = model2.hmm_fall.trans_logits.detach().cpu().numpy()

    start_prob_nn2 = model2.hmm_normal.start_logits.detach().cpu().numpy()
    trans_prob_nn2 = model2.hmm_normal.trans_logits.detach().cpu().numpy()

    prob_terms = np.sum(np.abs(start_prob_nn - model_hmm.startprob_)) + np.sum(np.abs(trans_prob_nn - model_hmm.transmat_)) + np.sum(np.abs(start_prob_nn2 - _model_hmm.startprob_)) + np.sum(np.abs(trans_prob_nn2 - _model_hmm.transmat_))
    score -= 0.05 * prob_terms
    os.makedirs('intermediate', exist_ok = True)
    with open('intermediate/stage1_z1_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage1_z1, f)
    with open('intermediate/stage1_z_embed_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage1_z_embed, f)
    with open('intermediate/stage1_inter_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage1_inter, f)
    with open('intermediate/stage2_ret_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage2_ret, f)
    with open('intermediate/stage2_pred_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage2_pred, f)
    with open('intermediate/stage2_inter_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage2_inter, f)
    with open('intermediate/stage2_inter2_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(stage2_inter2, f)

    with open('intermediate/_stage1_z1_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage1_z1, f)
    with open('intermediate/_stage1_z_embed_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage1_z_embed, f)
    with open('intermediate/_stage1_inter_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage1_inter, f)
    with open('intermediate/_stage2_ret_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage2_ret, f)
    with open('intermediate/_stage2_pred_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage2_pred, f)
    with open('intermediate/_stage2_inter_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage2_inter, f)
    with open('intermediate/_stage2_inter2_{}.pkl'.format(mode), 'wb') as f:
        pickle.dump(_stage2_inter2, f)

    return score, correct/total


class HMMStage(nn.Module):
    def __init__(self, d_model=32*2, n_states=3):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=1),
            num_layers=1
        )

        self.n_states = n_states
        self.state_head = nn.Linear(d_model, n_states)  
        self.start_logits = nn.Parameter(torch.randn(n_states))          
        self.trans_logits = nn.Parameter(torch.randn(n_states, n_states))  

    def forward(self, x):
        B, L, D = x.shape
        x_input = x                             # [B, L, 1]
        h = self.embed(x_input)                 # [B, L, D]
        h = h.permute(1, 0, 2)                  # [L, B, D]
        h = self.encoder(h)                     # [L, B, D]
        h = F.relu(h.permute(1, 0, 2))          # [B, L, D]

        emission_logits = self.state_head(h)            # [B, L, N_states]

        log_start = F.log_softmax(self.start_logits, dim=-1)             # [N]
        log_trans = F.log_softmax(self.trans_logits, dim=-1)             # [N, N]

        log_alpha = log_start + emission_logits[:, 0]                    # [B, N]
        for t in range(1, L):
            emit = emission_logits[:, t]                                 # [B, N]
            log_alpha = torch.logsumexp(
                log_alpha.unsqueeze(2) + log_trans.unsqueeze(0), dim=1
            ) + emit                                                     # [B, N]

        log_prob = torch.logsumexp(log_alpha, dim=1)                     # [B]
        return torch.sigmoid(log_prob), h.mean(dim = 1).squeeze(1)




class DualHMMClassifier(nn.Module):
    def __init__(self, n_states=3, obs_dim=1, hidden=64):
        super().__init__()
        self.hmm_fall = HMMStage()
        self.hmm_normal = HMMStage()

    def forward(self, x):
        logp_fall, _h_fall = self.hmm_fall(x)      # [B]
        logp_normal, _h_normal = self.hmm_normal(x)  # [B]

        logprobs = torch.stack([logp_normal, logp_fall], dim=1)  # [B, 2]
        probs = F.softmax(logprobs, dim=1)                       # [B, 2]
        predicted_class = torch.argmax(probs, dim=1)             # [B]
        return probs, predicted_class, _h_fall, _h_normal





class SimpleNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleNN, self).__init__()
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )


    def forward(self, x):
        outputs = []
        all_preds = []      
        all_soft_embeds = []
        all_int = []
        
        self.eval()  # optional: freeze dropout/layernorm if present
        with torch.no_grad():
            for i in range(len(x)):
                features = []
                for seg in x[i]:  # Each seg is a length-250 list
                    seg_np = np.array(seg)
                    features.append([
                        np.mean(seg_np),
                        np.std(seg_np),
                        np.median(seg_np)
                    ])
                input_seq = torch.tensor([features], dtype=torch.float32)  # shape: (1, T, 3)

                
                out1 = self.shared_mlp[0](input_seq)        
                out2 = self.shared_mlp[1](out1)       # ReLU
                all_int.append(out2.squeeze(0))
                logits = self.shared_mlp[2](out2) 

                probs = F.softmax(logits, dim=-1)            # [1, T, C]

                preds = torch.argmax(logits, dim=-1).squeeze(0)  # shape (T,)

                all_preds.append(preds.cpu().tolist())          # List[int] for this sequence
                all_soft_embeds.append(probs.squeeze(0).cpu())  # [T, C]

        return all_preds, all_soft_embeds, all_int  # [B, T], [B, T, C]




data0 = pickle.load(open('../dataset/train_fall_dataset_seq.pkl', 'rb'))
label0 = [1 for i in range(len(data0))]
data1 = pickle.load(open('../dataset/_train_fall_dataset_seq.pkl', 'rb'))
label1 = [0 for i in range(len(data1))]

#data = data0 + data1
#label = label0 + label1
data = data0[:] + data1[:]
label = label0[:] + label1[:]


data_test0 = pickle.load(open('../dataset/val_fall_dataset_seq.pkl', 'rb'))
label_test0 = [1 for i in range(len(data_test0))]

data_test1 = pickle.load(open('../dataset/_val_fall_dataset_seq.pkl', 'rb'))
label_test1 = [0 for i in range(len(data_test1))]

data_test = data_test0 + data_test1
label_test = label_test0 + label_test1


_data_test0 = pickle.load(open('../dataset/test_fall_dataset_seq.pkl', 'rb'))
_label_test0 = [1 for i in range(len(_data_test0))]

_data_test1 = pickle.load(open('../dataset/_test_fall_dataset_seq.pkl', 'rb'))
_label_test1 = [0 for i in range(len(_data_test1))]

_data_test = _data_test0 + _data_test1
_label_test = _label_test0 + _label_test1


model_hmm = CategoricalHMM(n_components=num_component)

_model_hmm = CategoricalHMM(n_components=num_component)

model_hmm = joblib.load("../hmm_model/trained_hmm_model.pkl")
_model_hmm = joblib.load("../hmm_model/_trained_hmm_model.pkl")


model0 = SimpleNN(hidden_size)
state_dict = torch.load(
        '../ainn_model/model_{}_state_{}.pth'.format(args.config, num_sample), map_location='cpu'
    )
model0.load_state_dict(state_dict['net'])


model1 = DualHMMClassifier()
state_dict = torch.load(
        '../ainn_model/model_{}_hmm_{}.pth'.format(args.config, num_sample), map_location='cpu'
    )
model1.load_state_dict(state_dict['net'])

print('constructing training part...')
fitness_function(model0, model1, data, label, model_hmm, _model_hmm, mode = 'train')
print('constructing validation part...')
fitness_function(model0, model1, data_test, label_test, model_hmm, _model_hmm, mode = 'val')
print('constructing testing part...')
fitness_function(model0, model1, _data_test, _label_test, model_hmm, _model_hmm, mode = 'test')
