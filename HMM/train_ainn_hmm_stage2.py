import joblib
import os
import torch
import pickle
import torch.nn as nn
import sys
import numpy as np
import random
import torch.nn.functional as F
from hmmlearn.hmm import CategoricalHMM
import warnings
import yaml
import argparse
"""
For each neural block, training converges faster (less iterations) than in a single end-to-end network because the constraints are tailored to the block’s specific subtask.
When incorporating prior knowledge as a loss regularizer, we assign it a small weight—typically at least an order of magnitude smaller than the main loss term—to prevent over-reliance, recognizing that AINN operates in a different representation space.
A gradient-free optimization method is provided here, which can be applied to both differentiable and non-differentiable designs.
A gradient-based optimization approach is also demonstrated in the keyword-spotting (DTW) task.
"""

parser = argparse.ArgumentParser('Train AINN - HMM')
parser.add_argument("--config", type=str, default = 'ainn', help="config")
args = parser.parse_args()

cfg = yaml.safe_load(open('config/{}.yaml'.format(args.config)))


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



def fitness_function(model1, model2, data, label, model_hmm, _model_hmm):
    model1.eval()
    model2.eval()

    z1, z_embed = model1(data)
    score = 0
    total = 0
    correct = 0
    for i in range(len(data)):
        obs_seq1 = z_embed[i].detach().cpu().numpy()
        ret, pred_label = model2(torch.tensor([obs_seq1]).to(torch.float32))

        score += (int(pred_label[0].detach().cpu().numpy()) == label[i]) 

        if int(pred_label[0].detach().cpu().numpy()) == int(label[i]):
            correct += 1
        total += 1
    #sys.exit(0)
    start_prob_nn = model2.hmm_fall.start_logits.detach().cpu().numpy()
    trans_prob_nn = model2.hmm_fall.trans_logits.detach().cpu().numpy()

    start_prob_nn2 = model2.hmm_normal.start_logits.detach().cpu().numpy()
    trans_prob_nn2 = model2.hmm_normal.trans_logits.detach().cpu().numpy()

    
    prob_terms = np.sum(np.abs(start_prob_nn - model_hmm.startprob_)) + np.sum(np.abs(trans_prob_nn - model_hmm.transmat_)) + np.sum(np.abs(start_prob_nn2 - _model_hmm.startprob_)) + np.sum(np.abs(trans_prob_nn2 - _model_hmm.transmat_))
    score -= 0.05 * prob_terms
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
        x_input = x               # [B, L, 1]
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
        return torch.sigmoid(log_prob)        


class DualHMMClassifier(nn.Module):
    def __init__(self, d_dim = 32 * 2):
        super().__init__()
        self.hmm_fall = HMMStage(d_model = d_dim) 
        self.hmm_normal = HMMStage(d_model = d_dim) 

    def forward(self, x):
        
        logp_fall = self.hmm_fall(x)      # [B]
        logp_normal = self.hmm_normal(x)  # [B]

        logprobs = torch.stack([logp_normal, logp_fall], dim=1)  # [B, 2]
        probs = F.softmax(logprobs, dim=1)                       # [B, 2]
        predicted_class = torch.argmax(probs, dim=1)             # [B]
        return probs, predicted_class




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
        all_preds = []       # Hard predictions: [B, T]
        all_soft_embeds = [] # Soft embeddings: [B, T, C]

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

                
                logits = self.shared_mlp(input_seq) 
                probs = F.softmax(logits, dim=-1)     

                preds = torch.argmax(logits, dim=-1).squeeze(0)  
                all_preds.append(preds.cpu().tolist())          
                all_soft_embeds.append(probs.squeeze(0).cpu()) 

        return all_preds, all_soft_embeds 




def particle_swarm_optimization(hidden_size, hstage_hidden_size, num_particles, num_iterations, data, label, data_test, label_test, model_hmm, _model_hmm, model0):
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_scores = []
    global_best_position = None
    global_best_score = float('-inf')
    global_best_cosine = -2
    best_fitness_score = -10000


    for _ in range(num_particles):
        mask_nn = DualHMMClassifier(hstage_hidden_size)
        random_params = {}
        for name, param in mask_nn.named_parameters():
            random_shape = param.shape
            random_tensor = torch.randn(random_shape)  # Fresh random values
            random_params[name] = random_tensor

        particles.append(random_params)

        velocities.append({k: torch.zeros_like(v) for k, v in random_params.items()})
        personal_best_positions.append({k: v.clone() for k, v in random_params.items()})
        personal_best_scores.append(float('-inf'))

    w, c1, c2 = cfg['stage2']['w'], cfg['stage2']['c1'], cfg['stage2']['c2']

    for iteration in range(num_iterations):
        for i in range(num_particles):
            model = DualHMMClassifier(hstage_hidden_size)
            model.load_state_dict(particles[i])
            
            fitness, acc = fitness_function(model0, model, data, label, model_hmm, _model_hmm)

            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = {k: v.clone() for k, v in particles[i].items()}

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_cosine = acc
                print(f"Iteration {iteration}: Particle={i}, Score={fitness:.2f}")
                global_best_position = {k: v.clone() for k, v in particles[i].items()}

                _fitness, _acc = fitness_function(model0, model, data_test, label_test, model_hmm, _model_hmm) #evaluate on validation
                print(f"        Val accuracy={_acc:.4f}")
                if _fitness > best_fitness_score:
                    print('	Saving...')
                    state_dict = {'net': model.state_dict()}
                    os.makedirs('ainn_model', exist_ok = True)
                    torch.save(state_dict, 'ainn_model/model_{}_hmm_{}.pth'.format(args.config, num_sample))

                    best_fitness_score = _fitness
                    _data_test0 = pickle.load(open('dataset/test_fall_dataset_seq.pkl', 'rb'))
                    _label_test0 = [1 for i in range(len(_data_test0))]
 
                    _data_test1 = pickle.load(open('dataset/_test_fall_dataset_seq.pkl', 'rb'))
                    _label_test1 = [0 for i in range(len(_data_test1))]

                    _data_test = _data_test0 + _data_test1
                    _label_test = _label_test0 + _label_test1

                    _, _test_acc = fitness_function(model0, model, _data_test, _label_test, model_hmm, _model_hmm)
                    print(f"        Test accuracy={_test_acc:.4f}")

        for i in range(num_particles):
            new_velocity = {}
            new_particle = {}
            for k in particles[i].keys():
                v = velocities[i][k]
                p = particles[i][k]
                pb = personal_best_positions[i][k]
                gb = global_best_position[k]
                new_v = w * v + c1 * torch.rand_like(v) * (pb - p) + c2 * torch.rand_like(v) * (gb - p)
                new_p = p + new_v
                new_velocity[k] = new_v
                new_particle[k] = new_p
            velocities[i] = new_velocity
            particles[i] = new_particle
        
    return global_best_position


data0 = pickle.load(open('dataset/train_fall_dataset_seq.pkl', 'rb'))
label0 = [1 for i in range(len(data0))]
d0_size = len(label0)

data1 = pickle.load(open('dataset/_train_fall_dataset_seq.pkl', 'rb'))
label1 = [0 for i in range(len(data1))]
d1_size = len(label1)

data = data0[:num_sample//2] + data1[:num_sample//2]
label = label0[:num_sample//2] + label1[:num_sample//2]


data_test0 = pickle.load(open('dataset/val_fall_dataset_seq.pkl', 'rb'))
label_test0 = [1 for i in range(len(data_test0))]

data_test1 = pickle.load(open('dataset/_val_fall_dataset_seq.pkl', 'rb'))
label_test1 = [0 for i in range(len(data_test1))]

data_test = data_test0 + data_test1
label_test = label_test0 + label_test1


model_hmm = CategoricalHMM(n_components=num_component)

_model_hmm = CategoricalHMM(n_components=num_component)


model_hmm = joblib.load("hmm_model/trained_hmm_model.pkl")
_model_hmm = joblib.load("hmm_model/_trained_hmm_model.pkl")


model0 = SimpleNN(hidden_size)
state_dict = torch.load(
        'ainn_model/model_{}_state_{}.pth'.format(args.config, num_sample), map_location='cpu'
    )
model0.load_state_dict(state_dict['net'])

_ = particle_swarm_optimization(
    hidden_size, hstage_hidden_size, num_particles, num_iterations, data, label, data_test, label_test, model_hmm, _model_hmm, model0)
