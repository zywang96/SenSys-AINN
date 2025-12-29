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
import yaml
import argparse

parser = argparse.ArgumentParser('Train AINN - HMM')
parser.add_argument("--config", type=str, default = 'ainn', help="config")
args = parser.parse_args()

cfg = yaml.safe_load(open('config/{}.yaml'.format(args.config)))

torch.manual_seed(cfg['seed'])

num_sample = cfg['data']['num_samples']
hidden_size = cfg['stage1']['hidden_size']
num_particles = cfg['stage1']['num_particles']
num_iterations = cfg['stage1']['num_iterations']
num_component = cfg['hmm_component']


def forward_scaling(startprob, transmat, frameprob):
    min_sum = 1e-300
    T, N = frameprob.shape

    fwdlattice = np.zeros((T, N))
    scaling = np.zeros(T)
    log_prob = 0.0

    # Initialization t = 0
    for i in range(N):
        fwdlattice[0, i] = startprob[i] * frameprob[0, i]
    s = np.sum(fwdlattice[0])
    if s < min_sum:
        raise ValueError("Forward pass failed with underflow at t=0; consider using log-space implementation")
    scale = 1.0 / s
    scaling[0] = scale
    for i in range(N):
        fwdlattice[0, i] *= scale
    log_prob -= np.log(scale)

    # Induction
    for t in range(1, T):
        for j in range(N):
            acc = 0.0
            for i in range(N):
                acc += fwdlattice[t - 1, i] * transmat[i, j]
            fwdlattice[t, j] = acc * frameprob[t, j]
        s = np.sum(fwdlattice[t])
        if s < min_sum:
            raise ValueError(f"Forward pass failed with underflow at t={t}; consider using log-space implementation")
        scale = 1.0 / s
        scaling[t] = scale
        for j in range(N):
            fwdlattice[t, j] *= scale
        log_prob -= np.log(scale)

    return log_prob, fwdlattice, scaling



def fitness_function(model1, data, label, model_hmm, _model_hmm):
    model1.eval()
    z1, _ = model1(data) 
    score = 0
    total = 0
    correct = 0
    for i in range(len(z1)):
        obs_seq1 = z1[i]

        seq_input1 = np.array(obs_seq1).reshape(-1, 1)
        frameprob1 = model_hmm._compute_likelihood(seq_input1)
        log_prob1, _, _ = forward_scaling(model_hmm.startprob_, model_hmm.transmat_, frameprob1)

        frameprob2 = _model_hmm._compute_likelihood(seq_input1)
        log_prob2, _, _ = forward_scaling(_model_hmm.startprob_, _model_hmm.transmat_, frameprob2)


        score += label[i] * (log_prob1 > log_prob2) + (1 - label[i]) * (log_prob2 > log_prob1)


        if (log_prob1 >= log_prob2 and label[i] == 1) or (log_prob1 < log_prob2 and label[i] == 0):
            correct += 1
        total += 1

    return score, correct/total




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
                input_seq = torch.tensor([features], dtype=torch.float32) 

                logits = self.shared_mlp(input_seq)  
                probs = F.softmax(logits, dim=-1)

                preds = torch.argmax(logits, dim=-1).squeeze(0)  # shape (T,)
                all_preds.append(preds.cpu().tolist())          # List[int] for this sequence
                all_soft_embeds.append(probs.squeeze(0).cpu())  # [T, C]

        return all_preds, all_soft_embeds 


def particle_swarm_optimization(hidden_size, num_particles, num_iterations, data, label, data_test, label_test, model_hmm, _model_hmm):
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_scores = []
    global_best_position = None
    global_best_score = float('-inf')
    global_best_cosine = -2
    best_fitness_score = -10000


    for _ in range(num_particles):
        mask_nn = SimpleNN(hidden_size)
        random_params = {}
        for name, param in mask_nn.named_parameters():
            random_shape = param.shape
            random_tensor = torch.randn(random_shape)  # Fresh random values
            random_params[name] = random_tensor

        particles.append(random_params)

        velocities.append({k: torch.zeros_like(v) for k, v in random_params.items()})
        personal_best_positions.append({k: v.clone() for k, v in random_params.items()})
        personal_best_scores.append(float('-inf'))

    w, c1, c2 = cfg['stage1']['w'], cfg['stage1']['c1'], cfg['stage1']['c2']

    for iteration in range(num_iterations):
        for i in range(num_particles):
            model = SimpleNN(hidden_size)
            model.load_state_dict(particles[i])
            
            fitness, acc = fitness_function(model, data, label, model_hmm, _model_hmm)

            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = {k: v.clone() for k, v in particles[i].items()}

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_cosine = acc
                print(f"Iteration {iteration}: Particle={i}, Score={fitness:.2f}")
                global_best_position = {k: v.clone() for k, v in particles[i].items()} 

                _fitness, _ = fitness_function(model, data_test, label_test, model_hmm, _model_hmm)
                print(f"	Val score={_fitness:.4f}")
                if _fitness > best_fitness_score:
                    print(f'	Saving...')
                    state_dict = {'net': model.state_dict()}
                    os.makedirs('ainn_model', exist_ok = True)
                    torch.save(state_dict, 'ainn_model/model_{}_state_{}.pth'.format(args.config, num_sample))

                    best_fitness_score = _fitness


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


_ = particle_swarm_optimization(
    hidden_size, num_particles, num_iterations, data, label, data_test, label_test, model_hmm, _model_hmm)
