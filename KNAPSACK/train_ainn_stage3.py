import torch
import torch.nn as nn
import sys
import os
import numpy as np
import random
import torch.nn.functional as F
from stage_helper.run_eval import main, run_inference
from flax.core import freeze, unfreeze
import yaml
import argparse

parser = argparse.ArgumentParser('Train AINN - KNAPSACK')
parser.add_argument("--config", type=str, default = 'ainn', help="config")
args = parser.parse_args()

cfg = yaml.safe_load(open('config/{}.yaml'.format(args.config)))


SEED = cfg['seed'] #12345
torch.manual_seed(SEED)
num_sample = cfg['data']['num_samples']

input_size = cfg['stage1']['input_size'] #10
hidden_size = cfg['stage1']['hidden_size'] #64
output_size = cfg['stage1']['output_size'] #5
fstage_hidden_size = cfg['stage3']['fstage_hidden_size']
num_particles = cfg['stage3']['num_particles'] #100
num_iterations = cfg['stage3']['num_iterations'] #50

def compute_acc(pred, labels):
    pred = torch.round(pred.float())
    res = (pred.type(torch.int64) == labels.type(torch.int64))
    a, b = res.size()
    return torch.sum(res), (a * b)
    c = 0
    t = 0
    for i in range(pred.size()[0]):
        res = torch.equal(pred[i].type(torch.int64), labels[i].type(torch.int64))
        c += res
        t += 1
    return c, t

def inverse_argsort(arr):
    inv_argsort = np.empty_like(arr)
    inv_argsort[arr] = np.arange(len(arr))
    return inv_argsort

def ret_cosine(a,b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


def fitness_function(model0, model1, model2, data, label, rng_key):
    model0.eval()
    model2.eval()
    batch_size = data.size(0)
    z = model0(data)  # z shape: (batch_size, 5)

    sorted_z_ind = run_inference(model1, z.detach().numpy(), rng_key)

    sorted_z_value2, sorted_z_ind2 = torch.sort(data[:, 5:] / data[:, :5], dim=-1, descending=True)
    reordered = torch.gather(data[:, :5] / 15, dim=1, index=torch.tensor(sorted_z_ind))
    k_mask, r = model2(torch.cumsum(reordered, dim = 1))
    final_value = build_fractional_mask(k_mask, r)
    sorted_z_ind = torch.tensor(sorted_z_ind)
    inverse_indices = torch.zeros_like(sorted_z_ind)
    inverse_indices.scatter_(1, sorted_z_ind, torch.arange(sorted_z_ind.size(1)).expand_as(sorted_z_ind))
    final_value = torch.gather(final_value, dim=1, index=inverse_indices)
    z_numpy = z.cpu().detach().numpy()

    inv_argsort = []
    for i in range(len(z_numpy)):
        inv_argsort.append(ret_cosine(sorted_z_ind[i], sorted_z_ind2[i])) 
    cos_score = np.mean(inv_argsort)
    loss = nn.L1Loss()(final_value.float(), (label == 1.0).float())

    return -loss.item(), cos_score 



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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



class FractionalKnapsackTwoStage(nn.Module):
    def __init__(self, d_model=32*4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=1),
            num_layers=2
        )
        self.k_head = nn.Linear(d_model, 1)   # [B, L] - logits for K
        self.r_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L = x.shape
        x_input = x.unsqueeze(-1)  #[B, L, 1]
        h = self.embed(x_input)    #[B, L, D]
        h = h.permute(1, 0, 2)     #[L, B, D]
        h = self.encoder(h)        #[L, B, D]
        h = F.relu(h.permute(1, 0, 2))     #[B, L, D]

        k_logits = self.k_head(h).squeeze(-1)       #[B, L]
        k_idx = torch.argmax(k_logits, dim=1)       #[B]
        k_mask = F.one_hot(k_idx, num_classes=L).float()  #[B, L]

        h_pooled = h.mean(dim=1)                    # [B, D]
        r = self.r_head(h_pooled).squeeze(-1)       # [B]

        return k_mask, r

def build_fractional_mask(k_mask, r):

    B, L = k_mask.shape
    
    one_index = torch.argmax(k_mask, dim=1)  # shape [B]

    mask = torch.arange(L).expand(B, L)
    full_mask = (mask <= one_index.unsqueeze(1)).float()

    return full_mask


def particle_swarm_optimization(model0, input_size, hidden_size, output_size, fstage_hidden_size, num_particles, num_iterations, data, label, model1, rng_key):
    particles = []
    velocities = []
    personal_best_positions = []
    personal_best_scores = []
    global_best_position = None
    global_best_score = float('-inf')
    global_best_cosine = -2
    best_fitness_score = -10000

    weights_test = np.load('dataset/feature_weight_val.npy')
    profits_test = np.load('dataset/feature_profit_val.npy')
    label_test = np.load('dataset/label_ind_val.npy')
    data_test = torch.tensor(np.concatenate((weights_test, profits_test), -1), dtype=torch.float32)
    label_test = torch.tensor(label_test, dtype=torch.float32)

    
    for _ in range(num_particles):
        mask_nn = FractionalKnapsackTwoStage(fstage_hidden_size)
        random_params = {}
        for name, param in mask_nn.named_parameters():
            random_shape = param.shape
            random_tensor = torch.randn(random_shape)  # Fresh random values
            random_params[name] = random_tensor

        particles.append(random_params)

        velocities.append({k: torch.zeros_like(v) for k, v in random_params.items()})
        personal_best_positions.append({k: v.clone() for k, v in random_params.items()})
        personal_best_scores.append(float('-inf'))
    
    w, c1, c2 = cfg['stage3']['w'], cfg['stage3']['c1'], cfg['stage3']['c2'] #0.5, 1.8, 1.8

    for iteration in range(num_iterations):
        for i in range(num_particles):
            
            mask_nn = FractionalKnapsackTwoStage(fstage_hidden_size)
            mask_nn.load_state_dict(particles[i])

            fitness, acc = fitness_function(model0, model1, mask_nn, data, label, rng_key)

            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = {k: v.clone() for k, v in particles[i].items()}

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_cosine = acc
                print(f"Iteration {iteration}: Particle={i}, Score={fitness:.5f}")
                global_best_position = {k: v.clone() for k, v in particles[i].items()}

                model0.eval()
                z = model0(data_test)
                sorted_z_ind = run_inference(model1, z.detach().numpy(), rng_key)

                
                sorted_z_value2, sorted_z_ind2 = torch.sort(data_test[:, 5:] / data_test[:, :5], dim=-1, descending=True)
                reordered = torch.gather(data_test[:, :5] / 15, dim=1, index=torch.tensor(sorted_z_ind))
                k_mask, r = mask_nn(torch.cumsum(reordered, dim = 1))
                k_list = torch.argmax(k_mask, dim = 1)
                _label_test = torch.gather(label_test, dim=1, index=torch.tensor(sorted_z_ind2))
                is_one = _label_test == 1.0
                is_one_reversed = is_one.flip(dims=[1])
                last_one_idx_reversed = torch.argmax(is_one_reversed.int(), dim=1)
                last_one_idx = label_test.shape[1] - 1 - last_one_idx_reversed #+ 1
                _fitness, _ = fitness_function(model0, model1, mask_nn, data_test, label_test, rng_key)
                print(f"    Val score={_fitness:.5f} acc={torch.sum(k_list == last_one_idx) / len(k_list)}")
                if _fitness > best_fitness_score:
                    print('saving...')
                    best_fitness_score = _fitness
                    os.makedirs('ainn_model', exist_ok = True)
                    state_dict = {'net': mask_nn.state_dict()}
                    torch.save(state_dict, 'ainn_model/model_{}_stage3_{}.pth'.format(args.config, num_sample))

                    

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

weights = np.load('dataset/feature_weight_train.npy')[:num_sample]
profits = np.load('dataset/feature_profit_train.npy')[:num_sample]
label = np.load('dataset/label_ind_train.npy')[:num_sample]
data = torch.tensor(np.concatenate((weights, profits), -1), dtype = torch.float32)
label = torch.tensor(label, dtype = torch.float32)



model = SimpleNN(input_size, hidden_size, output_size)
state_dict = torch.load(
        'ainn_model/model_{}_stage1_{}.pth'.format(args.config, num_sample), map_location='cpu'
    )
model.load_state_dict(state_dict['net'])

train_model, feedback, rng_key = main()
train_model.restore_model('model_{}_stage2_{}.pkl'.format(args.config, num_sample))


_ = particle_swarm_optimization(
    model, input_size, hidden_size, output_size, fstage_hidden_size, num_particles, num_iterations, data, label, train_model, rng_key)
