import torch
import torch.nn as nn
import sys
import os
import numpy as np
import random
import torch.nn.functional as F
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
num_particles = cfg['stage1']['num_particles'] #100
num_iterations = cfg['stage1']['num_iterations'] #50


def compute_profits(sorted_z_ind, weights, profits):
    batch_size = weights.size()[0]
    ind_value = torch.zeros((batch_size, 5))
    for i in range(batch_size):
        curr_ind = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        remaining_weight = 15
        for ind in sorted_z_ind[i]:
            if remaining_weight <= 0:
                break

            if weights[i][ind] <= remaining_weight:
                remaining_weight -= weights[i][ind]
                curr_ind[ind] = 1
            else:
                curr_ind[ind] = remaining_weight / weights[i][ind]
                remaining_weight = 0
                
        ind_value[i] = curr_ind 
    return ind_value


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

def fitness_function(model, data, label):
    model.eval()
    z = model(data)
    sorted_z_value, sorted_z_ind = torch.sort(z, dim = -1, descending = True)
    sorted_z_value2, sorted_z_ind2 = torch.sort(data[:,5:]/data[:,:5], dim = -1, descending = True)
    final_value = compute_profits(sorted_z_ind, data[:,:5], data[:,5:])
    z_numpy = z.cpu().detach().numpy()
    ratio_numpy = (data[:,5:] / data[:,:5]).cpu().detach().numpy()

    inv_argsort = []
    for i in range(len(z_numpy)):
        inv_argsort.append(ret_cosine(inverse_argsort(np.argsort(z_numpy[i])[::-1]), inverse_argsort(np.argsort(ratio_numpy[i])[::-1])))
        
    cos_score = np.mean(inv_argsort)

    loss = nn.L1Loss()(final_value.float(), label.float())
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


def particle_swarm_optimization(input_size, hidden_size, output_size, num_particles, num_iterations, data, label):
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
        w1 = torch.randn((hidden_size, 2), requires_grad=False)
        b1 = torch.randn((hidden_size,), requires_grad=False)
        w2 = torch.randn((1, hidden_size), requires_grad=False)
        b2 = torch.randn((1,), requires_grad=False)
        g_logits = torch.randn(4, requires_grad=False)

        particles.append((w1, b1, w2, b2, g_logits))
        velocities.append((torch.zeros_like(w1), torch.zeros_like(b1), torch.zeros_like(w2), torch.zeros_like(b2), torch.zeros_like(g_logits)))
        personal_best_positions.append((w1.clone(), b1.clone(), w2.clone(), b2.clone(), g_logits.clone()))
        personal_best_scores.append(float('-inf'))

    w, c1, c2 = cfg['stage1']['w'], cfg['stage1']['c1'], cfg['stage1']['c2'] #0.5, 1.8, 1.8

    for iteration in range(num_iterations):
        for i in range(num_particles):
            model = SimpleNN(input_size, hidden_size, output_size)
            w1, b1, w2, b2, g_logits = particles[i]

            model.shared_mlp[0].weight.data = w1
            model.shared_mlp[0].bias.data = b1
            model.shared_mlp[2].weight.data = w2
            model.shared_mlp[2].bias.data = b2
            model.shared_gate_logits.data = g_logits

            fitness, acc = fitness_function(model, data, label)

            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = (w1.clone(), b1.clone(), w2.clone(), b2.clone(), g_logits.clone())

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_cosine = acc
                print(f"Iteration {iteration}: Particle={i}, Score={fitness:.5f}")
                global_best_position = (w1.clone(), b1.clone(), w2.clone(), b2.clone(), g_logits.clone())

                model.eval()
                _fitness, _ = fitness_function(model, data_test, label_test)
                print(f"    Val score:{_fitness:.5f}")
                if _fitness > best_fitness_score:
                    print('saving...')
                    state_dict = {'net': model.state_dict()}
                    os.makedirs('ainn_model', exist_ok = True)
                    torch.save(state_dict, 'ainn_model/model_{}_stage1_{}.pth'.format(args.config, num_sample))
                    best_fitness_score = _fitness

        for i in range(num_particles):
            v1, v2, v3, v4, v5 = velocities[i]
            p1, p2, p3, p4, p5 = particles[i]
            g1, g2, g3, g4, g5 = global_best_position
            pb1, pb2, pb3, pb4, pb5 = personal_best_positions[i]
            new_v1 = w * v1 + c1 * torch.rand_like(p1) * (pb1 - p1) + c2 * torch.rand_like(p1) * (g1 - p1)
            new_v2 = w * v2 + c1 * torch.rand_like(p2) * (pb2 - p2) + c2 * torch.rand_like(p2) * (g2 - p2)
            new_v3 = w * v3 + c1 * torch.rand_like(p3) * (pb3 - p3) + c2 * torch.rand_like(p3) * (g3 - p3)
            new_v4 = w * v4 + c1 * torch.rand_like(p4) * (pb4 - p4) + c2 * torch.rand_like(p4) * (g4 - p4)
            new_v5 = w * v5 + c1 * torch.rand_like(p5) * (pb5 - p5) + c2 * torch.rand_like(p5) * (g5 - p5)
            velocities[i] = (new_v1, new_v2, new_v3, new_v4, new_v5)
            particles[i] = (p1 + new_v1, p2 + new_v2, p3 + new_v3, p4 + new_v4, p5 + new_v5)

    return global_best_position

weights = np.load('dataset/feature_weight_train.npy')[:num_sample]
profits = np.load('dataset/feature_profit_train.npy')[:num_sample]
label = np.load('dataset/label_ind_train.npy')[:num_sample]
data = torch.tensor(np.concatenate((weights, profits), -1), dtype = torch.float32)
label = torch.tensor(label, dtype = torch.float32)
model = SimpleNN(input_size, hidden_size, output_size)


_ = particle_swarm_optimization(
    input_size, hidden_size, output_size, num_particles, num_iterations, data, label)
