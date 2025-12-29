import torch
import torch.nn as nn
import sys
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
num_particles = cfg['stage2']['num_particles'] #100
num_iterations = cfg['stage2']['num_iterations'] #50

def compute_profits(sorted_z_ind, weights, profits):
    batch_size = weights.size()[0]
    ind_value = torch.zeros((batch_size, 5)) #.to('cuda')
    for i in range(batch_size):
        curr_ind = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32) #.to('cuda')
        remaining_weight = 15
        for ind in sorted_z_ind[i]:
            if remaining_weight <= 0:
                break

            if weights[i][ind] <= remaining_weight:
                remaining_weight -= weights[i][ind]
                curr_ind[ind] = 1
            else:
                curr_ind[ind] = remaining_weight / weights[i][ind]  #.to('cuda')
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


def fitness_function(model0, model1, data, label, rng_key):
    model0.eval()
    batch_size = data.size(0)
    z = model0(data)  # z shape: (batch_size, 5)

    sorted_z_ind = run_inference(model1, z.detach().numpy(), rng_key)

    sorted_z_value2, sorted_z_ind2 = torch.sort(data[:, 5:] / data[:, :5], dim=-1, descending=True)
    final_value = compute_profits(torch.tensor(sorted_z_ind).long(), data[:, :5], data[:, 5:])

    z_numpy = z.cpu().detach().numpy()

    inv_argsort = []
    for i in range(len(z_numpy)):
        inv_argsort.append(ret_cosine(sorted_z_ind[i], sorted_z_ind2[i]))
    cos_score = np.mean(inv_argsort)

    # --- Final loss ---
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

def particle_swarm_optimization(model0, input_size, hidden_size, output_size, num_particles, num_iterations, data, label, model1, rng_key):
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
        random_params = {}
        for name in model1.params:
            for sub_name in model1.params[name]:
                param = model1.params[name][sub_name]
                random_shape = param.shape
                random_params[name + ':' + sub_name] = torch.tensor(param) + 0.01 * torch.randn(random_shape)

        particles.append(random_params)

        velocities.append({k: torch.zeros_like(v) for k, v in random_params.items()})
        personal_best_positions.append({k: v.clone() for k, v in random_params.items()})
        personal_best_scores.append(float('-inf'))
    
    w, c1, c2 = cfg['stage2']['w'], cfg['stage2']['c1'], cfg['stage2']['c2'] #0.5, 1.8, 1.8

    for iteration in range(num_iterations):
        for i in range(num_particles):
            
            params = unfreeze(model1.params)

            for name in particles[i]:
                fname, lname = name.split(':')
                params[fname][lname] = particles[i][name].detach().numpy()
            model1.params = freeze(params)

            fitness, acc = fitness_function(model0, model1, data, label, rng_key)
            if fitness > personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = {k: v.clone() for k, v in particles[i].items()}

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_cosine = acc
                print(f"Iteration {iteration}: Particle={i}, Score={fitness:.5f}")
                global_best_position = {k: v.clone() for k, v in particles[i].items()}

                model0.eval()
                _fitness, _ = fitness_function(model0, model1, data_test, label_test, rng_key) 
                print(f"    Val score:{_fitness:.5f}")
                if _fitness > best_fitness_score:
                    print('saving...')
                    best_fitness_score = _fitness

                    model1.save_model('model_{}_stage2_{}.pkl'.format(args.config, num_sample))
                    
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
_ = particle_swarm_optimization(
    model, input_size, hidden_size, output_size, num_particles, num_iterations, data, label, train_model, rng_key)
