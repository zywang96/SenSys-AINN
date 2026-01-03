import numpy as np
import sys
from scipy.stats.stats import pearsonr


mode = 'train'
_c_list = np.load('intermediate/{}_c_list_le.npy'.format(mode))
_dp_list = np.load('intermediate/{}_dp_list_le.npy'.format(mode))
_dp_ref_list = np.load('intermediate/{}_dp_ref_list_le.npy'.format(mode))
n_items = len(_c_list)
#print(n_items)

np.random.seed(0)
random_indices = np.random.choice(n_items, size=2000, replace=False)

_c_list = _c_list[random_indices]
_dp_list = _dp_list[random_indices]
_dp_ref_list = _dp_ref_list[random_indices]


#print(np.shape(_c_list))

correct_sm = []
ours = []
mins = []
for idx in range(len(_c_list))[:]:
    for _i in range(48):
        dp_mat = _dp_list[idx][_i][1:,1:]
        tmp_mat = _dp_list[idx][_i][1:,1:] - 1 * _c_list[idx][_i]

        s = 0
        c, t = 0, 0
        for i in range(1, len(tmp_mat)):
            for j in range(1, len(tmp_mat)):
                t0 = dp_mat[i - 1, j]
                t1 = dp_mat[i, j - 1]
                t2 = dp_mat[i - 1, j - 1]
                ours.append(tmp_mat[i, j])
                mins.append(min(t2, min(t1, t0)))
                s += np.abs(tmp_mat[i, j] - min(t2, min(t1, t0)))
                t += 1

coefficients = np.polyfit(mins, ours, 1)
m, c = coefficients[0], coefficients[1]
print('m: ', m, ' c: ', c)