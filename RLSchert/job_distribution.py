import numpy as np
import pandas as pd
import pickle5 as pickle
import math

class Dist:

    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.job_len = job_len
        self.max_nw_size = max_nw_size
        f = open('final_vasp_data.pkl','rb')
        init_data = pickle.load(f)
        self.data = init_data.values
        self.all_data_size = self.data.shape[0]
        print(self.all_data_size)

    def bi_model_dist(self):
        idx = np.random.randint(0,self.all_data_size)
        nw_len = int(self.data[idx, 1])
        nw_len_para = min(int(self.data[idx, 2]), 100)
        nw_len_running = np.zeros(self.job_len).astype(np.int32)
        nw_len_running[:self.data[idx, 3].shape[0]] = self.data[idx, 3].astype(np.int32)
        nw_len_running[self.data[idx, 3].shape[0]:] = self.data[idx, 3].astype(np.int32)[-1]
        nw_len_running = np.clip(nw_len_running, 0, 105)
        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = min(math.ceil(self.data[idx, 4]/10), 10)

        return nw_len, nw_len_para, nw_len_running, nw_size


def generate_sequence_work(pa, seed):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_len_para_seq = np.zeros(simu_len, dtype=int)
    nw_len_running_seq = np.zeros((simu_len, pa.max_job_len), dtype=int)

    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)

    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate:  # a new job comes

            nw_len_seq[i], nw_len_para_seq[i], nw_len_running_seq[i], nw_size_seq[i, :] = nw_dist()

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    print('total:'+str(np.sum(nw_len_seq)))
    nw_len_para_seq = np.reshape(nw_len_para_seq,
                            [pa.num_ex, pa.simu_len])
    nw_len_running_seq = np.reshape(nw_len_running_seq,
                            [pa.num_ex, pa.simu_len, pa.max_job_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])

    return nw_len_seq, nw_len_para_seq, nw_len_running_seq, nw_size_seq
