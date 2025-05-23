import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):
        self.cuda = False
        self.pred_err = 0.0
        self.small_err = 0.0
        self.n_step = 10
        self.ppo_batch = 256
        self.ppo_epochs = 5
        self.ppo_tau = 0.95
        self.SMALL_NUM = 1e-8
        self.target_kl = 0.01

        self.output_filename = 'data/tmp'

        self.num_epochs = 10000         # number of training epochs
        self.simu_len = 10             # length of the busy cycle that repeats itself
        self.num_ex = 1                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 3000  # enforcing an artificial terminal

        self.num_res = 1               # number of resources in the system
        self.num_nw = 5                # maximum allowed number of work in the queue

        self.time_horizon = 120         # number of time steps in the graph
        self.max_job_len = 100         # maximum duration of new jobs
        self.res_slot = 20             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work
        self.max_run_job = 8 # maximum running jobs
        self.running_feat = 15

        self.backlog_size = 600         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 500          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1          # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = 0 #int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.running_width = 1 # running job feature
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + self.running_width + \
            1  # for extra info, 1) time since last new job

        self.network_input_width_t = (self.res_slot +self.max_job_size * self.num_nw) * self.num_res + self.backlog_width + 1

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + self.max_run_job + 1  # + 1 for void action
        self.network_output_dim_t = self.num_nw + 1

        self.delay_penalty = - 0.1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 1e-5          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = 0 #int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + self.running_width + \
            1  # for extra info, 1) time since last new job
        self.network_input_width_t = (self.res_slot +self.max_job_size * self.num_nw) * self.num_res + self.backlog_width + 1

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator
        self.network_output_dim_t = self.num_nw + 1

        self.network_output_dim = self.num_nw + self.max_run_job + 1  # + 1 for void action

