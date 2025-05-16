import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict

def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models

    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()

class PGLearner:
    def __init__(self, pa):
        self.cuda = pa.cuda
        self.epsilon = 0.2
        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.input_width_t = pa.network_input_width_t
        self.output_height = pa.network_output_dim
        self.output_height_t = pa.network_output_dim_t

        self.num_frames = pa.num_frames

        self.update_counter = 0

        print('network_input_height=' + str(pa.network_input_height))
        print('network_input_width=' + str(pa.network_input_width))
        print('network_input_width_t=' + str(pa.network_input_width_t))
        print('network_output_dim=' + str(pa.network_output_dim))
        print('network_output_dim_t=' + str(pa.network_output_dim_t))

        # image representation
        self.l_out = \
            build_pg_network(pa.network_input_height, pa.network_input_width, pa.network_output_dim)
        self.t_out = build_pg_network_t(pa.network_input_height, pa.network_input_width_t, pa.network_output_dim_t)
        if self.cuda: 
            self.l_out = self.l_out.cuda()
            self.t_out = self.t_out.cuda()

        # compact representation
        # self.l_out = \
        #     build_compact_pg_network(pa.network_input_height, pa.network_input_width, pa.network_output_dim)

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        #self.optimizer = optim.RMSprop(self.l_out.parameters(), lr=self.lr_rate, alpha=self.rms_rho)
        self.optimizer = optim.Adam(self.l_out.parameters(), lr=self.lr_rate, eps=1e-5)

    # get the action based on the estimated value
    def choose_action(self, state):

        act_prob, value, teacher_prob = self.get_one_act_prob(state)
        act = np.random.choice(np.arange(act_prob.shape[0]), 1, p=act_prob)
        teacher = np.random.choice(np.arange(teacher_prob.shape[0]), 1, p=teacher_prob)

        return act[0], value, teacher[0]

    def train(self, states, actions, teacher_actions, old_log_pis, values, advs):
        var_states = Variable(torch.from_numpy(states.astype(np.float32)))
        var_actions = Variable(torch.from_numpy(actions.astype(np.int64)))
        var_teacher_actions = Variable(torch.from_numpy(teacher_actions.astype(np.int64)))
        var_old_log_pis = Variable(torch.from_numpy(old_log_pis.astype(np.float32)))
        var_values = Variable(torch.from_numpy(values.astype(np.float32)))
        var_advs = Variable(torch.from_numpy(advs.astype(np.float32)))
        if self.cuda:
            var_states = var_states.cuda()
            var_actions = var_actions.cuda()
            var_teacher_actions = var_teacher_actions.cuda()
            var_old_log_pis = var_old_log_pis.cuda()
            var_values = var_values.cuda()
            var_advs = var_advs.cuda()
        policies, vs = self.l_out(var_states)
        probs = Fnn.softmax(policies, dim=-1)
        log_probs = Fnn.log_softmax(policies, dim=-1)
        entropy_loss = (log_probs * probs).sum(-1).mean()
        log_action_probs = log_probs.gather(1, var_actions)
        log_action_teacher_probs = log_probs.gather(1, var_teacher_actions)
        ratio = torch.exp(log_action_probs - var_old_log_pis)
        ratio = ratio.view((-1,1))
        policy_loss = -torch.mean(torch.min(ratio * var_advs, torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * var_advs))
        value_loss = Fnn.smooth_l1_loss(vs.squeeze(), var_values.squeeze())
        #value_loss = (.5 * (vs.squeeze() - var_values.squeeze()) ** 2.).mean()
        teacher_loss = -torch.mean(log_action_teacher_probs)
        loss = policy_loss + 0.25 * value_loss
        p_loss = policy_loss.item()
        v_loss = value_loss.item()
        if self.cuda: loss = loss.cuda()
        loss.backward()
        nn.utils.clip_grad_norm_(self.l_out.parameters(), 10.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        kl_curr = torch.sum(var_old_log_pis-log_action_probs)
        kl_curr = kl_curr.item()
        return loss.item(), kl_curr, p_loss, v_loss

    def get_one_act_prob(self, state):
        states = np.zeros((1, 1, self.input_height, self.input_width))
        states[0, :, :] = state
        print(state[1][0])
        act_prob, value, teacher_prob = self._get_act_prob(states)

        return act_prob[0].reshape(-1), value[0,0], teacher_prob[0].reshape(-1)


    def get_act_probs(self, states):  # multiple states, assuming in floatX format
        act_probs, values, teacher_prob = self._get_act_prob(states)
        return act_probs, values, teacher_prob

    def _get_act_prob(self, states):
        states_t = states.copy()
        states = Variable(torch.from_numpy(states.astype(np.float32)))
        states_t = Variable(torch.from_numpy(np.delete(states_t.astype(np.float32), -2, axis=-1)))
        if self.cuda: 
            states = states.cuda()
            states_t = states_t.cuda()
        act_probs, values = self.l_out(states)
        act_probs = Fnn.softmax(act_probs, dim=-1).data.cpu().numpy()
        teacher_probs, _ = self.t_out(states_t)
        teacher_probs = Fnn.softmax(teacher_probs, dim=-1).data.cpu().numpy()
        return act_probs, values.data.cpu().numpy(), teacher_probs

    #  -------- Supervised Learning --------
    #  -------- Save/Load network parameters --------
    def set_net_params(self, net_params):
        self.l_out.load_state_dict(net_params)

    def set_t_net_params(self, net_params):
        self.t_out.load_state_dict(net_params)


# ===================================
# build neural network
# ===================================

class PolicyCNNNet(nn.Module):
    def __init__(self, h_size, w_size, out_size):
        super().__init__()
        self.h_size = h_size
        self.w_size = w_size
        self.out_size = out_size
        self.conv = nn.Sequential(nn.Conv2d(1, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU(),)
        cnn_seq = [(0,1,4,2),(0,1,3,1)]
        self.h_out, self.w_out = self.cnn_to_fc(h_size,w_size,cnn_seq)
        self.fc = nn.Sequential(nn.Linear(64*self.h_out*self.w_out, 32),
                                nn.ReLU(),)
        self.pi = nn.Linear(32, out_size)
        self.v = nn.Linear(32, 1)

        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, states):
        N = states.size()[0]

        flat = self.conv(states).view(N, 64 * self.h_out * self.w_out)
        fc_out = self.fc(flat)
        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)
        
        return pi_out, v_out

    def cnn_to_fc(self, h_in, w_in, seq):
        h_out = h_in
        w_out = w_in
        for c in seq:
            h_out = int((h_out+2.0*c[0]-c[1]*(c[2]-1)-1)/c[3]+1)
            w_out = int((w_out+2.0*c[0]-c[1]*(c[2]-1)-1)/c[3]+1)
        return h_out, w_out

class PolicyNet(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Sequential(nn.Linear(in_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),)
        self.pi = nn.Linear(128, out_size)
        self.v = nn.Linear(128, 1)

        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def reset_v(self, cuda):
        print('reset value net')
        if cuda: self.v.weight.data = ortho_weights(self.v.weight.size()).cuda()
        else: self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, states):
        N = states.size()[0]

        flat = states.view(N, self.in_size)
        flat = self.fc(flat)

        pi_out = self.pi(flat)
        v_out = self.v(flat)

        return pi_out, v_out

class CompactPolicyNet(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.pi = nn.Sequential(nn.Linear(in_size, 520),
                                nn.ReLU(),
                                nn.Linear(520, 20),
                                nn.ReLU(),
                                nn.Linear(20, 20),
                                nn.ReLU(),
                                nn.Linear(20, out_size))

    def forward(self, states):
        N = states.size()[0]

        flat = states.view(N, self.in_size)

        pi_out = self.pi(flat)

        return pi_out

class PolicyTeacher(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Sequential(nn.Linear(in_size, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),)
        self.pi = nn.Linear(64, out_size)
        self.v = nn.Linear(64, 1)

        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, states):
        N = states.size()[0]

        flat = states.view(N, self.in_size)
        flat = self.fc(flat)

        pi_out = self.pi(flat)
        v_out = self.v(flat)

        return pi_out, v_out

def build_pg_cnn_network(input_height, input_width, output_length):
    return PolicyCNNNet(input_height, input_width, output_length)

def build_pg_network(input_height, input_width, output_length):
    return PolicyNet(input_height*input_width, output_length)

def build_pg_network_t(input_height, input_width, output_length):
    return PolicyTeacher(input_height*input_width, output_length)

def build_compact_pg_network(input_height, input_width, output_length):
    return CompactPolicyNet(input_height*input_width, output_length)
