import torch
import torch.nn as nn
import torch.distributions
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from uav2_charge2.exper_dppo_curiosity.utils import AddBias, init, init_normc_
from torch.distributions import Categorical

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class Categorical_1d(nn.Module):
    def __init__(self, num_inputs, num_outputs, device):
        super(Categorical_1d, self).__init__()
        self.device = device
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = nn.Sequential(
            init_(nn.Linear(num_inputs, num_outputs)),
        )

        self.dis_cat = None
        self.logits = 0
        self.sampler = Categorical
        self.weighted_sampler = WeightedRandomSampler
        self.train()

    def forward(self, x):
        x = self.linear(x)
        self.dis_cat = Categorical(logits=x)
        self.logits = self.dis_cat.logits
        return self.dis_cat

    def sample(self):
        return self.dis_cat.sample()

    def gumbel_softmax_sample(self, tau):
        dist = F.gumbel_softmax(self.logits, tau=tau, hard=False)
        action = torch.tensor(list(self.weighted_sampler(dist, 1, replacement=False))).to(self.device)
        return action.squeeze(-1)

    def log_probs(self, action):
        return self.dis_cat.log_prob(action.squeeze(-1)).unsqueeze(-1)

    def entropy(self):
        return self.dis_cat.entropy()


class _Categorical(Categorical):
    """
    a son class inherit from class torch.distributions.Categorical
    it adds a gumbel softmax sample method, for gumbel softmax sample
    and a mode method for argmax sample
    """

    def __init__(self, _logits):
        super(_Categorical, self).__init__(logits=_logits)
        self._logits = self.logits
        self.weighted_sampler = WeightedRandomSampler

    def gumbel_softmax_sample(self, tau, device):
        dist = F.gumbel_softmax(self._logits, tau=tau, hard=False)
        action = torch.tensor(list(self.weighted_sampler(dist, 1, replacement=False))).to(device)
        return action.squeeze(-1)

    def mode(self):
        return torch.argmax(self._logits, dim=-1, keepdim=False)


class MultiHeadCategorical(nn.Module):
    """
    define a multi-head Categorical for multi-label classification
    --init:
    num_inputs: input feature dim
    dim_vec: a list for dim of each action space, e.g. [2,3,5], 2-dim for action1, 3-dim for action2, 5-dim for action3
    device: running device
    --forward:
    inputs: flatten input feature
    """

    # @torchsnooper.snoop()
    def __init__(self, num_inputs, action_num, action_dim, device):
        super(MultiHeadCategorical, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.linear_list = torch.nn.ModuleList(
            [init_(nn.Linear(num_inputs, action_dim).to(device)) for _ in range(action_num)])
        self.logits_head = []
        self.weight_sample = WeightedRandomSampler
        self.device = device
        self.categorical_list = []
        self.train()

    def forward(self, inputs):
        self.categorical_list = [_Categorical(linear(inputs)) for linear in self.linear_list]

    def gumbel_softmax_sample(self, tau):
        action = torch.cat([p.gumbel_softmax_sample(tau, self.device) for p in self.categorical_list])
        return action

    def log_probs(self, action):
        return torch.cat([p.log_prob(a).unsqueeze(-1) for a, p in zip(action, self.categorical_list)], dim=-1)

    def mode(self):
        return torch.cat([p.mode() for p in self.categorical_list])

    def entropy(self):
        return torch.cat([p.entropy() for p in self.categorical_list])
