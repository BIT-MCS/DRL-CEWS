import torch
from torch.autograd import Variable
import copy
import numpy as np

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def prRed(prt): print("\033[91m {}\033[00m".format(prt))


def prGreen(prt): print("\033[92m {}\033[00m".format(prt))


def prYellow(prt): print("\033[93m {}\033[00m".format(prt))


def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))


def prPurple(prt): print("\033[95m {}\033[00m".format(prt))


def prCyan(prt): print("\033[96m {}\033[00m".format(prt))


def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))


def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


class Util:
    def __init__(self, device):
        self.device = device
        self.USE_CUDA = True if 'cuda' in device.type else False

    def to_numpy(self, var, is_deep_copy=True):

        # list type [ Tensor, Tensor ]
        if isinstance(var, list) and len(var) > 0:
            var_ = []
            for v in var:
                temp = v.cpu().data.numpy() if self.USE_CUDA else v.data.numpy()

                # this part is meaningless if Tensor is in gpu
                if is_deep_copy:
                    var_.append(copy.deepcopy(temp))
            return var_

        # dict type { key, Tensor }
        if isinstance(var, dict) and len(var) > 0:
            var_ = {}
            for k, v in var.iteritems():
                temp = v.cpu().data.numpy() if self.USE_CUDA else v.data.numpy()

                # this part is meaningless if Tensor is in gpu
                if is_deep_copy:
                    var_[k] = copy.deepcopy(temp)
            return var_

        var = var.cpu().data.numpy() if self.USE_CUDA else var.data.numpy()
        # this part is meaningless if Tensor is in gpu
        if is_deep_copy:
            var = copy.deepcopy(var)
        return var

    def to_tensor(self, ndarray, requires_grad=False, dtype=FLOAT, is_deep_copy=True):
        if ndarray is None:
            return ndarray

        # this part is meaningless if tensor is in gpu
        if is_deep_copy:
            ndarray = copy.deepcopy(ndarray)

        if isinstance(ndarray, list) and len(ndarray) > 0:
            var_ = []
            for v in ndarray:
                temp = torch.from_numpy(v).type(dtype)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_.append(temp)
            return var_
        if isinstance(ndarray, dict) and len(ndarray) > 0:
            var_ = {}
            for k, v in ndarray.iteritems():
                temp = torch.from_numpy(v).type(dtype)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_[k] = temp

            return var_

        ndarray = torch.from_numpy(ndarray).type(dtype)
        ndarray = ndarray.to(self.device)
        ndarray.requires_grad = requires_grad

        return ndarray

    def to_int_tensor(self, ndarray, requires_grad=False, is_deep_copy=True):
        if ndarray is None:
            return ndarray

        # this part is meaningless if tensor is in gpu
        if is_deep_copy:
            ndarray = copy.deepcopy(ndarray)

        if isinstance(ndarray, list) and len(ndarray) > 0:
            var_ = []
            for v in ndarray:
                temp = torch.from_numpy(v)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_.append(temp)
            return var_
        if isinstance(ndarray, dict) and len(ndarray) > 0:
            var_ = {}
            for k, v in ndarray.iteritems():
                temp = torch.from_numpy(v)
                temp = temp.to(self.device)
                temp.requires_grad = requires_grad
                var_[k] = temp

            return var_

        ndarray = torch.from_numpy(ndarray)
        ndarray = ndarray.to(self.device)
        ndarray.requires_grad = requires_grad

        return ndarray


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu
        np.random.seed(123456)

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx

        return self.X


import torch.multiprocessing as mp


class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)


class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0
