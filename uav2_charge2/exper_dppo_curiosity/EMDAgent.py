from uav2_charge2.exper_dppo_curiosity.model import EmbeddingModel
import torch
import torch.nn as nn


class EMDAgent(object):
    def __init__(self, uav_num, eta, action_dim, device, rank):
        self.device = device
        self.icm = EmbeddingModel(action_dim, device)
        self.icm.to(device)
        self.eta = eta
        self.uav_num = uav_num
        self.action_dim = action_dim
        self.rank = rank

    def convert_action(self, action):
        # change action to batch first
        action_new = action.transpose(0, 1)
        action_new = action_new.contiguous().view([-1])
        action_navigate_onehot = torch.eye(self.action_dim)[action_new].to(self.device)
        return action_navigate_onehot

    def compute_intrinsic_reward(self, state, next_state, action):
        action_one_hot = self.convert_action(action)
        with torch.no_grad():
            intrinsic_reward = self.eta * self.icm(state, next_state, action_one_hot).view(-1, self.uav_num).mean(
                -1, keepdim=True)
        return intrinsic_reward

    def compute_loss(self, state, next_state, action):
        action_one_hot = self.convert_action(action)
        forward_loss = self.icm(state, next_state, action_one_hot).mean()
        return 0, forward_loss

    def update(self, shared_icm_model_state_dict):
        self.icm.load_state_dict(shared_icm_model_state_dict)
