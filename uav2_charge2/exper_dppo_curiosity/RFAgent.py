import torch.nn as nn
from uav2_charge2.exper_dppo_curiosity.model import RFModel
import torch


class RFAgent(object):
    def __init__(self, uav_num, eta, device):
        self.device = device
        self.eta = eta
        self.icm = RFModel(uav_num, device)
        self.icm.to(device)
        self.uav_num = uav_num

    def convert_action(self, action):
        batch_num = action[0].size()[0]
        action_charge = action[0].clone()
        action_charge_onehot = torch.eye(2 ** self.uav_num)[action_charge].view(batch_num, -1).to(self.device)

        action_navigate = action[1].clone()
        action_navigate_onehot = torch.eye(25)[action_navigate].to(self.device)
        action_navigate_onehot = torch.cat([a for a in action_navigate_onehot], dim=-1)
        convert_action = torch.cat((action_charge_onehot, action_navigate_onehot), dim=-1)
        return convert_action

    def compute_intrinsic_reward(self, state, next_state, action):
        action_one_hot = self.convert_action(action)
        real_next_state_feature, pred_next_state_feature = self.icm(state, next_state, action_one_hot)
        with torch.no_grad():
            intrinsic_reward = self.eta * self.icm.compute_forward_loss(pred_next_state_feature, real_next_state_feature)
        return intrinsic_reward

    def compute_loss(self, state, next_state, action):
        action_one_hot = self.convert_action(action)
        real_next_state_feature, pred_next_state_feature = self.icm(state, next_state, action_one_hot)
        forward_loss = self.icm.compute_forward_loss(pred_next_state_feature, real_next_state_feature.detach()).mean()
        return 0, forward_loss

