import torch
import torch.nn as nn
from uav2_charge2.exper_dppo_curiosity.model import ICMModel


# params = Params()


class ICMAgent(object):
    def __init__(self, obs_shape, uav_num, eta, action_dim, device):
        self.device = device
        self.icm = ICMModel(obs_shape, action_dim, uav_num)
        self.icm.to(device)
        self.eta = eta
        self.uav_num = uav_num
        self.action_dim = action_dim
        self.cross_entropy = nn.CrossEntropyLoss()

    def convert_action(self, action):
        # batch_num = action[1].size()[0]
        # action_charge = action[0].clone()
        # action_charge_onehot = torch.eye(2 ** self.uav_num)[action_charge].view(batch_num, -1).to(self.device)

        action_navigate = action[1].clone()
        action_navigate_onehot = torch.eye(self.action_dim)[action_navigate].to(self.device)
        action_navigate_onehot = torch.cat([a for a in action_navigate_onehot], dim=-1)
        # convert_action = torch.cat((action_charge_onehot, action_navigate_onehot), dim=-1)
        # return convert_action
        return action_navigate_onehot

    def compute_forward_loss(self, pred_next_state_feature, real_next_state_feature):
        return (pred_next_state_feature - real_next_state_feature).pow(2).sum(-1, keepdim=True) / 2

    def compute_intrinsic_reward(self, state, next_state, action):
        action_one_hot = self.convert_action(action)

        real_next_state_feature, pred_next_state_feature, _ = self.icm(state, next_state, action_one_hot)
        with torch.no_grad():
            intrinsic_reward = self.eta * self.compute_forward_loss(pred_next_state_feature, real_next_state_feature)
        return intrinsic_reward

    def compute_inverse_loss(self, pred_logits, real_action):
        loss = 0
        for i in range(self.uav_num):
            tmp_pred_logits = pred_logits[:, i * self.action_dim:i * self.action_dim + self.action_dim]
            tmp_real_action = real_action[i]
            loss += self.cross_entropy(tmp_pred_logits, tmp_real_action) / self.action_dim
        # loss /= self.uav_num
        # return self.cross_entropy(pred_logits, real_action) / self.action_dim
        return loss

    def compute_loss(self, state, next_state, action):
        action_one_hot = self.convert_action(action)
        real_next_state_feature, pred_next_state_feature, action_features = self.icm(state, next_state, action_one_hot)
        forward_loss = self.compute_forward_loss(pred_next_state_feature, real_next_state_feature.detach()).mean()
        inverse_loss = self.compute_inverse_loss(action_features, action[1].squeeze().detach()).mean()
        return inverse_loss, forward_loss

    def update(self, shared_icm_model_state_dict):
        self.icm.load_state_dict(shared_icm_model_state_dict)
