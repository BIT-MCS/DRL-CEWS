from uav2_charge2.exper_dppo_curiosity.model import RNDModel


class RndAgent(object):
    def __init__(self, obs_shape, eta, device):
        self.icm = RNDModel(obs_shape, device)
        self.eta = eta

    def _compute_forward_loss(self, target_feature, predict_feature):
        forward_loss = (target_feature - predict_feature).pow(2).sum(-1, keepdim=True) / 2
        return forward_loss

    def compute_intrinsic_reward(self, obs, next_obs, action):
        predict_feature, target_feature = self.icm(next_obs)
        intrinsic_reward = self.eta * self._compute_forward_loss(target_feature.detach(), predict_feature.detach())
        return intrinsic_reward

    def compute_loss(self, obs, next_obs, action):
        predict_feature, target_feature = self.icm(next_obs)
        forward_loss = self._compute_forward_loss(target_feature, predict_feature).mean()
        return 0, forward_loss
