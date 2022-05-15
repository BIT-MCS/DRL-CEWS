from uav2_charge2.exper_dppo_curiosity.distributions import Categorical_1d, MultiHeadCategorical
from conv_map.new_conv_map_cell_2d_brd_f1_l21_s6 import *
from util.utils import Counter


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, obs_shape, num_of_uav, cat_ratio, dia_ratio, device, action_dim, trainable=True, hidsize=512):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.num_of_uav = num_of_uav
        # feature extract
        self.base = CNNBase(obs_shape[0], device, trainable)
        # actor
        self.cat_ratio = cat_ratio
        self.dia_ratio = dia_ratio
        # discreet,show which to charge
        self.dist_cat = Categorical_1d(hidsize, 2 ** num_of_uav, device)
        # discrete, show uavs (dx, dy)
        self.dist_dia = MultiHeadCategorical(hidsize, num_of_uav, action_dim, device)

        # critic
        self.critic = nn.Sequential(
            init_(nn.Linear(hidsize, 1))
        )
        self.device = device
        if trainable:
            self.train()
        else:
            self.eval()

    def act(self, inputs, tau):
        with torch.no_grad():
            obs_feature = self.base(inputs)

            value = self.critic(obs_feature)
            self.dist_cat(obs_feature)
            self.dist_dia(obs_feature)

            action_cat = self.dist_cat.gumbel_softmax_sample(tau)
            action_dia = self.dist_dia.gumbel_softmax_sample(tau)

            action_log_probs_cat = self.dist_cat.log_probs(action_cat).mean(-1, keepdim=True)
            action_log_probs_dia = self.dist_dia.log_probs(action_dia).mean(-1, keepdim=True)

            action_cat = action_cat.view(1, 1)
            action_dia = action_dia.view(1, self.num_of_uav)

        return value, (action_cat, action_dia), \
               (action_log_probs_cat * self.cat_ratio + action_log_probs_dia * self.dia_ratio)

    def get_value(self, inputs):
        obs_feature = self.base(inputs)
        value = self.critic(obs_feature)
        return value

    def evaluate_actions(self, inputs, action):
        obs_features = self.base(inputs)
        value = self.critic(obs_features)
        self.dist_cat(obs_features)
        self.dist_dia(obs_features)

        action_log_probs_cat = self.dist_cat.log_probs(action[0])
        action_log_probs_dia = self.dist_dia.log_probs(action[1]).mean(-1, keepdim=True)

        dist_entropy_cat = self.dist_cat.entropy().mean()
        dist_entropy_dia = self.dist_dia.entropy().mean()

        return value, self.dia_ratio * action_log_probs_dia + self.cat_ratio * action_log_probs_cat, \
               dist_entropy_dia * self.dia_ratio + dist_entropy_cat * self.cat_ratio

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


class RFModel(nn.Module):
    def __init__(self, num_of_uav, device):
        super(RFModel, self).__init__()
        self.base = CNNBase(3, device, trainable=False)
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        # define forward network:
        # input: cur one-hot action and cur state
        # output: predict next encode state
        action_shape = 2 ** num_of_uav + 25 * num_of_uav
        self.forward_net = nn.Sequential(
            init_(nn.Linear(action_shape + self.base.output_size, 512)),
            nn.ReLU(inplace=True),
            init_(nn.Linear(512, self.base.output_size))
        )
        # hd_size = 256
        # self.res_layer1 = nn.Sequential(
        #     init_(nn.Linear(action_shape + self.base.output_size, hd_size)),
        #     nn.LeakyReLU(),
        # )
        # self.res_layer2 = init_(nn.Linear(action_shape + hd_size, self.base.output_size))
        #
        # self.forward_net = nn.Sequential(
        #     init_(nn.Linear(action_shape + self.base.output, self.baes.output_size))
        # )

    def res(self, x, action):
        residual = self.res_layer1(torch.cat((x, action), dim=-1))
        residual = self.res_layer2(torch.cat((residual, action), dim=-1))
        return x + residual

    def forward(self, cur_state, next_state, action):
        encode_cur_state = self.base(cur_state)
        encode_next_state = self.base(next_state)
        # detach(), 防止更新base网络参数
        output = encode_cur_state.detach()
        # for _ in range(4):
        #     output = self.res(output, action)
        pred_next_state_feature = self.forward_net(torch.cat((output, action), dim=-1))

        return encode_next_state, pred_next_state_feature

    def compute_forward_loss(self, pred_next_state_feature, real_next_state_feature):
        # loss = (pred_next_state_feature - real_next_state_feature).pow(2).sum(-1, keepdim=True).sqrt() / 2
        loss = (pred_next_state_feature - real_next_state_feature).pow(2).sum(-1, keepdim=True) / 2
        return loss

    def print_grad(self):
        for n, p in self.named_parameters():
            print(n, p.requires_grad)
            if p.requires_grad is True:
                print(p.grad.data.mean())


class ICMModel(nn.Module):
    def __init__(self, obs_shape, action_dim, action_num):
        super(ICMModel, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        # input: 3* 80*80 output: 32 * 3 * 3 = 288
        self.state_feature = nn.Sequential(
            # 3 * 80 * 80
            nn.AvgPool2d(2, 2),
            # 3 * 40 * 40
            init_(nn.Conv2d(obs_shape[0], 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 20 * 20
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 10 * 10
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 5 * 5
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU()
        )

        # todo: change
        self.inverse_net = nn.Sequential(
            init_(nn.Linear(288 + 288, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, action_dim * action_num))
        )

        self.forward_net = nn.Sequential(
            init_(nn.Linear(288 + action_dim * action_num, 256)),
            nn.ReLU(),
            init_(nn.Linear(256, 288))
        )

    def forward(self, state, next_state, action):
        encode_state = self.state_feature(state)
        encode_next_state = self.state_feature(next_state)
        encode_state, encode_next_state = encode_state.view(-1, 288), encode_next_state.view(-1, 288)
        pred_action = self.inverse_net(torch.cat((encode_state, encode_next_state), -1))
        pred_next_state = self.forward_net(torch.cat((encode_state, action), -1))
        return encode_next_state, pred_next_state, pred_action


class EmbeddingModel(nn.Module):
    def __init__(self, action_dim, device, embedding_dim=8):
        super(EmbeddingModel, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.embedding = nn.Embedding(21 * 21, embedding_dim)

        self.forward_net = nn.Sequential(
            init_(nn.Linear(embedding_dim + action_dim, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, embedding_dim))
        ).to(device)

    def forward(self, cur_pos, next_pos, action):
        cur_pos = cur_pos.view([-1])
        next_pos = next_pos.view([-1])
        embedding_cur_pos = self.embedding(cur_pos)
        embedding_next_pos = self.embedding(next_pos)
        pred_next_pos = self.forward_net(torch.cat((embedding_cur_pos, action), dim=-1))
        loss = (pred_next_pos - embedding_next_pos).pow(2).sum(-1, keepdim=True) / 2
        return loss


class RNDModel(nn.Module):
    def __init__(self, obs_shape, device):
        super(RNDModel, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        # input: 3* 80*80 output: 32 * 3 * 3 = 288
        self.predictor = nn.Sequential(
            # 3 * 80 * 80
            nn.AvgPool2d(2, 2),
            # 3 * 40 * 40
            init_(nn.Conv2d(obs_shape[0], 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 20 * 20
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 10 * 10
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 5 * 5
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            Flatten(),
            nn.Linear(288, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ).to(device)

        self.target = nn.Sequential(
            # 3 * 80 * 80
            nn.AvgPool2d(2, 2),
            # 3 * 40 * 40
            init_(nn.Conv2d(obs_shape[0], 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 20 * 20
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 10 * 10
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            nn.ReLU(),
            # 32 * 5 * 5
            init_(nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            Flatten(),
            nn.Linear(288, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ).to(device)
        for params in self.target.parameters():
            params.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature


class CNNBase(nn.Module):
    def __init__(self, num_inputs, device, trainable=True, feature_size=512):
        super(CNNBase, self).__init__()
        self._feature_size = feature_size
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.feature = nn.Sequential(
            # input: 3*80*80
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.LayerNorm([32, 19, 19]),
            nn.ReLU(inplace=True),
            # input: 32*19*19
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(inplace=True),
            # input: 64*8*8
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.LayerNorm([32, 6, 6]),
            nn.ReLU(inplace=True),
            # output: 32*6*6

        ).to(device)
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.conv_to_flat = nn.Sequential(
            Flatten(),

            init_(nn.Linear(32 * 6 * 6, self._feature_size)),
            nn.LayerNorm([self._feature_size]),
            nn.ReLU(inplace=True),
        ).to(device)
        if trainable:
            self.train()
        else:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, inputs):
        x = self.feature(inputs)
        x = self.conv_to_flat(x)
        return x


class Shared_grad_buffers():
    def __init__(self, model, device):
        self.grads = {}
        self.counter = Counter()

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.grads[name + '_grad'] = torch.zeros(p.size()).share_memory_().to(device)

    def add_gradient(self, model):
        self.counter.increment()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.grads[name + '_grad'] += p.grad.data

    def average_gradient(self):
        counter_num = self.counter.get()
        for name, grad in self.grads.items():
            self.grads[name] /= counter_num

    def print_gradient(self):
        for grad in self.grads:
            if 'base.critic' in grad:
                # if grad == 'fc1.weight_grad':
                print(grad, '  ', self.grads[grad].mean())
        for name, grad in self.grads.items():
            if 'base.critic' in name:
                print(name, self.grads[name].mean())

    def reset(self):
        self.counter.reset()
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)