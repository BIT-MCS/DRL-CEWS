import torch.nn as nn
import torch.functional as F
import numpy as np
from util.utils import *


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    else:
        print('None bias')
    return module


# >>> inputs = torch.randn(20, 16, 50, 10, 20)
# >>> weights = torch.randn(16, 33, 3, 3, 3)
# >>> F.conv_transpose3d(inputs, weights) [20,33,52,12,22]

class ConvMapCell(torch.nn.Module):
    def __init__(
            self,
            m_features=1,
            m_h=21 * 16,
            m_x=6,
            m_y=6,
            device=None,
            soft_update=True,
            update_double_bias=True,
    ):
        super(ConvMapCell, self).__init__()
        self.util = Util(device)
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               # nn.init.xavier_uniform_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        print('convmap brd',m_features,m_h,m_x,m_y)
        # memory
        self.m_len = int(m_h / m_features)
        self.memory = None
        self.memory_shape = (m_features, self.m_len, m_x, m_y)
        self.update_double_bias = update_double_bias

        # reader
        self.reader = nn.Sequential(
            # (8, 16*21, 6, 6)  N,C,D,W,H

            init_(nn.Conv2d(m_h, int(m_h / 3 * 4), (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.BatchNorm2d(int(m_h / 3 * 4)),  # TODO bn1
            nn.ReLU(),
            nn.Dropout2d(0.5),  # TODO 2018-12-2
            # (8, 32 * 14, 6, 6)

            init_(nn.Conv2d(int(m_h / 3 * 4), int(m_h / 3), (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.BatchNorm2d(int(m_h / 3)),  # TODO bn1
            nn.ReLU(),
            nn.Dropout2d(0.5),  # TODO 2018-12-2
            #    (8, 16 * 7, 6, 6)
            init_(nn.Conv2d(int(m_h / 3), m_features, (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.BatchNorm2d(m_features),  # TODO bn1
            nn.ReLU(),
            nn.Dropout2d(0.5),  # TODO 2018-12-2
            # r->(8, 16, 6, 6)

        )

        # context
        self.q = nn.Sequential(
            # (8, 32+16, 6, 6) s+r
            init_(nn.Conv2d(32 + m_features, m_features, (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.ReLU(),

            # q -> (8, 16, 6, 6)
        )
        self.context_softmax = torch.nn.Softmax(dim=1)  # [8,21]
        # write
        self.writer_interface = nn.Sequential(
            # (8, 32, 6, 6) s
            init_(nn.Conv2d(32, m_features, (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.ReLU(),
            # -> (8, 16, 6, 6)
        )
        self.writer = nn.Sequential(
            # c r s

            # (8, 16*(1+1+1), 6, 6)

            init_(nn.ConvTranspose2d(m_features * (1 + 1 + 1), int(m_h / 3), (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.BatchNorm2d(int(m_h / 3)),  # TODO bn2
            nn.ReLU(),
            nn.Dropout2d(0.5),  # TODO 2018-12-2
            # (8, 16 * 7, 6, 6)

            init_(nn.ConvTranspose2d(int(m_h / 3), int(m_h / 3 * 4), (3, 3), stride=(1, 1),
                                     padding=(1, 1))),
            nn.BatchNorm2d(int(m_h / 3 * 4)),  # TODO bn2
            nn.ReLU(),
            nn.Dropout2d(0.5),  # TODO 2018-12-2
            # # (8,  32 * 14, 6, 6)
            #
            init_(nn.ConvTranspose2d(int(m_h / 3 * 4), m_h, (3, 3), stride=(1, 1), padding=(1, 1))),
            nn.BatchNorm2d(m_h),  # TODO bn2
            nn.ReLU(),
            nn.Dropout2d(0.5),  # TODO 2018-12-2
            # (8,  16 * 21 , 6, 6)
        )

        # output
        self.output_interface = nn.Sequential(
            # (8, 64, 6, 6)
            init_(nn.Conv2d(32 + m_features * 2, 32, (1, 1), stride=(1, 1), padding=(0, 0))),
            nn.ReLU(),
            # nn.BatchNorm2d(32),  # TODO bn2
            # nn.Dropout2d(0.5),  # TODO 2018-12-2
            # (8, 64, 6, 6)
        )

        # update
        self.soft_update = soft_update
        if soft_update:
            print('soft')
            self.w_i_r = init_(nn.Conv2d(m_h, m_h, (3, 3), stride=(1, 1), padding=(1, 1)))
            self.w_h_r = init_(
                nn.Conv2d(m_h, m_h, (3, 3), stride=(1, 1), padding=(1, 1), bias=self.update_double_bias))

            self.w_i_z = init_(nn.Conv2d(m_h, m_h, (3, 3), stride=(1, 1), padding=(1, 1)))
            self.w_h_z = init_(
                nn.Conv2d(m_h, m_h, (3, 3), stride=(1, 1), padding=(1, 1), bias=self.update_double_bias))

            self.w_i_n = init_(nn.Conv2d(m_h, m_h, (3, 3), stride=(1, 1), padding=(1, 1)))
            self.w_h_n = init_(
                nn.Conv2d(m_h, m_h, (3, 3), stride=(1, 1), padding=(1, 1), bias=self.update_double_bias))
        else:
            print('hard')

    def read(self, m):
        # m->(8, 16, 21, 6, 6)
        m_list = torch.chunk(m, chunks=m.shape[2], dim=2)
        # m->(8, 16, 1, 6, 6)

        m = torch.cat(m_list, dim=1).squeeze(dim=2)
        # m->(8, 16*21, 6, 6)

        r = self.reader(m)
        # r->(8, 16, 6, 6)

        r = r.unsqueeze(dim=2)
        # r->(8, 16, 1, 6, 6)

        return r

    def context(self, s, r, m):
        # s->(8, 32, 6, 6)
        # r->(8, 16, 6, 6)
        # m->(8, 16, 21, 6, 6)

        s_r = torch.cat([s, r.squeeze(dim=2)], dim=1)
        # s_r->(8, 48, 6, 6)

        # m->(8, 16, 21, 6, 6)
        m_flatten = m.permute(0, 2, 1, 3, 4).contiguous()
        m_flatten = m_flatten.view(m.shape[0], m.shape[2], -1)
        # m_flatten->(8, 21, 16*6*6)

        q = self.q(s_r)
        # q -> (8, 16, 6, 6)

        q = q.view(q.shape[0], -1)
        # q -> (8, 16* 6* 6)
        q = q.unsqueeze(dim=2).permute(0, 2, 1)
        # q -> (8, 1 , 16* 6* 6)



        # calculate a
        a = torch.einsum("abc,adcb->adb", (q, m_flatten.unsqueeze(dim=3)))
        # a-->(8, 21, 1)
        a = a.squeeze(dim=2)
        # a-->(8, 21)


        # calculate alpha
        alpha = self.context_softmax(a)

        alpha = alpha.unsqueeze(dim=2)
        # alpha-->(8, 21, 1)

        # calculate c
        c = alpha * m_flatten
        # c->(8, 21, 16*6*6)
        c = torch.sum(c, dim=1)
        # c->(8, 1, 16*6*6)

        c = c.contiguous().view(-1, self.memory_shape[0], self.memory_shape[2], self.memory_shape[3]).unsqueeze(
            dim=2)
        # c->(8, 16, 1, 6, 6)

        return c

    def write(self, c, r, s):
        s_in = self.writer_interface(s).unsqueeze(dim=2)  # -> (8, 16, 1, 6, 6)
        c_in = torch.nn.functional.relu_(c)  # -> (8, 16, 1, 6, 6)
        r_in = torch.nn.functional.relu_(r)  # -> (8, 16, 1, 6, 6)

        layer_in = torch.cat([c_in, r_in, s_in], dim=1).squeeze(dim=2)
        # layin-> (8, 16*3, 6, 6)

        w = self.writer(layer_in)

        w = w.view(w.shape[0], self.memory_shape[0], self.memory_shape[1], self.memory_shape[2],
                   self.memory_shape[3])
        return w

    def cal_output(self, s, r, c):
        c_r = torch.cat([c, r], dim=1).squeeze(dim=2)
        c_r_s = torch.cat([c_r, s], dim=1)

        output = self.output_interface(c_r_s)
        return output

    def merge_memory(self, m):
        # m->(8, 16, 21, 6, 6)
        m_list = torch.chunk(m, chunks=self.memory_shape[1], dim=2)
        # m->(8, 16, 1, 6, 6)

        m = torch.cat(m_list, dim=1).squeeze(dim=2)
        # m->(8, 16*21, 6, 6)
        return m

    def split_memory(self, m):
        # m->(8, 16*21, 6, 6)
        m_list = torch.chunk(m, chunks=self.memory_shape[1], dim=1)
        m_list = [m.unsqueeze(dim=2) for m in m_list]
        # m->(8, 16, 1, 6, 6)

        m = torch.cat(m_list, dim=2)
        # m->(8, 16,21, 6, 6)
        return m

    def update(self, x, h):

        x = self.merge_memory(x)
        h = self.merge_memory(h)

        r = torch.sigmoid(self.w_i_r(x) + self.w_h_r(h))
        z = torch.sigmoid(self.w_i_z(x) + self.w_h_z(h))
        n = torch.tanh(self.w_i_n(x) + r * (self.w_h_n(h)))
        # n = torch.relu(self.w_i_n(x) + r * (self.w_h_n(h))) #TODO 2018-12-10
        h_ = (1 - z) * n + z * h

        h_ = self.split_memory(h_)
        return h_

    def forward(self, s, m):
        r = self.read(m)

        c = self.context(s=s, r=r, m=m)

        w = self.write(c=c, r=r, s=s)

        o = self.cal_output(s=s, r=r, c=c)

        # TODO 2018-12-9
        if self.soft_update:
            w = self.update(x=w, h=m)
        # TODO 2018-12-9

        return o, w
