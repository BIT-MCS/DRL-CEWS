import torch.nn as nn
import numpy as np
from common_setting.path_setting import PATH
import csv
import os


class Params(nn.Module):
    def __init__(self):
        super(Params, self).__init__()
        # self.use_cuda = True
        # self.device_num = 6
        self.device_num = 0
        # --------------PPO parameters------------------
        self.lr = 3e-4
        self.dia_ratio = 1
        self.cat_ratio = 1
        self.clip = 0.1
        self.ent_coeff = 0.01
        self.value_coeff = 0.1
        self.clip_coeff = 1.
        self.max_episode_length = 2500
        self.exploration_steps = 500
        self.mini_batch_num = 4
        self.max_grad_norm = 250
        # todo: debug
        # self.num_processes = 2
        self.num_processes = 8
        self.update_threshold = self.num_processes - 1
        self.seed = 1
        self.ppo_epoch = 4
        # ------rnn parameters------------
        self.zero_init = True
        self.use_rnn = False
        self.rnn_hidden_size = 4
        self.rnn_seq_len = 5

        # ------environment setting--------------
        self.use_sparse_env = True
        # self.use_sparse_env = False

        # -----discounted return parameters-------
        self.use_gae = True
        self.gamma = 0.99
        self.gae_param = 0.95
        # -----icm model parameters-------
        self.use_icm = True
        # self.use_icm = False
        # self.eta = 0.01
        # self.icm_feature = "RF"
        # self.icm_feature = "IDF"
        # self.icm_feature = "RND"
        self.icm_feature = "EMD"
        self.eta = 0.3
        self.ext_coeff = 1.
        self.int_coeff = 1.
        self.beta = 0.2
        # ----------------method------------------
        self.distributed = False
        self.use_obs_norm = False
        self.running_obs_normal = False
        self.use_adv_norm = True
        # self.use_adv_norm = False

        # ----------------test---------------------
        self.trainable = False
        # self.trainable = True
        self.max_test_length = 100
        self.test_log_interval = 10
        # ----------environment parameters---------
        self.uav_num = 2
        self.discrete_action = np.array([
            [0, 0.8], [0, -0.8],
            [0.8, 0], [0.8, 0.8], [0.8, -0.8],
            [-0.8, 0], [-0.8, 0.8], [-0.8, -0.8]
        ])

    def log_info(self, local_time):
        log_file_path = os.path.join(PATH.root_path, str(local_time))
        os.makedirs(log_file_path)
        log_file_path = os.path.join(log_file_path, 'hyper_parameters.csv')
        log_file = open(log_file_path, 'a', newline='')
        file_reader = csv.writer(log_file)
        for p in self.__dict__:
            if p[0] == '_':
                continue
            file_reader.writerow([p, self.__getattribute__(p)])
        log_file.close()
