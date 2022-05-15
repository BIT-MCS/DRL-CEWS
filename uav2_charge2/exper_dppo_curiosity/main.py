from uav2_charge2.exper_dppo_curiosity.params import Params
import torch.multiprocessing as mp
from common_setting.env_setting import FLAGS
from uav2_charge2.exper_dppo_curiosity.model import Model, Shared_grad_buffers, ICMModel, RFModel, RNDModel, \
    EmbeddingModel
import os
import torch.optim as optim
import torch
from uav2_charge2.exper_dppo_curiosity.model_test import test
from util.utils import TrafficLight, Counter
from uav2_charge2.exper_dppo_curiosity.train import train
from uav2_charge2.exper_dppo_curiosity.chief import chief
from uav2_charge2.exper_dppo_curiosity.utils import seed_torch
import time

params = Params()
seed_torch(params.seed)


def main():
    local_time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    params.log_info(local_time)
    torch.cuda.set_device(0)
    if params.device_num == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+str(params.device_num))
    obs_shape = (FLAGS.image_depth, FLAGS.image_size, FLAGS.image_size)

    if params.trainable:
        traffic_light = TrafficLight()
        counter = Counter()
        son_process_counter = Counter()

        # --------------create shared model-------------------
        shared_model = Model(obs_shape, params.uav_num, params.cat_ratio, params.dia_ratio, device,
                             len(params.discrete_action))
        shared_model.share_memory().to(device)

        # ------------create shared grad buffer list----------
        shared_grad_buffer = Shared_grad_buffers(shared_model, device)

        # -----------create optimizer list --------------------
        optimizer = None
        shared_icm_model = None
        shared_grad_buffer_icm = None
        if params.use_icm:
            if params.icm_feature == "IDF":
                shared_icm_model = ICMModel(obs_shape, len(params.discrete_action), params.uav_num)
            elif params.icm_feature == "RF":
                shared_icm_model = RFModel(params.uav_num, device)
            elif params.icm_feature == "RND":
                shared_icm_model = RNDModel(obs_shape, device)
            elif params.icm_feature == "EMD":
                shared_icm_model = EmbeddingModel(len(params.discrete_action), device)
            else:
                print('Feature Error')
                exit(-1)
            shared_icm_model.share_memory().to(device)
            shared_grad_buffer_icm = Shared_grad_buffers(shared_icm_model, device)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          list(shared_model.parameters()) + list(shared_icm_model.parameters())),
                                   lr=params.lr)
        else:
            optimizer = optim.Adam(list(shared_model.parameters()), lr=params.lr)

        processes = []
        p = mp.Process(target=chief, args=(
            params.update_threshold, traffic_light, counter, shared_model, shared_icm_model, shared_grad_buffer,
            shared_grad_buffer_icm, optimizer, son_process_counter, params.max_grad_norm, local_time, params.use_icm))
        p.start()
        processes.append(p)

        for rank in range(0, params.num_processes):
            p = mp.Process(target=train, args=(
                rank, traffic_light, counter, shared_model, shared_icm_model, shared_grad_buffer,
                shared_grad_buffer_icm, local_time, son_process_counter))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    else:
        load_model_path = "ckpt/"
        p = mp.Process(target=test, args=(local_time, load_model_path))
        p.start()
        p.join()
