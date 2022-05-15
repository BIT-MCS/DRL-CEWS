import torch.nn as nn
import time
import datetime
import os
from common_setting.path_setting import PATH
import csv


def chief(update_threshold, traffic_light, counter, shared_model, shared_icm_model, shared_grad_buffers,
          shared_grad_buffers_icm, optimizer, son_process_counter, max_grad_norm, local_time, use_icm):
    start_time = datetime.datetime.now()
    while True:
        time.sleep(1)
        if counter.get() > update_threshold:
            optimizer.zero_grad()
            # shared_grad_buffers.average_gradient()
            for n, p in shared_model.named_parameters():
                if p.requires_grad:
                    p._grad = shared_grad_buffers.grads[n + '_grad'].clone().detach()
            nn.utils.clip_grad_norm_(shared_model.parameters(), max_grad_norm)

            if use_icm:
                # shared_grad_buffers_icm.average_gradient()
                for n, p in shared_icm_model.named_parameters():
                    if p.requires_grad:
                        p._grad = shared_grad_buffers_icm.grads[n + '_grad'].clone().detach()
                nn.utils.clip_grad_norm_(shared_icm_model.parameters(), max_grad_norm)
            optimizer.step()
            shared_grad_buffers.reset()
            if use_icm:
                shared_grad_buffers_icm.reset()
            counter.reset()
            traffic_light.switch()  # workers start new loss computation

        if son_process_counter.get() > update_threshold:
            break

    total_time = datetime.datetime.now() - start_time
    time_root_path = os.path.join(PATH.root_path, str(local_time))
    time_file = open(os.path.join(time_root_path, 'run_time.txt'), 'w', newline='')
    time_file.write(str(total_time))
    time_file.close()
