import torch
from uav2_charge2.exper_dppo_curiosity.envs import *
from uav2_charge2.exper_dppo_curiosity.params import Params
from common_setting.path_setting import PATH
from uav2_charge2.exper_dppo_curiosity.model import Model, ICMModel
from uav2_charge2.exper_dppo_curiosity.utils import seed_torch
from util.utils import Util
import os

params = Params()
torch.set_num_threads(1)
# torch.cuda.set_device(0)
if params.device_num == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:'+str(params.device_num))
# device = torch.device("cuda" if params.use_cuda else "cpu")
util = Util(device)


def action_convert(action):
    action = (util.to_numpy(action[0]), util.to_numpy(action[1]))
    num_of_process = action[0].shape[0]
    num_of_uav = int(action[1].shape[1])
    action_new = np.zeros([num_of_process, num_of_uav, 3], dtype=np.float32)
    for i in range(num_of_process):
        state = action[0][i, 0]
        for j in range(num_of_uav):
            if state % 2 == 0:
                action_new[i, j, 0] = 1  # collect data
            else:
                action_new[i, j, 0] = -1  # charge
            state = state // 2
            action_new[i, j, 1] = params.discrete_action[action[1][i, j]][0]
            action_new[i, j, 2] = params.discrete_action[action[1][i, j]][1]
    action_new = action_new.reshape([num_of_process, -1])
    return action_new


def test(local_time, load_model_path):
    seed_torch(params.seed)
    num_process = 1
    env_index = 0
    # ----------------make environment----------------------
    env = Make_Env(num_process, device, params.exploration_steps, local_time, env_index, params.use_sparse_env)
    # -----------------load parameters----------------------
    obs_shape = env.observ_shape
    uav_num = params.uav_num

    # ---------------create local model---------------------
    local_ppo_model = Model(obs_shape, uav_num, params.cat_ratio, params.dia_ratio, device, len(params.discrete_action),
                            trainable=False)
    ppo_model_path = os.path.join(load_model_path, 'ppo_model_2500.pt')
    local_ppo_model.load_state_dict(torch.load(ppo_model_path, map_location=device))
    local_ppo_model.to(device)

    episode_length = 0
    # --------------define file writer-----------------------
    file_root_path = os.path.join(PATH.root_path, str(local_time) + '/test/file')
    os.makedirs(file_root_path)

    # load local model parameters
    done = False
    tau = 0.1
    mean_data_collection = 0.
    mean_energy_consumption = 0.
    mean_task_efficiency1 = 0.
    mean_task_efficiency2 = 0.
    mean_data_coverage = 0
    mean_fairness = 0.

    while True:
        if episode_length >= params.max_test_length:
            print('training over')
            break
        print('---------------in episode ', episode_length, '-----------------------')

        step = 0
        av_reward = 0

        obs, _ = env.reset()
        while step < params.exploration_steps:
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                value, action, action_log_probs = local_ppo_model.act(obs, tau)
                obs, reward, done, info = env.step(action_convert(action), current_step=step)
            av_reward += reward
            step = step + 1
        d_c = env.data_collection_ratio
        e_c = env.energy_consumption_ratio
        e_f1 = env.efficiency1
        data_coverage = env.data_coverage
        mean_data_coverage += data_coverage
        mean_data_collection += d_c
        mean_energy_consumption += e_c
        mean_task_efficiency1 += e_f1
        f = np.round(env.fairness, 2)
        mean_fairness += f

        # --------------- draw & log -----------------------------
        # env.get_heatmap(episode_length)
        if episode_length % params.test_log_interval == 0:
            env.draw_path(episode_length, plot=True)

        episode_length = episode_length + 1

    mean_data_collection /= params.max_test_length
    mean_energy_consumption /= params.max_test_length
    mean_fairness /= params.max_test_length
    mean_task_efficiency1 /= params.max_test_length
    mean_data_coverage /= params.max_test_length

    print('mean_data_collection', mean_data_collection)
    print('mean_energy_consumption', mean_energy_consumption)
    print('mean_task_efficiency', mean_task_efficiency1)
    print('mean_fairness', mean_fairness)
    print('mean_data_coverage', mean_data_coverage)
