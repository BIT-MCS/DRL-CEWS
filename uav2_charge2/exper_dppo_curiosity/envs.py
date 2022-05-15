from environment_new.environment_new_uav2_charge2.log import *
from environment_new.environment_new_uav2_charge2.env import Env as DenseEnv
from sparse_environment.sparse_environment_uav2_charge2.env import Env as SparseEnv
from common_setting.path_setting import PATH
from util.utils import *

root_path = PATH.root_path


class Make_Env(object):
    def __init__(self, num_process, device, num_steps, local_time, env_index, use_sparse_env):
        self.num_process = num_process
        self.util = Util(device)
        self.time = str(local_time)
        if use_sparse_env:
            self.env_list = [SparseEnv(Log(env_index, num_steps, root_path, self.time))]
        else:
            self.env_list = [DenseEnv(Log(env_index, num_steps, root_path, self.time))]
        self.num_steps = num_steps

        self.env_action = [[] for i in range(num_process)]
        self.env_cur_energy = [[] for i in range(num_process)]

        self.env_data_collection = [[] for i in range(num_process)]
        self.env_fairness = [[] for i in range(num_process)]
        self.env_efficiency = [[] for i in range(num_process)]
        self.env_energy_consumption = [[] for i in range(num_process)]
        for i, env in enumerate(self.env_list):
            self.env_cur_energy[i].append([float(x) for x in list(env.cur_uav_energy)])
            self.env_data_collection[i].append(env.data_collection_ratio())
            self.env_fairness[i].append(env.geographical_fairness())
            self.env_efficiency[i].append(env.energy_efficiency_1())
            self.env_energy_consumption[i].append(env.energy_consumption_ratio())

    def reset(self):
        self.step_counter = 0
        self.totol_r = np.zeros(shape=[self.num_process], dtype=np.float32)
        obs = []
        uav_pos = []
        info = {}
        for env in self.env_list:
            ob, _, pos = env.reset()  # [80,80,3]
            ob = ob.transpose(2, 0, 1)  # [3,80,80]
            ob = np.expand_dims(ob, axis=0)  # [1,3,80,80]
            obs.append(self.util.to_tensor(ob))
            pos = np.ceil(pos / 0.8)
            # todo: 5 change new pos
            new_pos = pos[:, 0] * 21 + pos[:, 1]
            uav_pos.append(self.util.to_tensor(new_pos).int())

        obs = torch.cat(obs, dim=0)  # [num,3,80,80]
        uav_pos = torch.cat(uav_pos, dim=0)
        info['uav_position'] = uav_pos
        return obs, info

    def render(self):
        self.env_list[0].render()

    def get_uav_pos(self):
        pos = []
        for env in self.env_list:
            pos.append(env.get_uav_pos())
        return pos

    def imagine_step(self, action, current_step=None):
        obs = []
        reward = []
        for i, env in enumerate(self.env_list):
            # action [K,3]
            ob, r, d, _, _, _ = env.step(action[i], current_step)  # [80,80,3]
            ob = ob.transpose(2, 0, 1)  # [3,80,80]
            ob = np.expand_dims(ob, axis=0, dtype=np.float32)  # [1,3,80,80]
            obs.append(self.util.to_tensor(ob))
            r = np.array([r], dtype=np.float32)  # [1]
            r = np.expand_dims(r, axis=0)  # [1,1]
            reward.append(self.util.to_tensor(r))

        obs = torch.cat(obs, dim=0)  # [num,3,80,80]
        reward = torch.cat(reward, dim=0)  # [num,1]
        return obs, reward

    def step(self, action, current_step=None):

        self.step_counter += 1
        if self.step_counter <= self.num_steps:
            done = [False for i in range(self.num_process)]
        else:
            done = [True for i in range(self.num_process)]

        obs = []
        reward = []
        info = {}
        uav_pos = []

        for i, env in enumerate(self.env_list):
            # action [K,3]
            ob, r, d, _, pos, _ = env.step(action[i], current_step)  # [80,80,3]

            self.env_action[i].append([float(x) for x in list(np.reshape(action[i], [-1]))])
            self.env_cur_energy[i].append([float(x) for x in list(env.cur_uav_energy)])
            self.env_data_collection[i].append(env.data_collection_ratio())
            self.env_fairness[i].append(env.geographical_fairness())
            self.env_efficiency[i].append(env.energy_efficiency_1())
            self.env_energy_consumption[i].append(env.energy_consumption_ratio())

            self.totol_r[i] += r

            ob = ob.transpose(2, 0, 1)  # [3,80,80]
            ob = np.expand_dims(ob, axis=0)  # [1,3,80,80]
            obs.append(self.util.to_tensor(ob))

            pos = np.ceil(pos / 0.8)
            new_pos = pos[:, 0] * 21 + pos[:, 1]
            uav_pos.append(self.util.to_tensor(new_pos).int())

            r = np.array([r], dtype=np.float32)  # [1]
            r = np.expand_dims(r, axis=0)  # [1,1]
            reward.append(self.util.to_tensor(r))
        obs = torch.cat(obs, dim=0)  # [num,3,80,80]
        reward = torch.cat(reward, dim=0)  # [num,1]
        uav_pos = torch.cat(uav_pos, dim=0)  # [num, uav_num, 2]

        info['mean_episod_reward'] = np.mean(self.totol_r)
        info['max_episod_reward'] = np.max(self.totol_r)
        info['min_episod_reward'] = np.min(self.totol_r)
        info['uav_position'] = uav_pos

        return obs, reward, done, info

    def draw_path(self, step, plot=False):
        for env in self.env_list:
            env.log.draw_path(env, step, plot)

    def test_summary(self):
        summary_txt_path = self.log_path + '/' + 'test_summary.txt'
        f = open(summary_txt_path, 'w')
        f.writelines('mean effi is : ' + str(np.mean(self.mean_efficiency)) + '\n')
        f.writelines('mean d_c is : ' + str(np.mean(self.mean_data_collection_ratio)) + '\n')
        f.writelines('mean f is : ' + str(np.mean(self.mean_fairness)) + '\n')
        f.writelines('mean e_c is : ' + str(np.mean(self.mean_energy_consumption_ratio)))
        f.close()

        summary_npz_path = self.log_path + '/' + 'test_summary.npz'
        np.savez(summary_npz_path, self.efficiency, self.data_collection_ratio, self.fairness,
                 self.energy_consumption_ratio)

        self.env_uav_trace = [[[[] for _ in range(len(self.env_list[0].uav_trace))] for _ in
                               range(self.num_steps + 1)] for i in range(self.num_process)]
        for i, env in enumerate(self.env_list):
            for j in range(self.num_steps + 1):
                for s in range(len(env.uav_trace)):
                    if j < len(env.uav_trace[s]):
                        self.env_uav_trace[i][j][s].append([float(x) for x in list(env.uav_trace[s][j])])
                    else:
                        self.env_uav_trace[i][j][s].append(
                            [float(x) for x in list(env.uav_trace[s][len(env.uav_trace[s]) - 1])])
        json_dict = {}
        json_dict['action'] = self.env_action
        json_dict['trace'] = self.env_uav_trace
        json_dict['cur_energy'] = self.env_cur_energy
        json_dict['d_c'] = self.env_data_collection
        json_dict['f'] = self.env_fairness
        json_dict['e_c'] = self.env_energy_consumption
        json_dict['effi'] = self.env_efficiency

        import json

        json_str = json.dumps(json_dict)
        # print(json_str)
        # print(type(json_str))

        new_dict = json.loads(json_str)

        # print(new_dict)
        # print(type(new_dict))

        with open(self.log_path + '/' + 'record.json', "w") as f:
            json.dump(new_dict, f)

            print("加载入文件完成...")

    @property
    def log_path(self):
        return self.env_list[0].log.full_path

    @property
    def observ_shape(self):
        return self.env_list[0].observ.transpose(2, 0, 1).shape

    @property
    def action_space(self):
        return self.env_list[0].uav_num * 3

    @property
    def num_of_uav(self):
        return self.env_list[0].uav_num

    @property
    def mean_data_collection_ratio(self):
        return np.mean([env.data_collection_ratio() for env in self.env_list])

    @property
    def mean_fairness(self):
        return np.mean([env.geographical_fairness() for env in self.env_list])

    @property
    def mean_energy_consumption_ratio(self):
        return np.mean([env.energy_consumption_ratio() for env in self.env_list])

    @property
    def efficiency1(self):
        return np.array([env.energy_efficiency_1() for env in self.env_list])

    @property
    def data_collection_ratio(self):
        return np.array([env.data_collection_ratio() for env in self.env_list])

    @property
    def fairness(self):
        return np.array([env.geographical_fairness() for env in self.env_list])

    @property
    def energy_consumption_ratio(self):
        return np.array([env.energy_consumption_ratio() for env in self.env_list])

    @property
    def data_coverage(self):
        return np.array([env.get_data_coverage() for env in self.env_list])
