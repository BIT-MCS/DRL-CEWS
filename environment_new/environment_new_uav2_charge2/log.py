import time
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Log(object):
    def __init__(self, rank=None, num_of_step=None, root_path=None, time=None):
        self.time = time
        self.num_of_step = num_of_step
        self.full_path = os.path.join(root_path, self.time)
        self.rank = rank
        if rank is not None:
            self.full_path = os.path.join(self.full_path, str(rank))
        os.makedirs(self.full_path)

        self.file_path = self.full_path + '/REPORT.txt'
        file = open(self.file_path, 'w')
        file.close()
        self.result_path = self.full_path + '/' + 'result.npz'
        self.r = []
        self.d_c = []
        self.e_c = []
        self.f = []
        self.effi_1 = []
        self.effi_2 = []
        self.data_coverage = []

    def log(self, values):
        if isinstance(values, dict):
            with open(self.file_path, 'a') as file:
                for key, value in values.items():
                    file.write(str([key, value]) + '\n')
                    # print key, value, "file",file
        elif isinstance(values, list):
            with open(self.file_path, 'a') as file:
                for value in values:
                    file.write(str(value) + '\n')
                    # print value,"file",file
        else:
            with open(self.file_path, 'a') as file:
                file.write(str(values) + '\n')
                # print values, "file",file

    def circle(self, x, y, r, color=np.stack([1., 0., 0.]), count=50):
        xarr = []
        yarr = []
        for i in range(count):
            j = float(i) / count * 2 * np.pi
            xarr.append(x + r * np.cos(j))
            yarr.append(y + r * np.sin(j))
        plt.plot(xarr, yarr, c=color)

    def draw_path(self, env, step, plot=False):
        if plot:
            # print('in draw path')
            full_path = os.path.join(self.full_path, 'Path')
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            xxx = []
            colors = []
            for x in range(env.map_size_x):  # 16
                xxx.append((x, 1))
            for y in range(env.map_size_y):  # 16
                c = []
                for x in range(env.map_size_x):
                    if env.map_obstacle[x][y] == env.map_obstacle_value:
                        c.append((1, 0, 0, 1))
                    else:
                        c.append((1, 1, 1, 1))
                colors.append(c)

            Fig = plt.figure(figsize=(5, 5))
            PATH = env.uav_trace
            POI_PATH = [[] for i in range(env.uav_num)]
            POWER_PATH = [[] for i in range(env.uav_num)]
            DEAD_PATH = [[] for i in range(env.uav_num)]
            CONTINUE_CHARGE = [np.zeros(shape=self.num_of_step + 1, dtype=np.int32) for i in range(env.uav_num)]
            # txt_path = full_path + '/path_' + str(step) + '_continuous_charge_.txt'
            # f = open(txt_path, 'a')

            for i in range(env.uav_num):
                charge_counter = 0
                for j, pos in enumerate(PATH[i]):
                    if env.uav_state[i][j] == 0:  # DEAD
                        DEAD_PATH[i].append(pos)
                        if charge_counter > 0:
                            CONTINUE_CHARGE[i][charge_counter] += 1
                        charge_counter = 0
                    elif env.uav_state[i][j] == 1:  # collect data
                        POI_PATH[i].append(pos)
                        if charge_counter > 0:
                            CONTINUE_CHARGE[i][charge_counter] += 1
                        charge_counter = 0
                    elif env.uav_state[i][j] == -1:  # charge power
                        POWER_PATH[i].append(pos)
                        charge_counter += 1
                    else:
                        print(" error")
                if charge_counter > 0:
                    CONTINUE_CHARGE[i][charge_counter] += 1

            # for i in range(self.num_of_step + 1):
            #     for j in range(env.uav_num):
            #         if CONTINUE_CHARGE[j][i] > 0:
            #             f.writelines('uav-' + str(j) + ' continuous_charge=' + str(i) + ' frequency=' + str(
            #                 CONTINUE_CHARGE[j][i]) + '\n')
            # f.writelines('\n--------------*------------------\n')
            # for i in range(env.uav_num):
            #     # print('len(env.uav_trace_info[i])', len(env.uav_trace_info[i]))
            #     for info in env.uav_trace_info[i]:
            #         f.writelines(
            #             'uav-' + str(i) + ' step ' + str(info[0]) + ' type ' + str(info[1]) + ' pos ' + str(
            #                 info[2]) + ' collect power/data ' + str(info[3]) + ' energy ' + str(
            #                 info[4]) + ' reward ' + str(info[5]) + ' penalty ' + str(info[6]) + '\n')
            # f.close()

            for i1 in range(env.map_size_y):
                plt.broken_barh(xxx, (i1, 1), facecolors=colors[i1])

            plt.scatter(env.poi_data_pos[:, 0], env.poi_data_pos[:, 1], c=env.init_poi_data_val[:])

            for i in range(env.uav_num):
                # M = Fig.add_subplot(1, 1, i + 1)
                plt.ylim(ymin=0, ymax=env.map_size_x)
                plt.xlim(xmin=0, xmax=env.map_size_y)
                color = np.random.random(3)
                plt.plot(np.stack(PATH[i])[:, 0], np.stack(PATH[i])[:, 1], color=color)

                if len(POI_PATH[i]) > 0:
                    plt.scatter(np.stack(POI_PATH[i])[:, 0], np.stack(POI_PATH[i])[:, 1], color=color, marker='.')

                if len(POWER_PATH[i]) > 0:
                    plt.scatter(np.stack(POWER_PATH[i])[:, 0], np.stack(POWER_PATH[i])[:, 1], color=color * 0.5, marker='+')

                if len(DEAD_PATH[i]) > 0:
                    plt.scatter(np.stack(DEAD_PATH[i])[:, 0], np.stack(DEAD_PATH[i])[:, 1], color=color, marker='D')

            plt.grid(True, linestyle='-.', color='r')
            # plt.title(str(env.normal_energy) + ',' + str(env.leftrewards))

        r = np.round(env.episodic_total_uav_reward, 2)
        e_l = np.sum(env.cur_uav_energy)
        c_c = env.charge_counter
        d_c = np.round(env.data_collection_ratio(), 2)
        e_c = np.round(env.energy_consumption_ratio(), 2)
        f = np.round(env.get_fairness(), 2)
        effi_1 = np.round(env.energy_efficiency_1(), 2)
        data_coverage = np.round(env.get_data_coverage(), 2)

        if plot:
            plt.title(
                str(
                    step) + ' r=' + str(r) + ' e_l=' + str(e_l) + ' c_c=' + str(c_c) + ' d_c=' + str(d_c) + '\n e_c=' + str(
                    e_c) + ' f=' + str(f) + ' effi=' + str(effi_1))

        self.r.append(r)
        self.d_c.append(d_c)
        self.e_c.append(e_c)
        self.f.append(f)
        self.effi_1.append(effi_1)
        self.data_coverage.append(data_coverage)

        np.savez(self.result_path, np.asarray(self.r), np.asarray(self.d_c), np.asarray(self.e_c), np.asarray(self.f),
                 np.asarray(self.effi_1), np.asarray(self.data_coverage))

        if plot:
            plt.scatter(env.pow_pos[:, 0], env.pow_pos[:, 1], marker='*', color=np.stack([1., 0., 0.]))
            for i in range(env.pow_pos.shape[0]):
                self.circle(env.pow_pos[i, 0], env.pow_pos[i, 1], env.crange)

            Fig.savefig(full_path + '/path_' + str(step) + '.png')

            plt.close()

    def draw_convert(self, observ, img, step, name):
        max_val = np.max(observ)
        min_val = np.min(observ)
        for i in range(80):
            for j in range(80):

                if observ[i, j] < 0:
                    img[i, j, 0] = np.uint8(255)
                if observ[i, j] > 0:
                    if max_val > 0:
                        img[i, j, 1] = np.uint8(255 * (observ[i, j] / max_val))

        full_path = os.path.join(self.full_path, 'Observ')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        save_path = full_path + '/observ_' + str(step) + name + '.png'

        # cv2.imwrite(save_path, img)

    def draw_observation(self, env, step, is_start=True):
        observ = env.get_observation()
        observ_0 = observ[0, :, :, 0]
        observ_1 = observ[1, :, :, 1]
        observ_2_1 = observ[0, :, :, 2]
        observ_2_2 = observ[1, :, :, 2]

        img_0 = np.zeros([80, 80, 3], dtype=np.uint8)
        img_1 = np.zeros([80, 80, 3], dtype=np.uint8)
        img_2_1 = np.zeros([80, 80, 3], dtype=np.uint8)
        img_2_2 = np.zeros([80, 80, 3], dtype=np.uint8)
        if is_start:
            end = 'start'
        else:
            end = 'end'
        self.draw_convert(observ_0, img_0, step, 'wall_poi' + end)
        self.draw_convert(observ_1, img_1, step, 'visit_times' + end)
        self.draw_convert(observ_2_1, img_2_1, step, 'uav_power_1' + end)
        self.draw_convert(observ_2_2, img_2_2, step, 'uav_power_2' + end)
