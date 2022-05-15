from environment_new.environment_new_uav2_charge2.env_setting import Setting
from os.path import join as pjoin
import numpy as np
from environment_new.environment_new_uav2_charge2.draw_util import *
import copy
import os


def mypjoin(path1, path2, paths=None):
    full_path = pjoin(path1, path2)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    if paths is not None:
        full_path = pjoin(full_path, paths)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    return full_path


def myint(a):
    # return int(np.ceil(a))
    return int(np.floor(a))


class Env(object):
    def __init__(self, log):
        self.log = log
        self.num_of_step = self.log.num_of_step
        self.sg = Setting(log)
        self.draw = Draw(self.sg)
        self.mini_energy_ratio = self.sg.V['MINI_ENERGY_RATIO']
        self.pow_reward_ratio = self.sg.V['REWARD_POW_RATIO']

        # make save directory
        self.sg.log()
        self.log_dir = log.full_path

        # [ 16 , 16 ]
        self.map_size_x = self.sg.V['MAP_X']  # 16
        self.map_size_y = self.sg.V['MAP_Y']  # 16

        # map obstacle [ 16 , 16 ]
        self.map_obstacle = np.zeros((self.map_size_x, self.map_size_y)).astype(np.int8)
        self.map_obstacle_value = 1
        self.init_map_obstacle()

        self.obstacle = self.sg.V['OBSTACLE']

        # uav energy
        self.init_uav_energy = np.asarray(
            [self.sg.V['ENERGY']] * self.sg.V['NUM_UAV'],
            dtype=np.float32
        )
        # num of uavs
        self.uav_num = self.sg.V['NUM_UAV']

        # goal of the exploration
        self.goal = 0.05

        # action  [ K , 3 ]
        self.action = np.zeros(
            shape=[self.sg.V['NUM_UAV'],
                   self.sg.V['ACT_NUM']],
            dtype=np.float32
        )

        # data collecting range
        self.crange = self.sg.V['RANGE']  # 0.8
        # data collecting speed
        self.cspeed = self.sg.V['COLLECTION_PROPORTION']  # 0.2

        # max moving distance within a time step
        self.maxdistance = self.sg.V['MAXDISTANCE']  # 1.0
        self.mindistance = self.sg.V['MIN_DISTANCE']

        # PoI
        # todo: add poi process
        self.poi_data_pos, self.init_poi_data_val = self.filter_PoI_data()
        self.poi_num = len(self.poi_data_pos)

        self.pow_pos = self.filter_power_pos()
        self.pow_remain_energy = self.sg.V['POWER_INIT'] * np.ones([self.sg.V['POWER_NUM']])

        # for render
        self.max_uav_energy = self.sg.V['ENERGY'] / self.sg.V['ENERGY']
        self.total_graph_energy = self.sg.V['POWER_INIT'] * self.sg.V['POWER_NUM'] + self.sg.V['ENERGY'] * self.sg.V[
            'NUM_UAV']
        self.total_pow_energy = self.sg.V['POWER_INIT'] * self.sg.V['POWER_NUM']
        self.reset()

        self.action_space = self.sg.V['NUM_UAV'] * self.sg.V['ACT_NUM']

    def reset(self):
        self.goal = 0.05
        self.observ = None  # observation  [ K , 80 , 80 , 3 ]
        self.observ_0 = None  # Border Obstacle PoI
        self.observ_1 = None  # Po poi_data_pos->[256,2]  poi_data_val->[256]I visit times
        self.init_observation()

        self.uav_trace = [[] for i in range(self.uav_num)]
        self.uav_state = [[] for i in range(self.uav_num)]
        self.cur_uav_pos = np.copy(self.sg.V['INIT_POSITION'])
        for i in range(self.uav_num):
            self.uav_trace[i].append(copy.deepcopy(self.cur_uav_pos[i]))
            self.uav_state[i].append(1)

        # self.uav_charge_trace = [[] for i in range(self.uav_num)]  # for log

        self.cur_poi_data_val = np.copy(self.init_poi_data_val)
        # poi_data_val->[256]
        self.poi_access_times = np.zeros([len(self.cur_poi_data_val)], dtype=np.float32)

        self.cur_uav_energy = copy.deepcopy(self.init_uav_energy)

        self.uav_energy_consuming = np.zeros(shape=[self.uav_num], dtype=np.float32)
        self.uav_energy_charging = np.zeros(shape=[self.uav_num], dtype=np.float32)
        self.charge_counter = np.zeros(shape=[self.uav_num], dtype=np.int)
        # [K]

        self.dead_uav_list = [False] * self.uav_num

        self.uav_heat_map = np.zeros([self.uav_num, 20], dtype=np.float32)

        self.total_collected_data = np.zeros(shape=[self.uav_num], dtype=np.float32)
        self.cur_collected_data = np.zeros(shape=[self.uav_num], dtype=np.float32)
        self.cur_collected_power = np.zeros(shape=[self.uav_num], dtype=np.float32)
        self.uav_trace_info = [[] for _ in range(self.uav_num)]

        self.episodic_total_uav_reward = 0.
        # for render
        self.uav_render_color = []
        for i in range(self.uav_num):
            self.uav_render_color.append(np.asarray(
                [np.random.randint(low=0, high=255), np.random.randint(low=0, high=255),
                 np.random.randint(low=0, high=255)]))
        self.pow_remain_energy = self.sg.V['POWER_INIT'] * np.ones([self.sg.V['POWER_NUM']])
        self.total_graph_energy = self.sg.V['POWER_INIT'] * self.sg.V['POWER_NUM'] + self.sg.V['ENERGY'] * self.sg.V[
            'NUM_UAV']
        return self.get_observation(), np.copy(self.uav_heat_map), np.copy(self.cur_uav_pos)

    def init_map_obstacle(self):
        obs = self.sg.V['OBSTACLE']
        # self.pos_obstacle_dict = {}
        # draw obstacles in map_obstacle [16 , 16]    the obstacle is 1 , others is 0
        for i in obs:

            # self.pos_obstacle_dict[(i[0], i[1])] = i

            for x in range(i[0], i[0] + i[2], 1):
                for y in range(i[1], i[1] + i[3], 1):
                    self.map_obstacle[x][y] = self.map_obstacle_value

    def filter_PoI_data(self):

        PoI = np.reshape(self.sg.V['PoI'], (-1, 3)).astype(np.float16)
        # replace the PoI in obstacle position with the PoI out of obstacle position
        for index in range(PoI.shape[0]):
            while self.map_obstacle[myint(PoI[index][0] * self.map_size_x)][
                myint(PoI[index][1] * self.map_size_y)] == self.map_obstacle_value:
                PoI[index] = np.random.rand(3).astype(np.float16)

        # PoI data value [ 256]
        poi_data_val = copy.copy(PoI[:, 2])

        # for render
        self.max_poi_val = max(poi_data_val)

        # PoI data Position  [ 256 , 2 ]
        poi_data_pos = PoI[:, 0:2] * self.map_size_x

        # sum of all PoI data values
        self.totaldata = np.sum(PoI[:, 2])

        self.log.log(PoI)

        return poi_data_pos, poi_data_val

    def filter_power_pos(self):
        power = np.asarray(self.sg.V['POWER'], dtype=np.float32)

        # replace the power in obstacle position with the power out of obstacle position
        for index in range(self.sg.V['POWER_NUM']):
            while self.map_obstacle[myint(power[index, 0])][
                myint(power[index, 1])] == self.map_obstacle_value:
                power[index, :] = np.random.rand(2)
                # print "power position has been reseted, error in memory"
        # Power Position  [ 50 , 2 ]
        power_pos = power

        return power_pos

    def init_observation(self):
        # observation  [ 80 , 80 , 3 ]
        self.observ = np.zeros(
            shape=[self.sg.V['OBSER_X'],
                   self.sg.V['OBSER_Y'],
                   self.sg.V['OBSER_C']],
            dtype=np.float32
        )

        self.observ_0 = np.zeros([self.sg.V['OBSER_X'], self.sg.V['OBSER_Y']], dtype=np.float32)  # Border Obstacle PoI
        self.observ_1 = np.zeros([self.sg.V['OBSER_X'], self.sg.V['OBSER_Y']], dtype=np.float32)  # PoI visit times

        # empty wall  ----layer1
        # draw walls in the border of the map (self._image_data)
        # the value of the wall is -1
        # the width of the wall is 4, which can be customized in image/flag.py
        # after adding four wall borders, the shape of the map is still [80,80]
        self.draw.draw_border(self.observ_0)

        if self.sg.V['GODS_PERSPECTIVE']:
            obs = self.sg.V['OBSTACLE']
            for ob in obs:
                self.draw.draw_obstacle(x=ob[0], y=ob[1], width=ob[2], height=ob[3], map=self.observ_0)

            for index in range(self.poi_data_pos.shape[0]):
                self.draw.draw_point(x=self.poi_data_pos[index, 0], y=self.poi_data_pos[index, 1],
                                     value=self.init_poi_data_val[index], map=self.observ_0)

        self.observ[:, :, 0] = copy.deepcopy(self.observ_0)

        if self.sg.V['GODS_PERSPECTIVE']:
            self.draw_power_station(self.observ[:, :, 2])
        # loop of uav
        for i in range(self.uav_num):
            # draw uav
            # draw uavs in the map (self._image_position[i_n], i_n is the id of uav)
            # the position of uav is [x*4+8,y*4+8] of the [80,80] map,
            # where x,y->[0~15]
            # the size of uav is [4,4]
            # the value of uav is 1.

            self.draw.draw_UAV(self.sg.V['INIT_POSITION'][i][0], self.sg.V['INIT_POSITION'][i][1],
                               self.sg.V['ENERGY'] / self.sg.V['ENERGY'],
                               self.observ[:, :, 2])

    def draw_power_station(self, map):

        for power in self.pow_pos:
            self.draw.draw_point(x=power[0], y=power[1], value=self.sg.V['POWER_VALUE'],
                                 map=map)

    def cal_energy_consuming(self, poi_data=0, distance=0, add_energy=0):
        energy_of_poi = poi_data * self.sg.V['POI_ENERGY_FACTOR']
        energy_of_dis = distance * self.sg.V['DISTANCE_ENERGY_FACTOR']
        return energy_of_poi + energy_of_dis + add_energy

    def cal_uav_next_pos(self, i, cur_pos, action, ):
        move_distance = np.sqrt(np.power(action[0], 2) + np.power(action[1], 2))
        energy_consuming = self.cal_energy_consuming(poi_data=0, distance=move_distance)
        dx = action[0]
        dy = action[1]
        # # move distance is larger than max distance
        # if move_distance > self.maxdistance:
        #     dx = action[0] * (self.maxdistance / move_distance)
        #     dy = action[1] * (self.maxdistance / move_distance)
        #     # move distance is less than max distance

        # energy is enough
        if self.cur_uav_energy[i] >= energy_consuming:
            new_x = cur_pos[0] + dx
            new_y = cur_pos[1] + dy

        else:
            # energy is not enough
            new_x = cur_pos[0] + dx * (self.cur_uav_energy[i] / energy_consuming)
            new_y = cur_pos[1] + dy * (self.cur_uav_energy[i] / energy_consuming)

        return (new_x, new_y)

    def get_poi_heatmap(self):
        map_vis_time = np.zeros((self.map_size_x, self.map_size_y), dtype=np.float32)
        for poi_index, poi_pos in enumerate(self.poi_data_pos):
            dx = (int)(poi_pos[0])
            dy = (int)(poi_pos[1])
            map_vis_time[dx][dy] += self.poi_access_times[poi_index]
        map_vis_time /= np.sum(map_vis_time)
        return map_vis_time

    def cal_distance(self, pos1, pos2):
        return np.sqrt(
            np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2)
        )

    def judge_obstacle(self, pos, next_pos):
        try:
            if 0 <= next_pos[0] <= self.map_size_x \
                    and 0 <= next_pos[1] <= self.map_size_y \
                    and self.is_valid(pos, next_pos):
                return False
        except:
            #
            pass
        return True

    def in_obstacle(self, x, y):
        for x0, y0, dx, dy in self.obstacle:
            if x > x0 and x < (x0 + dx) and y > y0 and y < (y0 + dy):
                return True
        return False

    def is_valid(self, pos, next_pos):
        if self.in_obstacle(next_pos[0], next_pos[1]):
            return False
        else:
            return True

    def update_observ_0(self, pos, is_obstacle, pos_val=None):

        if is_obstacle:
            # update obstacle
            if self.map_obstacle[myint(pos[0])][myint(pos[1])] != self.map_obstacle_value:
                # border do not need to be update
                return
                # get obstacle information by pos
            # obstacle = self.pos_obstacle_dict[(myint(pos[0]), myint(pos[1]))]

            # draw obstacle in observ_0
            self.draw.draw_obstacle(myint(pos[0]), myint(pos[1]), 1, 1, self.observ_0)
        else:
            # update poi
            self.draw.draw_point(x=pos[0], y=pos[1], value=pos_val, map=self.observ_0)

    def update_observ_1(self, pos, is_power=False, visit_times=None, i=None):

        if is_power:
            # update power station pos
            self.draw.draw_point(x=pos[0], y=pos[1], value=self.sg.V['POWER_VALUE'], map=self.observ[i, :, :, 2])
        else:
            # update visit times
            self.draw.draw_point(x=pos[0], y=pos[1], value=visit_times, map=self.observ_1)

    def update_observ(self, i, cur_pos, pre_pos):

        self.draw.clear_uav(pre_pos[0], pre_pos[1], self.observ[:, :, 2])
        self.draw.draw_UAV(cur_pos[0], cur_pos[1], self.cur_uav_energy[i] / self.sg.V['ENERGY'],
                           self.observ[:, :, 2])

        if self.sg.V['GODS_PERSPECTIVE']:
            self.draw_power_station(self.observ[:, :, 2])

    def is_uav_out_of_energy(self, i):
        return self.cur_uav_energy[i] < self.sg.V['EPSILON']

    def poi_data_collection(self, i, uav_pos):

        for poi_index, (poi_pos, poi_val) in enumerate(zip(self.poi_data_pos, self.cur_poi_data_val)):

            # cal distance between uav and poi
            distance = self.cal_distance(uav_pos, poi_pos)

            # if distance is within collecting range
            if distance <= self.sg.V['RANGE']:

                # update observ1 according to access times
                self.update_observ_1(
                    pos=self.poi_data_pos[poi_index],
                    is_power=False,
                    visit_times=self.poi_access_times[poi_index]
                )

                # collect data
                collected_data = min(self.init_poi_data_val[poi_index] * self.sg.V['COLLECTION_PROPORTION']
                                     , self.cur_poi_data_val[poi_index])
                # this poi has no data left
                if collected_data <= self.sg.V['EPSILON']:
                    continue

                # update access times
                self.poi_access_times[poi_index] += self.sg.V['VISIT']
                # cal energy consuming
                energy = self.cal_energy_consuming(poi_data=collected_data)

                # uav energy is not enough
                if energy > self.cur_uav_energy[i]:
                    collected_data = collected_data * (self.cur_uav_energy[i] / energy)

                # update uav data collection

                self.step_uav_data_collection[i] += collected_data

                # update current poi data value
                self.cur_poi_data_val[poi_index] -= collected_data
                self.cur_poi_data_val[poi_index] = max(0, self.cur_poi_data_val[poi_index])

                # update poi value in observation0
                self.update_observ_0(
                    pos=self.poi_data_pos[poi_index],
                    is_obstacle=False,
                    pos_val=self.cur_poi_data_val[poi_index]
                )

                # update uav energy
                # print(self.log.rank, 'energy', energy, 'cur_uav_energy[i]', self.cur_uav_energy[i])
                self.uav_energy_consuming[i] += min(energy, self.cur_uav_energy[i])
                self.cur_uav_energy[i] = max(self.cur_uav_energy[i] - energy, 0)

                # update uav dead list
                if self.is_uav_out_of_energy(i):
                    self.dead_uav_list[i] = True

                # update total data collection
                self.total_collected_data[i] += collected_data
                self.cur_collected_data[i] += collected_data

                if self.dead_uav_list[i]:
                    return

    def power_charging(self, i, uav_pos):
        for power_index, power_pos in enumerate(self.pow_pos):

            # cal distance between uav and power
            distance = self.cal_distance(uav_pos, power_pos)

            # if distance is within collecting range
            if distance <= self.sg.V['RANGE']:
                # charge power
                collected_power = min(self.sg.V['DELTA_ENERGY'], self.sg.V['ENERGY'] - self.cur_uav_energy[i])
                collected_power = min(collected_power, self.pow_remain_energy[power_index])
                self.pow_remain_energy[power_index] -= collected_power
                # update uav data collection
                self.step_uav_power_collection[i] = collected_power
                if not self.sg.V['GODS_PERSPECTIVE']:
                    # update power pos in observation2
                    self.update_observ_1(
                        pos=power_pos,
                        is_power=True,
                        i=i
                    )

                # update uav energy
                self.cur_uav_energy[i] += collected_power
                self.uav_energy_charging[i] += collected_power
                self.charge_counter[i] += 1
                self.cur_collected_power[i] = collected_power
                # self.uav_charge_trace[i].append(
                #     (self.current_step, copy.deepcopy(self.cur_uav_pos[i]), collected_power, self.cur_uav_energy[i]))

                # only charge once
                return
        self.cur_collected_power[i] = 0.
        # self.uav_charge_trace[i].append(
        #     (self.current_step, copy.deepcopy(self.cur_uav_pos[i]), 0, self.cur_uav_energy[i]))

    def cal_reward(self, i, distance):
        poi_energy_consuming = self.cal_energy_consuming(
            poi_data=self.step_uav_data_collection[i],
            distance=distance
        )
        # poi_r: the reward paid for collecting data
        poi_r = self.step_uav_data_collection[i] / (poi_energy_consuming + self.sg.V['EPSILON'])
        pow_r = (self.step_uav_power_collection[i] / self.sg.V['ENERGY'])
        return poi_r + pow_r * self.pow_reward_ratio

    def cal_charge_reward(self, charge_ratio):
        if charge_ratio > 0.2:
            return 1
        else:
            return 0

    def cal_data_reward(self, data_collection):
        if data_collection >= self.goal:
            return 1.
        else:
            return 0.

    def get_fairness(self):
        values = self.poi_access_times
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square < 1e-4:
            return 0.
        jain_fairness_index = square_of_sum / sum_of_square / float(len(values))
        return jain_fairness_index

    def get_observation(self):
        self.observ[:, :, 0] = self.observ_0[:, :]
        self.observ[:, :, 1] = self.observ_1[:, :]

        return copy.deepcopy(self.observ)  # [W,H,C]

    def get_observation_trans(self):
        self.observ[:, :, 0] = self.observ_0[:, :]
        self.observ[:, :, 1] = self.observ_1[:, :]

        return copy.deepcopy(self.observ).transpose([2, 0, 1])  # C,W,H

    def step(self, action, current_step=None):
        self.cur_collected_data = np.zeros(shape=[self.uav_num], dtype=np.float32)
        self.cur_collected_power = np.zeros(shape=[self.uav_num], dtype=np.float32)

        if current_step is not None:
            self.current_step = current_step
        # action [K,3]
        self.action[:, :] = np.reshape(np.copy(action), [self.uav_num, -1])

        self.action = np.clip(self.action, -1e3, 1e3)  #

        self.uav_reward_list = np.zeros([self.uav_num], dtype=np.float32)
        self.uav_penalty_list = np.zeros([self.uav_num], dtype=np.float32)
        self.step_uav_data_collection = np.zeros([self.uav_num], dtype=np.float32)
        self.step_uav_power_collection = np.zeros([self.uav_num], dtype=np.float32)

        self.uav_heat_map = np.zeros([self.uav_num, 20], dtype=np.float32)
        # loop of uavupdate_observ_0

        for i in range(self.uav_num):
            # skip the uav which runs out of energy
            if self.dead_uav_list[i]:
                continue

            penalty = 0

            # cal uav next position
            uav_next_pos = self.cal_uav_next_pos(i, self.cur_uav_pos[i], self.action[i, 1:])
            move_distance = self.cal_distance(self.cur_uav_pos[i], uav_next_pos)
            bump_obstacle = False
            # --------------PENALTY ONE: if obstacle is in next position ------------------------
            if self.judge_obstacle(self.cur_uav_pos[i], uav_next_pos):
                # cal move distance
                bump_obstacle = True
                # add obstacle penalty: -1 * 0.1
                penalty += self.sg.V['OBSTACLE_PENALTY'] * self.sg.V['NORMALIZE']

                # update observation_0
                if not self.sg.V['GODS_PERSPECTIVE']:
                    self.update_observ_0(uav_next_pos, is_obstacle=True)

                # stand still
                uav_next_pos = copy.deepcopy(self.cur_uav_pos[i])

            # update uav current position
            cur_pos = copy.deepcopy(uav_next_pos)
            pre_pos = copy.deepcopy(self.cur_uav_pos[i])
            self.cur_uav_pos[i] = copy.deepcopy(uav_next_pos)

            # add uav current position to trace
            self.uav_trace[i].append(copy.deepcopy(self.cur_uav_pos[i]))

            self.uav_energy_consuming[i] += min(self.cal_energy_consuming(distance=move_distance),
                                                self.cur_uav_energy[i])
            self.cur_uav_energy[i] = max(self.cur_uav_energy[i] - self.cal_energy_consuming(distance=move_distance), 0)

            # judge whether a uav is out of energy
            if self.is_uav_out_of_energy(i):
                self.dead_uav_list[i] = True
                self.uav_state[i].append(int(0))  # 0-> uav out of energy
                continue
            collect_data_flag = True
            # collect data
            if self.action[i, 0] > 0:
                collect_data_flag = True

                self.uav_state[i].append(1)  # uav collects data

                self.uav_heat_map[i, 0:10] = 1

                self.poi_data_collection(i, self.cur_uav_pos[i])

            # charge
            else:

                collect_data_flag = False

                self.uav_state[i].append(-1)  # uav charges power

                self.uav_heat_map[i, 10:] = 1

                self.power_charging(i, self.cur_uav_pos[i])

            # update uav energy and pos
            self.update_observ(i=i, cur_pos=cur_pos, pre_pos=pre_pos)

            # cal reward
            charge_ratio = self.step_uav_power_collection[i] / self.sg.V['ENERGY']
            reward = self.cal_charge_reward(charge_ratio)

            # ------------------PENALTY TWO : mini energy penalty--------------------
            if self.cur_uav_energy[i] < self.sg.V['MINI_ENEGY']:
                penalty += self.sg.V['NORMALIZE'] * self.sg.V['LESS_ENEGY_PENALTY']

            self.uav_penalty_list[i] = penalty

            if bump_obstacle:
                self.uav_reward_list[i] = 0
            else:
                self.uav_reward_list[i] = reward

            if collect_data_flag:
                self.uav_trace_info[i].append((self.current_step, 'collect-data', copy.deepcopy(self.cur_uav_pos[i]),
                                               copy.deepcopy(self.cur_collected_data[i]),
                                               copy.deepcopy(self.cur_uav_energy[i]), reward, penalty))
            else:
                self.uav_trace_info[i].append((self.current_step, 'charge-power', copy.deepcopy(self.cur_uav_pos[i]),
                                               copy.deepcopy(self.cur_collected_power[i]),
                                               copy.deepcopy(self.cur_uav_energy[i]), reward, penalty))

        # calculate reward
        data_collection_ratio = self.data_collection_ratio()
        data_reward = self.cal_data_reward(data_collection_ratio)
        if data_collection_ratio >= self.goal:
            self.goal += 0.05
            if data_collection_ratio >= 0.6:
                data_reward = 5.

        fairness = self.get_fairness()
        e_cur = self.cal_left_energy_ratio()
        e_cur = min(e_cur + self.mini_energy_ratio, 1.)
        total_uav_reward = e_cur * fairness * np.mean(self.uav_reward_list) * 2 + np.mean(self.uav_penalty_list) * 2

        total_uav_reward /= self.sg.V['NORMALIZE']
        self.episodic_total_uav_reward += total_uav_reward

        # done = False if False in self.dead_uav_list else True
        done = False

        observation = self.get_observation()

        # pos = [np.stack([[15.9, 8.0]])]
        pos_array = np.zeros(shape=[len(self.cur_uav_pos), 2],
                             dtype=np.float32)  # batch,num_of_uav,2

        pos_array[:, :] = self.cur_uav_pos[:][:]

        total_uav_reward = np.mean(self.uav_penalty_list) * 2 + np.mean(self.uav_reward_list) * 2 + data_reward

        return observation, total_uav_reward, done, self.uav_heat_map, pos_array, np.copy(
            self.cur_uav_energy / self.sg.V['ENERGY'])

    def cal_left_energy_ratio(self):
        return np.sum(self.cur_uav_energy) / np.sum(self.init_uav_energy)

    def data_collection_ratio(self):
        return np.sum(self.total_collected_data) / self.totaldata

    def energy_consumption_ratio(self):
        res = np.zeros(self.uav_num)
        for i in range(self.uav_num):
            res[i] = self.uav_energy_consuming[i] / (self.init_uav_energy[i] + self.uav_energy_charging[i])
        return np.mean(res)

    def geographical_fairness(self):
        return self.get_fairness()

    def energy_efficiency_1(self):
        res = np.zeros(self.uav_num)
        for i in range(self.uav_num):
            res[i] = self.total_collected_data[i] / self.uav_energy_consuming[i]
        return self.get_fairness() * np.mean(res)

    def get_uav_pos(self):
        return self.cur_uav_pos

    def get_data_coverage(self):
        res = np.zeros_like(self.init_poi_data_val)
        for i in range(np.shape(res)[0]):
            res[i] = (self.init_poi_data_val[i] - self.cur_poi_data_val[i]) / self.init_poi_data_val[i]
        return res.mean()

    def draw_convert(self, observ, img, max_val, color, is_power=False):
        pow_list = []
        for i in range(80):
            for j in range(80):

                if observ[i, j] < 0 and is_power == False:
                    img[i, j, 0] = np.uint8(255)
                elif observ[i, j] < 0 and is_power == True:
                    img[i, j, 2] = np.uint8(255)
                    pow_list.append((i, j))
                elif observ[i, j] > 0:

                    img[i, j, :] = np.uint8(color * (observ[i, j] / max_val))

        if len(pow_list) > 0:
            return pow_list
