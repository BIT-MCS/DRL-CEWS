from sparse_environment.sparse_environment_uav1_charge2.image.map import Map
from common_setting.env_setting import FLAGS
from PIL import Image
import time
import os


class MapS(Map):

    def __init__(self, width=80, height=80):
        super(MapS, self).__init__(width, height)
        self.__time = time.time()
        self.full_path = os.path.join(FLAGS.dir_sum, 'img_map')
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

    def getmap(self, map):
        wall = FLAGS.wall_value
        width = FLAGS.wall_width
        for j in range(0, 80, 1):
            for i in range(80 - width, 80, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
            for i in range(0, width, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
        for i in range(0, 80, 1):
            for j in range(0, width, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
            for j in range(80 - width, 80, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)

    def __trans(self, x, y):
        return 8 * x + 8, y * 8 + 8

    def draw_point(self, x, y, value, map):
        self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 4, 4, value, map)

    def draw_UAV(self, x, y, map):
        self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        # value = self.get_value(x, y)
        self.draw_sqr(x, y, 8, 8, 1, map)
        # self.draw_sqr(x, y, 4, 4, value)

    def clear_cell(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 8, 8, 0, map)

    def draw_goal(self, x, y, map):
        x, y = self.__trans(x, y)
        # value = self.get_value(x, y, map)
        self.draw_sqr(x, y, 8, 8, 1, map)
        self.draw_sqr(x, y, 4, 4, 0, map)

    def save_as_png(self, map, ip=None):
        img = Image.fromarray(map * 255)
        img = img.convert('L')
        # img.show()
        if ip is None:
            name = time.time() - self.__time
        else:
            name = str(ip)
        img.save(os.path.join(FLAGS.dir_sum, 'img_map', str(name)), 'png')
