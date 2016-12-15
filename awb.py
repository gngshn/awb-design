#!/usr/bin/env python3
# author: grant_shen

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import abc


def awb_locus(x, a, b):
    return a / x + b


def awb_locus2(x, a, b, c):
    return a / (x + b) + c


class Plane(object):
    _min_t = 1800  # min color temperature
    _max_t = 15000  # max color temperature
    _step = 50  # color temperature step
    _th = 0.01  # rg & bg min threshold
    _duv = (-0.02, 0.02, -0.02, 0.02)  # max delta uv for all illuminant

    _t = ()
    index = ()

    def __init__(self):
        self._line_h = self._line_v = np.array([])
        self._illuminant_h = self._illuminant_v = np.array([])
        self._cct_h = self._cct_v = np.array([])
        self._h_limit = self._v_limit = np.array([])
        self._h_ticks = self._v_ticks = 0
        self._x_label = self._y_label = ''
        self._illuminant_name = ''

    def get_line(self):
        return self._line_h, self._line_v

    def get_cct(self):
        return self._cct_h, self._cct_v

    def calc_cct(self, plane):
        cct_h, cct_v = plane.get_cct()
        self._cct_h, self._cct_v = self.get_value(cct_h, cct_v)

    def get_illuminant(self):
        return self._illuminant_h, self._illuminant_v

    def draw(self):
        plt.figure(figsize=(16, 12))
        plt.gca().set_xticks(self._h_ticks)
        plt.gca().set_yticks(self._v_ticks)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(self._x_label, fontsize=24)
        plt.ylabel(self._y_label, fontsize=24)
        plt.gca().set_aspect('equal')
        plt.xlim(self._h_limit)
        plt.ylim(self._v_limit)
        plt.grid()
        plt.plot(self._line_h, self._line_v, linewidth=3, color="black")
        i = 0
        for index in self.index:
            if self.get_t()[index] < 5500:
                plt.plot(self._cct_h[i, :], self._cct_v[i, :], color="red",
                         linewidth=0.5)
            else:
                plt.plot(self._cct_h[i, :], self._cct_v[i, :], color="blue",
                         linewidth=0.5)
            i += 1
        # noinspection PyTypeChecker
        plt.scatter(self._illuminant_h[[0, 1, 2, 7]],
                    self._illuminant_v[[0, 1, 2, 7]],
                    s=100, color="y", marker='o', label='ABCE')
        plt.scatter(self._illuminant_h[3:7], self._illuminant_v[3:7],
                    s=100, color="c", marker=',', label='D50556575')
        plt.scatter(self._illuminant_h[8:], self._illuminant_v[8:],
                    s=50, color="m", marker='*', label='F1-F12')
        plt.savefig(self.__class__.__name__ + '.png', format='png')
        plt.show()

    @staticmethod
    def get_t():
        return Plane._t

    @abc.abstractmethod
    def get_value(self, h, v):
        """Method that should do something."""


class XyPlane(Plane):
    def __init__(self, uv):
        super().__init__()
        u, v = uv.get_line()
        self._line_h, self._line_v = self.get_value(u, v)
        Plane._illuminant_name, self._illuminant_h, self._illuminant_v = \
            self.get_xy_illuminant()
        self._h_limit = (0.25, 0.55)
        self._v_limit = (0.25, 0.45)
        self._h_ticks = self._v_ticks = np.arange(0.25, 0.55, 0.02)
        self._x_label = 'x'
        self._y_label = 'y'

    def get_value(self, u, v):
        x = 3 * u / (2 * u - 8 * v + 4)
        y = 2 * v / (2 * u - 8 * v + 4)
        return x, y

    @staticmethod
    def get_xy_illuminant():
        return (('A', 'B', 'C', 'D50', 'D55', 'D65', 'D75', 'E',
                 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                 'F7', 'F8', 'F9', 'F10', 'F11', 'F12'),
                np.array([0.44757, 0.34842, 0.31006, 0.34567, 0.33242, 0.31271,
                          0.29902, 1 / 3, 0.31310, 0.37208, 0.40910, 0.44018,
                          0.31379, 0.37790, 0.31292, 0.34588, 0.37417, 0.34609,
                          0.38052, 0.43695]),
                np.array([0.40745, 0.35161, 0.31616, 0.35850, 0.34743, 0.32902,
                          0.31485, 1 / 3, 0.33727, 0.37529, 0.39430, 0.40329,
                          0.34531, 0.38835, 0.32933, 0.35875, 0.37281, 0.35986,
                          0.37713, 0.40441]))

    @staticmethod
    def get_planck_locus():
        base_t = 1667
        t1 = np.arange(base_t, 2223)
        x1 = (-0.2661239e9 / (t1 * t1 * t1) - 0.2343580e6 / (t1 * t1) +
              0.8776956e3 / t1 + 0.179910)
        y1 = (-1.1063814 * x1 * x1 * x1 - 1.34811020 * x1 * x1 +
              2.18555832 * x1 - 0.20219683)
        t2 = np.arange(2223, 4001)
        x2 = (-0.2661239e9 / (t2 * t2 * t2) - 0.2343580e6 / (t2 * t2) +
              0.8776956e3 / t2 + 0.179910)
        y2 = (-0.9549476 * x2 * x2 * x2 - 1.37418593 * x2 * x2 +
              2.09137015 * x2 - 0.16748867)
        t3 = np.arange(4001, 25001)
        x3 = (-3.0258469e9 / (t3 * t3 * t3) + 2.1070379e6 / (t3 * t3) +
              0.2226347e3 / t3 + 0.240390)
        y3 = (3.0817580 * x3 * x3 * x3 - 5.87338670 * x3 * x3 +
              3.75112997 * x3 - 0.37001483)
        return (np.hstack((x1, x2, x3)), np.hstack((y1, y2, y3)),
                np.hstack((t1, t2, t3)))


class UvPlane(Plane):
    def __init__(self):
        super().__init__()
        self._line_h, self._line_v, Plane._t = self.get_planck_locus()
        name, x, y = XyPlane.get_xy_illuminant()
        self._illuminant_h, self._illuminant_v = self.get_value(x, y)
        self._h_limit = (0.16, 0.32)
        self._v_limit = (0.26, 0.38)
        self._h_ticks = np.arange(0.16, 0.32, 0.02)
        self._v_ticks = np.arange(0.26, 0.38, 0.02)
        self._x_label = 'u'
        self._y_label = 'v'

    def calc_cct(self, plane):
        duv = self._duv
        base_t = plane.get_t()[0]
        index = np.arange((self._min_t - base_t), (self._max_t - base_t + 1),
                          self._step)
        Plane.index = index
        numbers = index.shape[0]
        di = 1
        cnt = 0
        for i in index:
            cnt += 1
            duv_min = duv[0] + (duv[2] - duv[0]) * cnt / numbers
            duv_max = duv[1] + (duv[3] - duv[1]) * cnt / numbers
            uv = np.linspace(duv_min, duv_max, 1001)
            du = self._line_h[i] - self._line_h[i - di]
            dv = self._line_v[i] - self._line_v[i - di]
            k = - du / dv
            if self._cct_h != ():
                self._cct_h = \
                    np.vstack((self._cct_h,
                               self._line_h[i] + uv / np.sqrt(k * k + 1)))
                self._cct_v = \
                    np.vstack((self._cct_v,
                               self._line_v[i] + k * uv / np.sqrt(k * k + 1)))
            else:
                self._cct_h = self._line_h[i] + uv / np.sqrt(k * k + 1)
                self._cct_v = self._line_v[i] + k * uv / np.sqrt(k * k + 1)

    def get_value(self, x, y):
        u = 4 * x / (12 * y - 2 * x + 3)
        v = 6 * y / (12 * y - 2 * x + 3)
        return u, v

    @staticmethod
    def get_planck_locus():
        t = np.arange(1000, 15000 + 1)
        u = ((0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) /
             (1 + 8.42420235e-4 * t + 7.08145163e-7 * t * t))
        v = ((0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) /
             (1 - 2.89741816e-5 * t + 1.61456053e-7 * t * t))
        return u, v, t


class RgBgPlane(Plane):
    def __init__(self, xy):
        super().__init__()
        x, y = xy.get_line()
        self._line_h, self._line_v = self.get_value(x, y)
        x, y = xy.get_illuminant()
        self._illuminant_h, self._illuminant_v = self.get_value(x, y)
        self._h_limit = (0, 3.6)
        self._v_limit = (0, 2)
        self._h_ticks = np.arange(0, 3.6, 0.2)
        self._v_ticks = np.arange(0, 2, 0.2)
        self._x_label = 'R/G'
        self._y_label = 'B/G'

    def calc_cct(self, plane):
        cct_h, cct_v = plane.get_cct()
        self._cct_h, self._cct_v = self.get_value(cct_h, cct_v)
        self.clip_value()

    # noinspection PyUnresolvedReferences
    def clip_value(self):
        for i in range(0, self._cct_h.shape[0]):
            for j in range(0, self._cct_h.shape[1]):
                if self._cct_h[i][j] <= 0:
                    self._cct_h[i][j] = np.NaN
        for i in range(0, self._cct_v.shape[0]):
            for j in range(0, self._cct_v.shape[1]):
                if self._cct_v[i][j] <= 0:
                    self._cct_v[i][j] = np.NaN

    def get_value(self, x, y):
        rg = ((0.41847 * x - 0.15866 * y - 0.082835 * (1 - x - y)) /
              (-0.091169 * x + 0.25243 * y + 0.015708 * (1 - x - y)))
        bg = ((0.0009209 * x - 0.0025498 * y + 0.1786 * (1 - x - y)) /
              (-0.091169 * x + 0.25243 * y + 0.015708 * (1 - x - y)))
        return rg, bg

    def draw(self):
        plt.figure(figsize=(16, 12))
        plt.gca().set_xticks(self._h_ticks)
        plt.gca().set_yticks(self._v_ticks)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(self._x_label, fontsize=24)
        plt.ylabel(self._y_label, fontsize=24)
        plt.gca().set_aspect('equal')
        plt.xlim(self._h_limit)
        plt.ylim(self._v_limit)
        plt.grid()
        plt.plot(self._line_h, self._line_v, linewidth=3, color="black")
        i = 0
        for index in self.index:
            if self.get_t()[index] < 5500:
                plt.plot(self._cct_h[i, :], self._cct_v[i, :], color="red",
                         linewidth=0.5)
            else:
                plt.plot(self._cct_h[i, :], self._cct_v[i, :], color="blue",
                         linewidth=0.5)
            i += 1
        # noinspection PyUnresolvedReferences
        plt.scatter(self._illuminant_h[[0, 1, 2, 7]],
                    self._illuminant_v[[0, 1, 2, 7]],
                    s=100, color="y", marker='o', label='A B C E')
        plt.scatter(self._illuminant_h[3:7], self._illuminant_v[3:7],
                    s=100, color="c", marker=',', label='D50556575')
        plt.scatter(self._illuminant_h[8:], self._illuminant_v[8:],
                    s=50, color="m", marker='*', label='F1-F12')
        # noinspection PyUnresolvedReferences
        p_opt, p_cov = opt.curve_fit(awb_locus,
                                     self._illuminant_h[[0, 3, 4, 5, 6, 9,
                                                         14, 15, 17, 19]],
                                     self._illuminant_v[[0, 3, 4, 5, 6, 9,
                                                         14, 15, 17, 19]])
        print("In theory: y = ", p_opt[0], " / x + ", p_opt[1])
        x = np.linspace(0.3, 3.5, num=1000)
        y = awb_locus(x, p_opt[0], p_opt[1])
        x1 = x2 = y1 = y2 = 0
        a1 = 0.70
        a2 = 1 / a1
        # noinspection PyTypeChecker
        y1 = awb_locus(x + x1, a1 * p_opt[0], p_opt[1]) - y1
        # noinspection PyTypeChecker
        y2 = awb_locus(x - x2, a2 * p_opt[0], p_opt[1]) + y2
        plt.plot(x, y, '--')
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.savefig(self.__class__.__name__ + '.png', format='png')
        plt.show()


class GbGrPlane(Plane):
    def __init__(self, rgbg):
        super().__init__()
        rg, bg = rgbg.get_line()
        self._line_h, self._line_v = self.get_value(rg, bg)
        rg, bg = rgbg.get_illuminant()
        self._illuminant_h, self._illuminant_v = self.get_value(rg, bg)
        self._h_limit = (0, 5)
        self._v_limit = (0, 3)
        self._h_ticks = np.arange(0, 5, 0.4)
        self._v_ticks = np.arange(0, 3, 0.4)
        self._x_label = 'G/B'
        self._y_label = 'G/R'

    def get_value(self, rg, bg):
        gb = 1 / bg
        gr = 1 / rg
        return gb, gr


def draw_our_locus(rgbg, x, y, title):
    illuminant_x, illuminant_y = rgbg.get_illuminant()
    """A, U40, U35, CWF, D50, D65 illuminant"""
    illuminant_x = illuminant_x[[0, 19, 10, 9, 3, 5]]
    illuminant_y = illuminant_y[[0, 19, 10, 9, 3, 5]]
    plt.figure()
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.xlabel('R / G', fontsize=24)
    plt.ylabel('B / G', fontsize=24)
    plt.gca().set_xticks(np.arange(0, 3, 0.5))
    plt.gca().set_yticks(np.arange(-0.5, 2, 0.5))
    plt.xlim(0, 3)
    plt.ylim(-0.5, 2)
    plt.scatter(illuminant_x, illuminant_y, color='b', marker='x')
    plt.scatter(x, y, color='r', marker='x')
    x = x[[0, 1, 2, 4, 5]]
    y = y[[0, 1, 2, 4, 5]]
    plt.scatter(x, y, color='g', marker='x')
    p_opt, p_cov = opt.curve_fit(awb_locus, x, y)
    print(title, ": y = ", p_opt[0], " / x + ", p_opt[1])
    x = np.linspace(0.11, 5, num=1000)
    y = awb_locus(x, p_opt[0], p_opt[1])
    plt.plot(x, y, '--')
    x1 = x2 = y1 = y2 = 0
    a1 = 0.70
    a2 = 1 / a1
    # noinspection PyTypeChecker
    y = awb_locus(x + x1, a1 * p_opt[0], p_opt[1]) - y1
    plt.plot(x, y)
    # noinspection PyTypeChecker
    y = awb_locus(x - x2, a2 * p_opt[0], p_opt[1]) + y2
    plt.plot(x, y)
    plt.savefig(title + '.png', format='png')
    plt.show()


def draw_sensor_awb_locus(rgbg):
    title = 'OV9726'
    x = np.array([1.0555555556, 1.0285714286, 0.9285714286,
                  0.6746987952, 0.6829268293, 0.6153846154])
    y = np.array([0.4027777778, 0.4428571429, 0.5595238095,
                  0.5180722892, 0.7682926829, 0.9230769231])
    draw_our_locus(rgbg, x, y, title)

    title = 'MI1040'
    x = np.array([1.1279069767, 1.0752688172, 1,
                  0.7741935484, 0.7875, 0.7])
    y = np.array([0.4418604651, 0.4516129032, 0.5421686747,
                  0.5322580645, 0.725, 0.88])
    draw_our_locus(rgbg, x, y, title)

    title = 'S5K6A1'
    x = np.array([1.3058823529, 1.2567567568, 1.1648351648,
                  0.8734177215, 0.8846153846, 0.7714285714])
    y = np.array([0.3882352941, 0.4054054054, 0.4945054945,
                  0.4683544304, 0.6442307692, 0.7857142857])
    draw_our_locus(rgbg, x, y, title)

    title = 'OV9732'
    x = np.array([32 / 39, 32 / 39, 32 / 66, 32 / 50, 32 / 54, 32 / 62])
    y = np.array([32 / 74, 32 / 73, 32 / 40, 32 / 66, 32 / 52, 32 / 46])
    draw_our_locus(rgbg, x, y, title)

    title = 'OV2710'
    x = np.array([32 / 26, 32 / 29, 32 / 47, 32 / 37, 32 / 40, 32 / 46])
    y = np.array([32 / 73, 32 / 69, 32 / 43, 32 / 71, 32 / 53, 32 / 46])
    draw_our_locus(rgbg, x, y, title)


def draw_daylight_cct_locus(xy):
    """
    daylight cct:
    4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400
    5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500 6600 6700 6800 6900
    7000 7100 7200 7300 7400 7500 7600 7700 7800 7900 8000 8100 8200 8300 8400
    8500 9000 9500 10000 11000 12000 13000 14000 15000 20000 25000
    """
    x1 = np.array([0.3823, 0.3779, 0.3737, 0.3697, 0.3658, 0.3621, 0.3585,
                   0.3551, 0.3519, 0.3487, 0.3457, 0.3429, 0.3401, 0.3375,
                   0.3349, 0.3325, 0.3302, 0.3279, 0.3258, 0.3237, 0.3217,
                   0.3198, 0.3179, 0.3161, 0.3144, 0.3128, 0.3112, 0.3097,
                   0.3082, 0.3067, 0.3054, 0.304, 0.3027, 0.3015, 0.3003,
                   0.2991, 0.298, 0.2969, 0.2958, 0.2948, 0.2938, 0.2928,
                   0.2919, 0.291, 0.2901, 0.2892, 0.2853, 0.2818, 0.2788,
                   0.2737, 0.2697, 0.2664, 0.2637, 0.2614, 0.2539, 0.2499])
    y1 = np.array([0.3838, 0.3812, 0.3786, 0.376, 0.3734, 0.3709, 0.3684,
                   0.3659, 0.3634, 0.361, 0.3587, 0.3564, 0.3541, 0.3519,
                   0.3497, 0.3476, 0.3455, 0.3435, 0.3416, 0.3397, 0.3378,
                   0.336, 0.3342, 0.3325, 0.3308, 0.3292, 0.3276, 0.326,
                   0.3245, 0.3231, 0.3216, 0.3202, 0.3189, 0.3176, 0.3163,
                   0.315, 0.3138, 0.3126, 0.3115, 0.3103, 0.3092, 0.3081,
                   0.3071, 0.3061, 0.3051, 0.3041, 0.2996, 0.2956, 0.292,
                   0.2858, 0.2808, 0.2767, 0.2732, 0.2702, 0.2603, 0.2548])
    x2, y2 = xy.get_line()
    x3, y3 = xy.get_illuminant()
    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.scatter(x3, y3, s=20, marker='o')
    plt.savefig('daylight.png', format='png')
    plt.show()


def draw_illuminant_box(xy, rgbg):
    plt.figure()
    plt.gca().set_aspect('equal')
    line_h, line_v = xy.get_line()
    cct_h, cct_v = xy.get_cct()
    plt.plot(line_h, line_v, linewidth=3, color="black")
    i = 0
    for index in xy.index:
        if xy.get_t()[index] < 5500:
            plt.plot(cct_h[i, :], cct_v[i, :], color="red",
                     linewidth=0.5)
        else:
            plt.plot(cct_h[i, :], cct_v[i, :], color="blue",
                     linewidth=0.5)
        i += 1

    my_illuminant_x = np.array([0.3163, 0.3482, 0.3945, 0.4546, 0.4708])
    my_illuminant_y = np.array([0.3303, 0.3609, 0.4011, 0.4068, 0.4163])
    plt.scatter(my_illuminant_x, my_illuminant_y, color='r', marker='x')
    illuminant_x, illuminant_y = xy.get_illuminant()
    """D65, D50, CWF, U30, A illuminant"""
    illuminant_x = illuminant_x[[5, 3, 9, 19, 0]]
    illuminant_y = illuminant_y[[5, 3, 9, 19, 0]]
    plt.scatter(illuminant_x, illuminant_y, color='b', marker='o')
    plt.savefig('our_xy.png', format='png')
    plt.show()

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.xlim(0.5, 2.5)
    plt.ylim(0, 1.5)
    my_illuminant_rg, my_illuminant_bg = rgbg.get_value(my_illuminant_x,
                                                        my_illuminant_y)
    illuminant_rg, illuminant_bg = rgbg.get_value(illuminant_x, illuminant_y)
    line_h, line_v = rgbg.get_line()
    cct_h, cct_v = rgbg.get_cct()
    plt.plot(line_h, line_v, linewidth=3, color="black")
    i = 0
    for index in rgbg.index:
        if rgbg.get_t()[index] < 5500:
            plt.plot(cct_h[i, :], cct_v[i, :], color="red",
                     linewidth=0.5)
        else:
            plt.plot(cct_h[i, :], cct_v[i, :], color="blue",
                     linewidth=0.5)
        i += 1
    plt.scatter(my_illuminant_rg, my_illuminant_bg, color='r', marker='x')
    plt.scatter(illuminant_rg, illuminant_bg, color='b', marker='o')
    plt.savefig('our_rgbg.png', format='png')
    plt.show()


class LocusAb(object):
    @staticmethod
    def draw_our_locus1(rgbg, rg, bg, title):
        plt.figure()
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.xlabel('R / G', fontsize=24)
        plt.ylabel('B / G', fontsize=24)
        plt.gca().set_xticks(np.arange(-1, 5, 0.5))
        plt.gca().set_yticks(np.arange(-1, 5, 0.5))
        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        illuminant_x, illuminant_y = rgbg.get_illuminant()
        illuminant_x = illuminant_x[[0, 19, 3, 5, 6]]
        illuminant_y = illuminant_y[[0, 19, 3, 5, 6]]
        plt.scatter(illuminant_x, illuminant_y, color='b', marker='x')
        rg = rg[[0, 1, 4, 5, 2]]
        bg = bg[[0, 1, 4, 5, 2]]
        plt.scatter(rg, bg, color='g', marker='x')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus, rg, bg, p0=(1, 0))
        h = np.linspace(0.0001, 5, num=1000)
        v = awb_locus(h, p_opt[0], p_opt[1])
        plt.plot(h, v, color='g')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus, illuminant_x, illuminant_y,
                                     p0=(1, 0))
        v = awb_locus(h, p_opt[0], p_opt[1])
        plt.plot(h, v, color='b')
        plt.show()

    @staticmethod
    def draw_minus_locus1(rgbg, rg, bg, title):
        illuminant_x, illuminant_y = rgbg.get_illuminant()
        illuminant_x = illuminant_x[[0, 19, 3, 5, 6]]
        illuminant_y = illuminant_y[[0, 19, 3, 5, 6]]
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus, illuminant_x, illuminant_y,
                                     p0=(1, 0))
        illuminant_y -= p_opt[1]
        plt.figure()
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.xlabel('R / G', fontsize=24)
        plt.ylabel('B / G', fontsize=24)
        plt.gca().set_xticks(np.arange(-1, 5, 0.5))
        plt.gca().set_yticks(np.arange(-1, 5, 0.5))
        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        plt.scatter(illuminant_x, illuminant_y, color='b', marker='x')
        rg = rg[[0, 1, 4, 5, 2]]
        bg = bg[[0, 1, 4, 5, 2]]
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus, rg, bg, p0=(1, 0))
        bg -= p_opt[1]
        plt.scatter(rg, bg, color='g', marker='x')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus, rg, bg, p0=(1, 0))
        h = np.linspace(0.0001, 5, num=1000)
        v = awb_locus(h, p_opt[0], p_opt[1])
        plt.plot(h, v, color='g')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus, illuminant_x, illuminant_y,
                                     p0=(1, 0))
        v = awb_locus(h, p_opt[0], p_opt[1])
        plt.plot(h, v, color='b')
        # noinspection SpellCheckingInspection
        colors = "rgbyc"
        h = np.linspace(-1, 5, num=1000)
        for i in range(5):
            plt.scatter(rg[i], bg[i], color=colors[i])
            plt.scatter(illuminant_x[i], illuminant_y[i], color=colors[i])
            k = (illuminant_y[i] - bg[i]) / (illuminant_x[i] - rg[i])
            # noinspection PyUnresolvedReferences
            v = k * (h - rg[i]) + bg[i]
            plt.plot(h, v)
        plt.show()

    @staticmethod
    def main(rgbg):
        title = 'OV9732 & theory diff'
        x = np.array([32 / 39, 32 / 39, 32 / 66, 32 / 50, 32 / 54, 32 / 62])
        y = np.array([32 / 74, 32 / 73, 32 / 40, 32 / 66, 32 / 52, 32 / 46])
        LocusAb.draw_our_locus1(rgbg, x, y, title)
        LocusAb.draw_minus_locus1(rgbg, x, y, title)
        title = 'OV2710 & theory diff'
        x = np.array([32 / 26, 32 / 29, 32 / 47, 32 / 37, 32 / 40, 32 / 46])
        y = np.array([32 / 73, 32 / 69, 32 / 43, 32 / 71, 32 / 53, 32 / 46])
        LocusAb.draw_our_locus1(rgbg, x, y, title)
        LocusAb.draw_minus_locus1(rgbg, x, y, title)
        title = 'AR0237 & theory diff'
        x = np.array([32 / 33, 32 / 31, 32 / 54, 32 / 40, 32 / 43, 32 / 51])
        y = np.array([32 / 96, 32 / 94, 32 / 42, 32 / 71, 32 / 53, 32 / 45])
        LocusAb.draw_our_locus1(rgbg, x, y, title)
        LocusAb.draw_minus_locus1(rgbg, x, y, title)


class LocusAbc(object):
    @staticmethod
    def draw_our_locus2(rgbg, rg, bg, title):
        plt.figure()
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.xlabel('R / G', fontsize=24)
        plt.ylabel('B / G', fontsize=24)
        plt.gca().set_xticks(np.arange(-1, 5, 0.5))
        plt.gca().set_yticks(np.arange(-1, 5, 0.5))
        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        illuminant_x, illuminant_y = rgbg.get_illuminant()
        illuminant_x = illuminant_x[[0, 19, 3, 5, 6]]
        illuminant_y = illuminant_y[[0, 19, 3, 5, 6]]
        plt.scatter(illuminant_x, illuminant_y, color='b', marker='x')
        rg = rg[[0, 1, 4, 5, 2]]
        bg = bg[[0, 1, 4, 5, 2]]
        plt.scatter(rg, bg, color='g', marker='x')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus2, rg, bg, p0=(1, 0, 0))
        h = np.linspace(0.0001, 5, num=1000)
        v = awb_locus2(h, p_opt[0], p_opt[1], p_opt[2])
        plt.plot(h, v, color='g')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus2, illuminant_x, illuminant_y,
                                     p0=(1, 0, 0))
        v = awb_locus2(h, p_opt[0], p_opt[1], p_opt[2])
        plt.plot(h, v, color='b')
        plt.show()

    @staticmethod
    def draw_minus_locus2(rgbg, rg, bg, title):
        illuminant_x, illuminant_y = rgbg.get_illuminant()
        illuminant_x = illuminant_x[[0, 19, 3, 5, 6]]
        illuminant_y = illuminant_y[[0, 19, 3, 5, 6]]
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus2, illuminant_x, illuminant_y,
                                     p0=(1, 0, 0))
        illuminant_x += p_opt[1]
        illuminant_y -= p_opt[2]
        plt.figure()
        plt.title(title, fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.xlabel('R / G', fontsize=24)
        plt.ylabel('B / G', fontsize=24)
        plt.gca().set_xticks(np.arange(-1, 5, 0.5))
        plt.gca().set_yticks(np.arange(-1, 5, 0.5))
        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        plt.scatter(illuminant_x, illuminant_y, color='b', marker='x')
        rg = rg[[0, 1, 4, 5, 2]]
        bg = bg[[0, 1, 4, 5, 2]]
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus2, rg, bg, p0=(1, 0, 0))
        rg += p_opt[1]
        bg -= p_opt[2]
        plt.scatter(rg, bg, color='g', marker='x')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus2, rg, bg, p0=(1, 0, 0))
        h = np.linspace(0.0001, 5, num=1000)
        v = awb_locus2(h, p_opt[0], p_opt[1], p_opt[2])
        plt.plot(h, v, color='g')
        # noinspection PyTypeChecker
        p_opt, p_cov = opt.curve_fit(awb_locus2, illuminant_x, illuminant_y,
                                     p0=(1, 0, 0))
        v = awb_locus2(h, p_opt[0], p_opt[1], p_opt[2])
        plt.plot(h, v, color='b')
        # noinspection SpellCheckingInspection
        colors = "rgbyc"
        h = np.linspace(-1, 5, num=1000)
        for i in range(5):
            plt.scatter(rg[i], bg[i], color=colors[i])
            plt.scatter(illuminant_x[i], illuminant_y[i], color=colors[i])
            k = (illuminant_y[i] - bg[i]) / (illuminant_x[i] - rg[i])
            # noinspection PyUnresolvedReferences
            v = k * (h - rg[i]) + bg[i]
            plt.plot(h, v)
        plt.show()

    @staticmethod
    def main(rgbg):
        title = 'OV9732 & theory diff'
        x = np.array([32 / 39, 32 / 39, 32 / 66, 32 / 50, 32 / 54, 32 / 62])
        y = np.array([32 / 74, 32 / 73, 32 / 40, 32 / 66, 32 / 52, 32 / 46])
        LocusAbc.draw_our_locus2(rgbg, x, y, title)
        LocusAbc.draw_minus_locus2(rgbg, x, y, title)
        title = 'OV2710 & theory diff'
        x = np.array([32 / 26, 32 / 29, 32 / 47, 32 / 37, 32 / 40, 32 / 46])
        y = np.array([32 / 73, 32 / 69, 32 / 43, 32 / 71, 32 / 53, 32 / 46])
        LocusAbc.draw_our_locus2(rgbg, x, y, title)
        LocusAbc.draw_minus_locus2(rgbg, x, y, title)
        title = 'AR0237 & theory diff'
        x = np.array([32 / 33, 32 / 31, 32 / 54, 32 / 40, 32 / 43, 32 / 51])
        y = np.array([32 / 96, 32 / 94, 32 / 42, 32 / 71, 32 / 53, 32 / 45])
        LocusAbc.draw_our_locus2(rgbg, x, y, title)
        LocusAbc.draw_minus_locus2(rgbg, x, y, title)


def main():
    np.seterr(invalid='ignore')

    uv = UvPlane()
    xy = XyPlane(uv)
    rgbg = RgBgPlane(xy)
    gbgr = GbGrPlane(rgbg)

    uv.calc_cct(xy)
    xy.calc_cct(uv)
    rgbg.calc_cct(xy)
    gbgr.calc_cct(rgbg)

    uv.draw()
    xy.draw()
    gbgr.draw()
    rgbg.draw()

    draw_daylight_cct_locus(xy)
    draw_sensor_awb_locus(rgbg)
    draw_illuminant_box(xy, rgbg)

    LocusAb.main(rgbg)
    LocusAbc.main(rgbg)


if __name__ == "__main__":
    main()
