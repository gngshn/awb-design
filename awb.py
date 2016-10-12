#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 13:37:29 2016

@author: grant_shen
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import abc


def awb_locus(x, a, b):
    return a / x + b

def awb_locus2(x, a, b, c):
    return a / (x * x) + b / x + c


class Plane(object):
    _mint = 1800  # min color temperature
    _maxt = 15000  # max color temperature
    _step = 50  # color temperature step
    _th = 0.01  # rg & bg min threshold
    _duv = (-0.012, 0.012, -0.016, 0.008)  # max delta uv for all illuminant

    _t = ()
    _index = ()

    def __init__(self):
        self._lineh = self._linev = ()
        self._illumh = self._illumv = ()
        self._ccth = self._cctv = ()
        self._hlim = self._vlim = ()
        self._hticks = self._vticks = 0
        self._xlabel = self._ylabel = ''
        self._illumName = ''

    def get_line(self):
        return self._lineh, self._linev

    def get_cct(self):
        return self._ccth, self._cctv

    def calc_cct(self, plane):
        ccth, cctv = plane.get_cct()
        self._ccth, self._cctv = self.get_value(ccth, cctv)

    def get_illum(self):
        return self._illumh, self._illumv

    def draw(self):
        plt.figure(figsize=(16, 12))
        plt.gca().set_xticks(self._hticks)
        plt.gca().set_yticks(self._vticks)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(self._xlabel, fontsize=24)
        plt.ylabel(self._ylabel, fontsize=24)
        plt.gca().set_aspect('equal')
        plt.xlim(self._hlim)
        plt.ylim(self._vlim)
        plt.grid()
        plt.plot(self._lineh, self._linev, linewidth=3, color="black")
        i = 0
        for index in self._index:
            if self.get_t()[index] < 5500:
                plt.plot(self._ccth[i, :], self._cctv[i, :], color="red",
                         linewidth=0.5)
            else:
                plt.plot(self._ccth[i, :], self._cctv[i, :], color="blue",
                         linewidth=0.5)
            i += 1
        plt.scatter(self._illumh[[0, 1, 2, 7]], self._illumv[[0, 1, 2, 7]],
                    s=100, color="y", marker='o', label='ABCE')
        plt.scatter(self._illumh[3:7], self._illumv[3:7],
                    s=100, color="c", marker=',', label='D50556575')
        plt.scatter(self._illumh[8:], self._illumv[8:],
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
        Plane.__init__(self)
        u, v = uv.get_line()
        self._lineh, self._linev = self.get_value(u, v)
        Plane._illumName, self._illumh, self._illumv = self.get_xy_illum()
        self._hlim = (0.25, 0.55)
        self._vlim = (0.25, 0.45)
        self._hticks = self._vticks = np.arange(0.25, 0.55, 0.02)
        self._xlabel = 'x'
        self._ylabel = 'y'

    def get_value(self, u, v):
        x = 3 * u / (2 * u - 8 * v + 4)
        y = 2 * v / (2 * u - 8 * v + 4)
        return x, y

    @staticmethod
    def get_xy_illum():
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
    def get_planckian_locus():
        BASET = 1667
        t1 = np.arange(BASET, 2223)
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
        Plane.__init__(self)
        self._lineh, self._linev, Plane._t = self.get_planckian_locus()
        name, x, y = XyPlane.get_xy_illum()
        self._illumh, self._illumv = self.get_value(x, y)
        self._hlim = (0.16, 0.32)
        self._vlim = (0.26, 0.38)
        self._hticks = np.arange(0.16, 0.32, 0.02)
        self._vticks = np.arange(0.26, 0.38, 0.02)
        self._xlabel = 'u'
        self._ylabel = 'v'

    def calc_cct(self):
        duv = self._duv
        BASET = xy.get_t()[0]
        index = np.arange((self._mint - BASET), (self._maxt - BASET + 1),
                          self._step)
        Plane._index = index
        max = index.shape[0]
        di = 1
        cnt = 0
        for i in index:
            cnt += 1
            duvmin = duv[0] + (duv[2] - duv[0]) * cnt / max
            duvmax = duv[1] + (duv[3] - duv[1]) * cnt / max
            # uv = np.arange(duvmin, duvmax + 0.00005, 0.00005)
            uv = np.linspace(duvmin, duvmax, 1001)
            du = self._lineh[i] - self._lineh[i - di]
            dv = self._linev[i] - self._linev[i - di]
            k = - du / dv
            if self._ccth != ():
                self._ccth = \
                    np.vstack((self._ccth,
                               self._lineh[i] + uv / np.sqrt(k * k + 1)))
                self._cctv = \
                    np.vstack((self._cctv,
                               self._linev[i] + k * uv / np.sqrt(k * k + 1)))
            else:
                self._ccth = self._lineh[i] + uv / np.sqrt(k * k + 1)
                self._cctv = self._linev[i] + k * uv / np.sqrt(k * k + 1)

    def get_value(self, x, y):
        u = 4 * x / (12 * y - 2 * x + 3)
        v = 6 * y / (12 * y - 2 * x + 3)
        return u, v

    @staticmethod
    def get_planckian_locus():
        t = np.arange(1000, 15000 + 1)
        u = ((0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) /
             (1 + 8.42420235e-4 * t + 7.08145163e-7 * t * t))
        v = ((0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) /
             (1 - 2.89741816e-5 * t + 1.61456053e-7 * t * t))
        return u, v, t


class RgBgPlane(Plane):
    def __init__(self, xy):
        Plane.__init__(self)
        x, y = xy.get_line()
        self._lineh, self._linev = self.get_value(x, y)
        x, y = xy.get_illum()
        self._illumh, self._illumv = self.get_value(x, y)
        self._hlim = (0, 3.6)
        self._vlim = (0, 2)
        self._hticks = np.arange(0, 3.6, 0.2)
        self._vticks = np.arange(0, 2, 0.2)
        self._xlabel = 'R/G'
        self._ylabel = 'B/G'

    def calc_cct(self, plane):
        ccth, cctv = plane.get_cct()
        self._ccth, self._cctv = self.get_value(ccth, cctv)
        self.clip_value()

    def clip_value(self):
        for i in range(0, self._ccth.shape[0]):
            for j in range(0, self._ccth.shape[1]):
                if self._ccth[i][j] <= 0:
                    self._ccth[i][j] = np.NaN
        for i in range(0, self._cctv.shape[0]):
            for j in range(0, self._cctv.shape[1]):
                if self._cctv[i][j] <= 0:
                    self._cctv[i][j] = np.NaN

    def get_value(self, x, y):
        rg = ((0.41847 * x - 0.15866 * y - 0.082835 * (1 - x - y)) /
              (-0.091169 * x + 0.25243 * y + 0.015708 * (1 - x - y)))
        bg = ((0.0009209 * x - 0.0025498 * y + 0.1786 * (1 - x - y)) /
              (-0.091169 * x + 0.25243 * y + 0.015708 * (1 - x - y)))
        return rg, bg

    def draw(self):
        plt.figure(figsize=(16, 12))
        plt.gca().set_xticks(self._hticks)
        plt.gca().set_yticks(self._vticks)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(self._xlabel, fontsize=24)
        plt.ylabel(self._ylabel, fontsize=24)
        plt.gca().set_aspect('equal')
        plt.xlim(self._hlim)
        plt.ylim(self._vlim)
        plt.grid()
        plt.plot(self._lineh, self._linev, linewidth=3, color="black")
        i = 0
        for index in self._index:
            if self.get_t()[index] < 5500:
                plt.plot(self._ccth[i, :], self._cctv[i, :], color="red",
                         linewidth=0.5)
            else:
                plt.plot(self._ccth[i, :], self._cctv[i, :], color="blue",
                         linewidth=0.5)
            i += 1
        plt.scatter(self._illumh[[0, 1, 2, 7]], self._illumv[[0, 1, 2, 7]],
                    s=100, color="y", marker='o', label='ABCE')
        plt.scatter(self._illumh[3:7], self._illumv[3:7],
                    s=100, color="c", marker=',', label='D50556575')
        plt.scatter(self._illumh[8:], self._illumv[8:],
                    s=50, color="m", marker='*', label='F1-F12')
        popt, pcov = opt.curve_fit(awb_locus,
                                   self._illumh[[0, 3, 4, 5, 6, 9,
                                                 14, 15, 17, 19]],
                                   self._illumv[[0, 3, 4, 5, 6, 9,
                                                 14, 15, 17, 19]])
        print("In theory: y = ", popt[0], " / x + ", popt[1])
        x = np.linspace(0.3, 3.5, num=1000)
        y = awb_locus(x, popt[0], popt[1])
        x1 = x2 = y1 = y2 = 0
        a1 = 0.80
        a2 = 1.25
        y1 = awb_locus(x + x1, a1 * popt[0], popt[1]) - y1
        y2 = awb_locus(x - x2, a2 * popt[0], popt[1]) + y2
        plt.plot(x, y)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.savefig(self.__class__.__name__ + '.png', format='png')
        plt.show()


class GbGrPlane(Plane):
    def __init__(self, rgbg):
        Plane.__init__(self)
        rg, bg = rgbg.get_line()
        self._lineh, self._linev = self.get_value(rg, bg)
        rg, bg = rgbg.get_illum()
        self._illumh, self._illumv = self.get_value(rg, bg)
        self._hlim = (0, 5)
        self._vlim = (0, 3)
        self._hticks = np.arange(0, 5, 0.4)
        self._vticks = np.arange(0, 3, 0.4)
        self._xlabel = 'G/B'
        self._ylabel = 'G/R'

    def get_value(self, rg, bg):
        gb = 1 / bg
        gr = 1 / rg
        return gb, gr


def draw_our_locus(x, y, title):
    illum_x, illum_y = rgbg.get_illum()
    """A, U40, U35, CWF, D50, D65 illuminant"""
    illum_x = illum_x[[0, 19, 10, 9, 3, 5]]
    illum_y = illum_y[[0, 19, 10, 9, 3, 5]]
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
    plt.scatter(illum_x, illum_y, color='b', marker='x')
    plt.scatter(x, y, color='r', marker='x')
    x = x[[0, 1, 2, 4, 5]]
    y = y[[0, 1, 2, 4, 5]]
    xbak = x
    ybak = y
    popt, pcov = opt.curve_fit(awb_locus, x, y)
    print(title, ": y = ", popt[0], " / x + ", popt[1])
    x = np.linspace(0.11, 5, num=1000)
    y = awb_locus(x, popt[0], popt[1])
    plt.plot(x, y)
    x1 = x2 = y1 = y2 = 0
    a1 = 0.80
    a2 = 1.25
    y = awb_locus(x + x1, a1 * popt[0], popt[1]) - y1
    plt.plot(x, y)
    y = awb_locus(x - x2, a2 * popt[0], popt[1]) + y2
    plt.plot(x, y)
#    popt, pcov = opt.curve_fit(awb_locus2, xbak, ybak)
#    y = awb_locus2(x, popt[0], popt[1])
#    plt.plot(x, y)
    plt.savefig(title + '.png', format='png')
    plt.show()

def draw_sensor_awb_locus():
    title = 'OV9726'
    x = np.array([1.0555555556, 1.0285714286, 0.9285714286,
                  0.6746987952, 0.6829268293, 0.6153846154])
    y = np.array([0.4027777778, 0.4428571429, 0.5595238095,
                  0.5180722892, 0.7682926829, 0.9230769231])
    draw_our_locus(x, y, title)

    title = 'MI1040'
    x = np.array([1.1279069767, 1.0752688172, 1,
                  0.7741935484, 0.7875, 0.7])
    y = np.array([0.4418604651, 0.4516129032, 0.5421686747,
                  0.5322580645, 0.725, 0.88])
    draw_our_locus(x, y, title)

    title = 'S5K6A1'
    x = np.array([1.3058823529, 1.2567567568, 1.1648351648,
                  0.8734177215, 0.8846153846, 0.7714285714])
    y = np.array([0.3882352941, 0.4054054054, 0.4945054945,
                  0.4683544304, 0.6442307692, 0.7857142857])
    draw_our_locus(x, y, title)

def draw_daylight_cct_locus():
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
    x3, y3 = xy.get_illum()
    plt.figure()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.scatter(x3, y3, s=20, marker='o')
    plt.savefig('daylight.png', format='png')
    plt.show()

np.seterr(invalid='ignore')

uv = UvPlane()
xy = XyPlane(uv)
rgbg = RgBgPlane(xy)
gbgr = GbGrPlane(rgbg)

uv.calc_cct()
xy.calc_cct(uv)
rgbg.calc_cct(xy)
gbgr.calc_cct(rgbg)

uv.draw()
xy.draw()
gbgr.draw()
rgbg.draw()

draw_daylight_cct_locus()
draw_sensor_awb_locus()
