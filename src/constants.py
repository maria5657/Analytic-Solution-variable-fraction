from scipy import *
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Constants:
    val_a_top: float = -1
    val_x_fr: float = 1
    val_k_f: float = 1
    val_k_r: float = 1
    val_a_scl: float = 1
    val_n: float = 0.5


def calc_A(s: float, vals: Constants = Constants()):
    tmp1 = vals.val_a_scl ** (1 / vals.val_n)
    tmp2 = vals.val_k_r * vals.val_x_fr / vals.val_k_f
    return tmp1 * tmp2 * np.sqrt(abs(s))


def calc_C(s: float, vals: Constants = Constants()):
    return calc_A(s, vals) / (vals.val_x_fr ** 2)


def calc_w_f(x_D: float, vals: Constants):
    tmp1 = vals.val_x_fr * x_D - vals.val_a_top
    return 2 * tmp1 ** (1 / vals.val_n)


def calc_F_D(x_D: float, vals: Constants):
    tmp1 = vals.val_k_f * calc_w_f(x_D, vals)
    tmp2 = vals.val_k_r * vals.val_x_fr
    return tmp1 / tmp2


def calc_b(x_D: float, vals: Constants):
    return - np.pi / calc_F_D(x_D, vals)


def calc_ksi(C: float):
    return np.sqrt(abs(4 * C + 1)) / 2


def calc_z(x_D: float, vals: Constants):
    return vals.val_x_fr * x_D - vals.val_a_top
