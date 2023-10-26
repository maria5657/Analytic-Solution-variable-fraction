from sympy import *
from sympy.solvers.solveset import linsolve
from typing import TypedDict
from dataclasses import dataclass
import numpy as np
from mpmath import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import *

C1_, C2_, Xfr_, Atop_, ksi_, b_, s_ = symbols('C1, C2, Xfr, Atop, ksi, b, s')

def eq_alpha_minus_2():
    res = linsolve(
        [Xfr_ * (C1_ * (0.5 + ksi_) * (-Atop_) ** (ksi_ - 0.5) + C2_ * (0.5 - ksi_) * (-Atop_) ** (
                -ksi_ - 0.5)) - b_ / s_,
         Xfr_ * (C1_ * (0.5 + ksi_) * (Xfr_ - Atop_) ** (ksi_ - 0.5) + C2_ * (0.5 - ksi_) * (Xfr_ - Atop_) ** (
                 -ksi_ - 0.5))],
        (C1_, C2_)
    )
    return simplify(res.args[0][0]), simplify(res.args[0][1])


def get_p_alpha_minus_2(x_D: float, val_s: float, vals: Constants):
    val_b = calc_b(0, vals)
    val_C = calc_C(val_s, vals)
    val_ksi = calc_ksi(val_C)
    calc_C1, calc_C2 = eq_alpha_minus_2()

    val_C1 = calc_C1.subs(
        [(Xfr_, vals.val_x_fr), (Atop_, vals.val_a_top), (ksi_, val_ksi), (b_, val_b), (s_, val_s)])

    val_C2 = calc_C2.subs(
        [(Xfr_, vals.val_x_fr), (Atop_, vals.val_a_top), (ksi_, val_ksi), (b_, val_b), (s_, val_s)])

    val_z = calc_z(x_D, vals)

    return val_C1 * val_z ** (0.5 + val_ksi) + val_C2 * val_z ** (0.5 - val_ksi)