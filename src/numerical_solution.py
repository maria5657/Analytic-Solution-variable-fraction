from scipy import *
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from constants import *


class CoefType(Enum):
    constant: int = 0
    nonconstant: int = 1


def f_real(t, y, s_val: float, vals: Constants, coef_type: CoefType):
    if coef_type is CoefType.constant:
        z = vals.val_x_fr * 0 - vals.val_a_top
    else:
        z = vals.val_x_fr * t - vals.val_a_top
    coef = calc_C(s=s_val, vals=vals) * z ** (-2)
    return [
        y[1],
        coef * y[0]
    ]


def boundary_residual_real(ya, yb, s: float, vals: Constants):
    return np.array([
        ya[1] - calc_b(x_D=0, vals=vals) * vals.val_x_fr / s,
        yb[1] - 0
    ])


def get_numerical_solution(s_val: float, constants_values: Constants, coef_type: CoefType):
    f = lambda t, y: f_real(t, y, s_val, constants_values, coef_type)
    boundary_residual = lambda ya, yb: boundary_residual_real(ya, yb, s_val, constants_values)

    a, b = 0, 1
    N = 1000
    x = np.linspace(a, b, N)
    y_guess = np.zeros((2, N), dtype=float)

    sol = integrate.solve_bvp(f, boundary_residual, x, y_guess)

    return sol.y[0][0]
