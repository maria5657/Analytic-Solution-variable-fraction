from constants import *
import numpy as np

x_D = 0


def get_constant_coef_solution(s: float, constants_value: Constants):
    F_D_hard_code = calc_F_D(x_D, constants_value)

    a_hard_code = 2 / F_D_hard_code
    b_hard_code = - np.pi / F_D_hard_code

    psi = a_hard_code ** (0.5) * s ** (0.25)

    their = - b_hard_code * (np.exp(psi * (x_D - 2)) + np.exp(- psi * x_D)) / (psi * s * (1 - np.exp(- 2 * psi)))

    their2 = - b_hard_code / (psi * s * np.tanh(psi))

    return their2
