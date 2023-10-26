from sympy import *
from sympy.solvers.solveset import linsolve
from typing import TypedDict
from dataclasses import dataclass
import numpy as np
from mpmath import *
import math as m
import matplotlib.pyplot as plt
from tqdm import tqdm

mp.dps = 15
mp.pretty = True

C1_, C2_, Xfr_, Atop_, ksi_, b_, s_ = symbols('C1, C2, Xfr, Atop, ksi, b, s')


#def eq_alpha_minus_2():
  #  res = linsolve(
   #     [Xfr_ * (C1_ * (0.5 + ksi_) * (-Atop_) ** (ksi_ - 0.5) + C2_ * (0.5 - ksi_) * (-Atop_) ** (
    #            -ksi_ - 0.5)) - b_ / s_,
    #     Xfr_ * (C1_ * (0.5 + ksi_) * (Xfr_ - Atop_) ** (ksi_ - 0.5) + C2_ * (0.5 - ksi_) * (Xfr_ - Atop_) ** (
   #              -ksi_ - 0.5))],
    #    (C1_, C2_)
   # )
   # return simplify(res.args[0][0]), simplify(res.args[0][1])


@dataclass
class Constants:
    val_a_top: float = 1
    val_x_fr: float = 1
    val_k_f: float = 1
    val_k_r: float = 1
    val_a_scl: float = 1
    val_n: float = 1


def calc_w_f(x_D: float, vals: Constants):
    tmp1 = vals.val_x_fr * x_D - vals.val_a_top
    return 2 * tmp1 ** (1 / vals.val_n)


def calc_F_D(x_D: float, vals: Constants):
    tmp1 = vals.val_k_f * calc_w_f(x_D, vals)
    tmp2 = vals.val_k_r * vals.val_x_fr
    return tmp1 / tmp2


def calc_b(x_D: float, vals: Constants):
    return - np.pi / calc_F_D(x_D, vals)


def calc_A(s: float, vals: Constants):
    tmp1 = vals.val_a_scl ** (1 / vals.val_n)
    tmp2 = vals.val_k_r * vals.val_x_fr / vals.val_k_f
    return tmp1 * tmp2 * np.sqrt(abs(s))


def calc_C(s: float, vals: Constants):
    return calc_A(s, vals) / (vals.val_x_fr ** 2)


def calc_ksi(C: float):
    return np.sqrt(abs(4 * C + 1)) / 2


def calc_z(x_D: float, vals: Constants):
    return vals.val_x_fr * x_D - vals.val_a_top


#def calc_p_alpha_minus_2(x_D: float, val_s: float, vals: Constants):
  #  val_b = calc_b(0, constants)
  #  val_C = calc_C(val_s, constants)
  #  #print(val_C)#
  #  val_ksi = calc_ksi(val_C)
   # calc_C1, calc_C2 = eq_alpha_minus_2()

   # val_C1 = calc_C1.subs(
    #    [(Xfr_, constants.val_x_fr), (Atop_, constants.val_a_top), (ksi_, val_ksi), (b_, val_b), (s_, val_s)])

   # val_C2 = calc_C2.subs(
   #     [(Xfr_, constants.val_x_fr), (Atop_, constants.val_a_top), (ksi_, val_ksi), (b_, val_b), (s_, val_s)])

    # tmp = - (vals.val_x_fr - vals.val_a_top) ** (-2 * val_ksi) * (-vals.val_a_top) ** (val_ksi - 0.5) + (
    #     -vals.val_a_top) ** (-val_ksi - 0.5)
    #our_C2 = val_b / (val_s * (0.5 - val_ksi) ** 2 * vals.val_x_fr * tmp)

    #our_C1 = - our_C2 * (0.5 - val_ksi) * (vals.val_x_fr - vals.val_a_top) ** (-2 * val_ksi) / (0.5 + val_ksi)
    #
    # print(our_C1, val_C1)
    # print(our_C2, val_C2)

    val_z = calc_z(x_D, vals)

    # print(val_z ** (-1 / vals.val_n) * calc_A(val_s, vals))
    # print('our', our_C1 * val_z ** (0.5 + val_ksi) + our_C2 * val_z ** (0.5 - val_ksi))
    # print('sympy', val_C1 * val_z ** (0.5 + val_ksi) + val_C2 * val_z ** (0.5 - val_ksi))
 #   return val_C1 * val_z ** (0.5 + val_ksi) + val_C2 * val_z ** (0.5 - val_ksi)


@dataclass
class Constants:
    val_a_top: float = 1
    val_x_fr: float = 1
    val_k_f: float = 1
    val_k_r: float = 1
    val_a_scl: float = 1
    val_n: float = 1


constants = Constants()
constants.val_a_top = -1
constants.val_x_fr = 1
constants.val_k_f = 1
constants.val_k_r = 1
constants.val_a_scl = 1
constants.val_n = 0.5

x_D = 0
#_val_s = 2

# our = calc_p_alpha_minus_2(x_D=x_D, val_s=_val_s, vals=constants)
#
# F_D_hard_code = calc_F_D(x_D, constants)
#
# a_hard_code = 2 / F_D_hard_code
# b_hard_code = - np.pi / F_D_hard_code
#
# psi = a_hard_code ** (0.5) * _val_s ** (0.25)
#
# #print(psi ** 2)
#
# their = - b_hard_code * (np.exp(psi * (x_D - 2)) + np.exp(- psi * x_D)) / (psi * _val_s * (1 - np.exp(- 2 * psi)))
#
# print(their,  - b_hard_code / (psi * _val_s * np.tanh(psi)))

#print(our, their, our - their, (our - their) * 100 / their)

#t_moments = np.logspace(3, 6.5, num=5)
#p_f_d_x_D_t = []

#only_s = lambda q: calc_p_alpha_minus_2(x_D=x_D, val_s=q, vals=constants)

#for t_moment in tqdm(t_moments):
#    p_f_d_x_D_t.append(invertlaplace(only_s, t_moment, method='talbot'))

#plt.plot(t_moments, p_f_d_x_D_t, '-o')
#plt.yscale('log')
#plt.show()







int q=3
int a1=1
for k in range(1,3):
    a_i=a1*((2*k-1)*q-1)/(k*q-1)
print(a_i)
def calc_Z(x_D: float, vals: Constants):
    return (q**(-1))*(calc_z()**q)*(calc_C()**(0.5))

U_1=0
def calc_U_1(x_D: float, vals: Constants):
    for i in range(n):
    U_1=U_1+(a_i*(-1)**(i)*(calc_Z())**i)/factorial(i)
    return U_1
dU_1_dZ=0
def calc_dU_1_dZ(x_D: float, vals: Constants):
    for i in range(n):
    dU_1_dZ=dU_1_dZ+(a_i*(-1)**(i)*(calc_Z())**(i-1))/factorial(i-1)
    return dU_1_dZ
U_2=0
def calc_U_2(x_D: float, vals: Constants):
    for i in range(n):
    U_2=U_2+(a_i*(calc_Z())**i)/factorial(i)
    return U_2
dU_2_dZ=0
def calc_dU_2_dZ(x_D: float, vals: Constants):
    for i in range(n):
    dU_2_dZ=dU_2_dZ+(a_i*(calc_Z())**(i-1))/factorial(i-1)
    return dU_2_dZ

def eq_alpha_ne_minus_2():
 res = linsolve(
       [(C1_*m.exp(calc_Z(0))*(calc_U_1(0)+calc_dU_1_dZ(0))+C2_*m.exp(-calc_Z(0))*(-calc_U_2(0)+calc_dU_2_dZ(0)))*((calc_C()**(0.5))*(-Atop_**(q-1))*Xfr_)-b_/s_,
       (C1_*m.exp(calc_Z(1))*(calc_U_1(1)+calc_dU_1_dZ(1))+C2_*m.exp(-calc_Z(1))*(-calc_U_2(1)+calc_dU_2_dZ(1)))*((calc_C()**(0.5))*((Xfr_-Atop_)**(q-1))*Xfr_)],
        (C1_, C2_)
    )
    return simplify(res.args[0][0]), simplify(res.args[0][1])
p=C1_*calc_U_1()*exp(calc_Z())+C2_*calc_U_2()*exp(-calc_Z())