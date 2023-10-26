import matplotlib.pyplot as plt

from numerical_solution import get_numerical_solution, CoefType
from alpha_minus_2_solution import get_p_alpha_minus_2
from constant_coefs import get_constant_coef_solution
from constants import *
from inverse_laplace_functions import get_inv_laplace


constants_vals = Constants()
constants_vals.val_a_top = -1
constants_vals.val_x_fr = 1
constants_vals.val_k_f = 1
constants_vals.val_k_r = 1
constants_vals.val_a_scl = 1
constants_vals.val_n = 0.5

x_D = 0
s_value = 2

# Тест-сверка аналитической формулы с постоянными коэффициентами и численным решением

print('Тест-сверка аналитической формулы с постоянными коэффициентами и численным решением')

analytic_constant_coef_sol = get_constant_coef_solution(s=s_value, constants_value=constants_vals)
numerical_constant_coef_sol = get_numerical_solution(s_val=s_value, constants_values=constants_vals, coef_type=CoefType.constant)

print(
    f'Аналитическая формула: {"%.2e"%analytic_constant_coef_sol}',
    f'Численная формула: {"%.2e"%numerical_constant_coef_sol}',
    f'Разница: {"%.2e"%abs(analytic_constant_coef_sol - numerical_constant_coef_sol)}',
    sep='\n',
    end='\n\n',
)

# Тест-сверка аналитической формулы с непостоянными коэффициентами и численным решением

print('Тест-сверка аналитической формулы с непостоянными коэффициентами и численным решением')

analytic_nonconstant_coef_sol = get_p_alpha_minus_2(x_D=0, val_s=s_value, vals=constants_vals)
numerical_nonconstant_coef_sol = get_numerical_solution(s_val=s_value, constants_values=constants_vals, coef_type=CoefType.nonconstant)

print(
    f'Аналитическая формула: {"%.2e"%analytic_nonconstant_coef_sol}',
    f'Численная формула: {"%.2e"%numerical_nonconstant_coef_sol}',
    f'Разница: {"%.2e"%abs(analytic_nonconstant_coef_sol - numerical_nonconstant_coef_sol)}',
    sep='\n',
    end='\n\n',
)


# Обратное преобразование Лапласса

print('Обратное преобразование Лапласса')

t_moments = np.exp(np.linspace(np.log(10-4), np.log(10 * 3600), num=2))

p_f_sol = get_inv_laplace(t_moments,lambda q: get_p_alpha_minus_2(x_D=x_D, val_s=q, vals=constants_vals))

plt.plot(t_moments, p_f_sol, '-o')
plt.ylabel('Давление, бар')
plt.xlabel('Время t, сек')
plt.xscale('log')
plt.grid(True)
plt.savefig('res/res1.png', dpi=200)

plt.clf()
plt.cla()

plt.plot([it/3600 for it in t_moments], p_f_sol, '-o')
plt.ylabel('Давление, бар')
plt.xlabel('Время t, ч')
plt.xscale('log')
plt.grid(True)
plt.savefig('res/res2.png', dpi=200)


