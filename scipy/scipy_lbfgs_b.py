#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
import plot


# 0. Генерируем данные
x_np = np.linspace(-5, 5, 400)
y_np = (np.exp(-(x_np - 2)**2 / 0.1) +
        np.exp(-(x_np + 2)**2 / 0.2) +
        0.5 * np.exp(-(x_np)**2 / 0.05))


def parse_params(params, num_gaussians):
    #params — это плоский вектор [centers..., sigmas..., weights...]
    c = params[0 : num_gaussians]
    s = params[num_gaussians : 2*num_gaussians]
    w = params[2*num_gaussians : 3*num_gaussians]
    return c, s, w


def model(c, s, w, x):
    dist_sq = (x[:, None] - c) ** 2
    rbf = np.exp(-dist_sq / (2 * s ** 2 + 1e-8))
    return np.dot(rbf, w)


# 1. Определяем кастомный лосс на чистом NumPy
def loss_function(params, x, y_true, num_gaussians, min_distance=0.4, lambda_l1=0.005):
    c, s, w = parse_params(params, num_gaussians)
    y_pred = model(c, s, w, x)

    # MSE
    mse = np.mean((y_pred - y_true) ** 2)

    # L1 (поскольку мы на CPU и можем использовать L-BFGS-B из SciPy, 
    # здесь тоже желательно использовать сглаженный L1, либо обычный)
    l1 = lambda_l1 * np.sum(np.sqrt(w**2 + 1e-5))

    # Repulsion loss (расталкивание)
    dist_matrix = np.abs(c[:, None] - c[None, :])
    mask = np.triu(np.ones_like(dist_matrix), k=1)
    too_close = np.clip(min_distance - dist_matrix * mask, a_min=0, a_max=None)
    too_close[np.tril_indices(num_gaussians)] = 0 # убираем диагональ и низ
    repulsion = 5.0 * np.sum(too_close ** 2)

    return mse + l1 + repulsion


# 2. Задаем жесткие границы (Bounds) для параметров
# Для центров: от -5 до 5. Для сигм: от 0.01 до 0.5. Для весов: от 0 до 2 (неотрицательные!)
num_g = 30
bounds = (
    [(-5.0, 5.0)] * num_g +    # Границы для центров
    [(0.01, 0.5)] * num_g +    # Границы для сигм
    [(0.0, 2.0)] * num_g       # Границы для весов (неотрицательность!)
)

# 3. Начальное приближение
init_params = np.concatenate([
    np.linspace(-5, 5, num_g), # центры
    np.ones(num_g) * 0.2,       # сигмы
    np.ones(num_g) * 0.1        # веса
])

# 4. Запуск оптимизации (SciPy сам подберет L-BFGS-B, так как есть Bounds)
res = minimize(
    loss_function, init_params, args=(x_np, y_np, num_g), 
    bounds=bounds, method='L-BFGS-B'
)

# Получаем результат
final_centers, final_sigmas, final_weights = parse_params(res.x, num_g)
active_gaussians = np.sum(final_weights > 0.01)

print(f"Активных гауссиан осталось: {active_gaussians}")
print("Финальные веса:", final_weights)
print("Финальные центры:", final_centers)
print(f"Оптимизация завершена. Финальный лосс: {res.fun:.5f}")

plot.show_plot(x_np, y_np, model(final_centers, final_sigmas, final_weights, x_np))
