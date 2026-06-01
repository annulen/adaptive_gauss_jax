#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize, differential_evolution
import plot

# 0. Генерируем данные
x_np = np.linspace(-5, 5, 400)
y_np = (np.exp(-(x_np - 2)**2 / 0.1) +
        np.exp(-(x_np + 2)**2 / 0.2) +
        0.5 * np.exp(-(x_np)**2 / 0.05))


def hybrid_loss(centers, x, y_true, num_gaussians):
    """
    Внешняя функция потерь для Differential Evolution.
    Принимает ТОЛЬКО центры (размерность = num_gaussians).
    """
    # Внутренняя задача: оптимизируем веса и сигмы под ЭТИ конкретные центры
    # Параметры внутренней оптимизации: [sigmas..., weights...] -> размерность 2 * num_gaussians

    def internal_loss(internal_params):
        sigmas = internal_params[0 : num_gaussians]
        weights = internal_params[num_gaussians : 2 * num_gaussians]

        # Считаем модель
        dist_sq = (x[:, None] - centers) ** 2
        rbf = np.exp(-dist_sq / (2 * sigmas ** 2 + 1e-8))
        y_pred = np.dot(rbf, weights)

        # Локальный лосс: MSE + сглаженный L1 для весов
        mse = np.mean((y_pred - y_true) ** 2)
        l1 = 0.005 * np.sum(np.sqrt(weights**2 + 1e-5))

        # Примечание: repulsion_loss для центров здесь считать НЕ НУЖНО! 
        # Эволюция сама разберется, как их расставить, чтобы минимизировать MSE.
        return mse + l1

    # Начальное приближение для внутренних параметров на этой итерации
    init_internal = np.concatenate([
        np.ones(num_gaussians) * 0.2,  # Стартовые сигмы
        np.ones(num_gaussians) * 0.1   # Стартовые веса
    ])

    # Границы для внутренних параметров
    internal_bounds = (
        [(0.01, 0.5)] * num_gaussians +  # Сигмы
        [(0.0, 2.0)] * num_gaussians     # Веса (неотрицательные)
    )

    # Быстрый локальный спуск (ограничим maxiter=15..20, чтобы не тормозить эволюцию)
    # За несколько шагов L-BFGS-B идеально адаптирует веса под текущие центры
    print("Оптимизация с центрами:", centers)
    res = minimize(
        internal_loss, init_internal, 
        bounds=internal_bounds, method='L-BFGS-B', 
        options={'maxiter': 20}
    )
    print(f"Текущий лосс: {res.fun:.5f}")

    # Возвращаем финальное значение ошибки для эволюции
    return res.fun


num_g = 30

# Границы только для центров (размерность всего 30!)
centers_bounds = [(-5.0, 5.0)] * num_g

# Запуск эволюции
ga_result = differential_evolution(
    hybrid_loss, 
    bounds=centers_bounds, 
    args=(x_np, y_np, num_g),
    popsize=10,        # Можно уменьшить, так как размерность упала
    maxiter=50,
    polish=True        # В конце сделает финальный лоск для центров
)

# Лучшие найденные центры
final_centers = ga_result.x

print("Финальные центры:", final_centers)
