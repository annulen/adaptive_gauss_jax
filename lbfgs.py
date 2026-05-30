#!/usr/bin/env python

import jax
import jax.numpy as jnp
from jaxopt import LBFGS  # Импортируем LBFGS из JAXopt
from model import model, x_np, y_np

# 1. Слегка сглаживаем лосс для корректной работы BFGS
def smooth_loss_fn(model, x, y_true, min_distance=0.4, lambda_l1=0.005, lambda_repulsion=5.0):
    y_pred = model(x)
    mse_loss = jnp.mean((y_pred - y_true) ** 2)

    # Сглаженный L1 (Huber/Pseudo-Huber подход)
    l1_loss = jnp.sum(jnp.sqrt(model.weights**2 + 1e-5))

    # Расстояния между центрами
    centers = model.centers
    dist_matrix = jnp.abs(centers[:, None] - centers[None, :])
    mask = jnp.triu(jnp.ones_like(dist_matrix), k=1)
    pair_distances = dist_matrix * mask

    too_close = min_distance - pair_distances
    too_close = jnp.where(mask == 1, too_close, 0.0)

    # Вместо jnp.clip используем softplus для идеальной гладкости
    repulsion_loss = jnp.sum(jax.nn.softplus(too_close * 10) / 10) # Масштабируем для крутизны

    return mse_loss + lambda_l1 * l1_loss + lambda_repulsion * repulsion_loss

# 2. Обертка для JAXopt
# JAXopt ожидает, что функция принимает плоский вектор параметров, 
# но Equinox работает с PyTrees. Мы можем передавать модель прямо!
def jaxopt_objective(params, x, y_true):
    return smooth_loss_fn(params, x, y_true)

# 3. Настройка и запуск LBFGS
# Здесь не нужен цикл по эпохам! JAXopt сам проведет всю оптимизацию.
lbfgs = LBFGS(fun=jaxopt_objective, maxiter=150, implicit_diff=False)

# Запускаем оптимизацию (передаем начальную модель как стартовую точку)
# С помощью run() JAXopt сам откомпилирует всё через JIT и выполнит итерации
res = lbfgs.run(model, x=x_np, y_true=y_np)

# Проверяем, сколько гауссиан осталось активными
active_gaussians = jnp.sum(jnp.abs(model.weights) > 0.01)
print(f"\nАктивных гауссиан осталось: {active_gaussians.item()}")

# В res.params лежит наша полностью обученная модель Equinox!
best_model = res.params
print(f"Оптимизация завершена. Финальный лосс: {res.state.error:.5f}")
