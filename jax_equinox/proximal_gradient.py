#!/usr/bin/env python

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxopt import ProximalGradient, LBFGS
from model import model, x_np, y_np
import plot


# 1. Теперь функция потерь принимает НЕ модель, а кортеж её изменяемых параметров (PyTree)
# Это нужно, чтобы jaxopt четко понимал, по чем мы берем градиент
def smooth_loss_only(params, static, x, y_true):
    # Собираем модель обратно из параметров и статической структуры Equinox
    model = eqx.combine(params, static)

    y_pred = model(x)
    mse_loss = jnp.mean((y_pred - y_true) ** 2)

    # Штраф за скученность центров
    centers = model.centers
    # Матрица попарных расстояний
    dist_matrix = jnp.abs(centers[:, None] - centers[None, :])
    mask = jnp.triu(jnp.ones_like(dist_matrix), k=1)
    pair_distances = dist_matrix * mask

    min_distance = 0.4
    # Если расстояние > 0 и меньше порога, считаем штраф
    too_close = min_distance - pair_distances
    # Убираем пары, которые изначально равны 0 (расстояние элемента с самим собой)
    too_close = jnp.where(mask == 1, too_close, 0.0)

    repulsion_loss = jnp.sum(jnp.clip(too_close, a_min=0.0) ** 2)

    return mse_loss + 5.0 * repulsion_loss


# 2. Проксимальный оператор теперь работает с чистым PyTree параметров.
# Мы применяем soft-thresholding ТОЛЬКО к полю weights.
def l1_prox(params, hyperparams_lambda, scaling=1.0):
    step = hyperparams_lambda * scaling

    # Извлекаем текущие веса из структуры параметров
    weights = params.weights

    # Применяем жесткое зануление (Мягкий порог для положительных, зануление для отрицательных)
    new_weights = jnp.sign(weights) * jnp.clip(weights - step, a_min=0.0)

    # Возвращаем обновленную структуру параметров через equinox
    return eqx.tree_at(lambda p: p.weights, params, new_weights)


# 3. Разделяем модель на изменяемые параметры (params) и статическую структуру (static)
# Это стандартный и самый мощный паттерн в Equinox
params, static = eqx.partition(model, eqx.is_array)

# 4. Настраиваем ProximalGradient
# Передаем аргумент argument_nums=(0,), чтобы jaxopt знал, что оптимизируются только params
pg = ProximalGradient(fun=smooth_loss_only, prox=l1_prox, maxiter=800)

# 5. Запускаем оптимизацию. 
# Сила L1 штрафа = 0.002. Передаем static как дополнительный аргумент после params!
res = pg.run(params, 0.002, static=static, x=x_np, y_true=y_np)

# 6. Делаем debiasing - "дожимаем" результат до настоящего минимума
lbfgs = LBFGS(fun=smooth_loss_only, maxiter=150, implicit_diff=False)
res = lbfgs.run(res.params, static=static, x=x_np, y_true=y_np)

# 7. Собираем финальную модель обратно и проверяем результат
best_model = eqx.combine(res.params, static)
active_gaussians = jnp.sum(jnp.abs(best_model.weights) > 0.01)
print(f"Активных гауссиан осталось: {active_gaussians.item()}")

# Выведем сами веса, чтобы убедиться, что там теперь честные, абсолютные нули
print("Финальные веса:", best_model.weights)

# В res.params лежит наша полностью обученная модель Equinox!
best_model = res.params
print(f"Оптимизация завершена. Финальный лосс: {res.state.error:.5f}")

plot.show_plot(x_np, y_np, best_model(x_np))
