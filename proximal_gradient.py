#!/usr/bin/env python

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxopt import ProximalGradient
from model import model, x_np, y_np


# 1. Выносим L1 из основной функции потерь. Оставляем только гладкую часть!
def smooth_loss_only(model, x, y_true):
    y_pred = model(x)
    mse_loss = jnp.mean((y_pred - y_true) ** 2)

    # Расстояния между центрами
    centers = model.centers
    dist_matrix = jnp.abs(centers[:, None] - centers[None, :])
    mask = jnp.triu(jnp.ones_like(dist_matrix), k=1)
    pair_distances = dist_matrix * mask
    too_close = 0.4 - pair_distances
    too_close = jnp.where(mask == 1, too_close, 0.0)
    repulsion_loss = jnp.sum(jax.nn.softplus(too_close * 10) / 10)

    return mse_loss + 5.0 * repulsion_loss


# 2. Определяем проксимальный оператор для L1-регуляризации.
# Он будет применять штраф ТОЛЬКО к весам (weights), оставляя центры и сигмы в покое.
def l1_prox(model, hyperparams_lambda, scaling=1.0):
    # hyperparams_lambda — это сила L1 штрафа (наш lambda_l1)
    step = hyperparams_lambda * scaling

    # Мягкое пороговое зануление для весов:
    new_weights = jnp.sign(model.weights) * jnp.clip(jnp.abs(model.weights) - step, a_min=0.0)

    # Возвращаем обновленное PyTree модели
    return eqx.tree_at(lambda m: m.weights, model, new_weights)


# 3. Настраиваем проксимальный градиентный спуск (вместо LBFGS)
pg = ProximalGradient(fun=smooth_loss_only, prox=l1_prox, maxiter=500)

# Запускаем. Второй аргумент — это hyperparams_lambda, который придет в l1_prox
res = pg.run(model, 0.005, x=x_np, y_true=y_np)


# Проверяем, сколько гауссиан осталось активными
active_gaussians = jnp.sum(jnp.abs(model.weights) > 0.01)
print(f"\nАктивных гауссиан осталось: {active_gaussians.item()}")

# В res.params лежит наша полностью обученная модель Equinox!
best_model = res.params
print(f"Оптимизация завершена. Финальный лосс: {res.state.error:.5f}")
