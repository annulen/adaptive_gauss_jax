#!/usr/bin/env python

import jax
import jax.numpy as jnp
import equinox as eqx
from model import model, x_np, y_np
import optax  # Библиотека для оптимизаторов в экосистеме JAX
import plot


def loss_fn(model, x, y_true, min_distance=0.4, lambda_l1=0.005, lambda_repulsion=5.0):
    y_pred = model(x)

    # 1. Основная ошибка (MSE)
    mse_loss = jnp.mean((y_pred - y_true) ** 2)

    # 2. L1-штраф на веса для авто-подбора количества (прунинг)
    l1_loss = jnp.sum(jnp.abs(model.weights))

    # 3. Штраф за скученность центров
    centers = model.centers
    # Матрица попарных расстояний
    dist_matrix = jnp.abs(centers[:, None] - centers[None, :])

    # В JAX вместо масок лучше использовать зануление нижней диагонали через jnp.triu
    mask = jnp.triu(jnp.ones_like(dist_matrix), k=1)
    pair_distances = dist_matrix * mask

    # Если расстояние > 0 и меньше порога, считаем штраф
    too_close = min_distance - pair_distances
    # Убираем пары, которые изначально равны 0 (расстояние элемента с самим собой)
    too_close = jnp.where(mask == 1, too_close, 0.0)

    repulsion_loss = jnp.sum(jnp.clip(too_close, a_min=0.0) ** 2)

    total_loss = mse_loss + lambda_l1 * l1_loss + lambda_repulsion * repulsion_loss
    return total_loss, (mse_loss, repulsion_loss)


# Используем Adam из библиотеки optax
optimizer = optax.adam(learning_rate=0.02)
opt_state = optimizer.init(model)
opt_update = optimizer.update


# Функция одного шага обучения
@jax.jit
def train_step(model, opt_state, x, y):
    # jax.value_and_grad вычисляет и значение лосса, и градиенты по отношению к модели
    # has_aux=True говорит, что функция возвращает дополнительные метрики (mse, rep)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, x, y)

    # Обновляем состояние оптимизатора и параметры модели
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux


# Цикл оптимизации
for epoch in range(1001):
    model, opt_state, loss, (mse, rep) = train_step(model, opt_state, x_np, y_np)

    if epoch % 200 == 0:
        print(f"Эпоха {epoch} | Total Loss: {loss:.4f} | MSE: {mse:.4f} | Штраф за близость: {rep:.4f}")

# Проверяем, сколько гауссиан осталось активными
active_gaussians = jnp.sum(jnp.abs(model.weights) > 0.01)
print(f"\nАктивных гауссиан осталось: {active_gaussians.item()}")
print("Финальные веса:", model.weights)

plot.show_plot(x_np, y_np, model(x_np))
