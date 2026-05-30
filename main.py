#!/usr/bin/env python

import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # Библиотека для оптимизаторов в экосистеме JAX

class AdaptiveGaussianLayer(eqx.Module):
    centers: jnp.ndarray
    raw_sigma: jnp.ndarray
    weights: jnp.ndarray
    max_sigma: float = eqx.field(static=True)

    def __init__(self, max_gaussians, x_min, x_max, max_sigma, key):
        self.max_sigma = max_sigma

        # В JAX генерация случайных чисел требует явных ключей (PRNGKey)
        w_key, _ = jax.random.split(key)

        # Инициализируем параметры
        self.centers = jnp.linspace(x_min, x_max, max_gaussians)
        self.raw_sigma = jnp.zeros(max_gaussians)
        self.weights = jax.random.normal(w_key, (max_gaussians,)) * 0.1

    def get_sigmas(self):
        # Ограничиваем сигму сверху через синусоидальную сигмоиду или jax.nn.sigmoid
        return jax.nn.sigmoid(self.raw_sigma) * self.max_sigma

    def __call__(self, x):
        # x имеет форму (N,)
        sigmas = self.get_sigmas()

        # Используем векторизацию JAX (broadcasting) для вычисления матрицы расстояний
        # x[:, None] превращает (N,) в (N, 1), а self.centers имеет форму (M,)
        dist_sq = (x[:, None] - self.centers) ** 2

        # Вычисляем гауссианы
        rbf = jnp.exp(-dist_sq / (2 * (sigmas ** 2) + 1e-8))

        # Линейная комбинация
        return jnp.dot(rbf, self.weights)


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


# 1. Генерируем данные
x_np = jnp.linspace(-5, 5, 400)
y_np = (jnp.exp(-(x_np - 2)**2 / 0.1) + 
        jnp.exp(-(x_np + 2)**2 / 0.2) + 
        0.5 * jnp.exp(-(x_np)**2 / 0.05))

# 2. Инициализируем модель и оптимизатор (Optax)
key = jax.random.PRNGKey(42)
model = AdaptiveGaussianLayer(max_gaussians=30, x_min=-5.0, x_max=5.0, max_sigma=0.5, key=key)

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


# 3. Цикл оптимизации
for epoch in range(1001):
    model, opt_state, loss, (mse, rep) = train_step(model, opt_state, x_np, y_np)

    if epoch % 200 == 0:
        print(f"Эпоха {epoch} | Total Loss: {loss:.4f} | MSE: {mse:.4f} | Штраф за близость: {rep:.4f}")

# Проверяем, сколько гауссиан осталось активными
active_gaussians = jnp.sum(jnp.abs(model.weights) > 0.01)
print(f"\nАктивных гауссиан осталось: {active_gaussians.item()}")
