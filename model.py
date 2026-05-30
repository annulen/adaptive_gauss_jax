#!/usr/bin/env python

import jax
import jax.numpy as jnp
import equinox as eqx


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


# 1. Генерируем данные
x_np = jnp.linspace(-5, 5, 400)
y_np = (jnp.exp(-(x_np - 2)**2 / 0.1) +
        jnp.exp(-(x_np + 2)**2 / 0.2) +
        0.5 * jnp.exp(-(x_np)**2 / 0.05))

# 2. Инициализируем модель и оптимизатор (Optax)
key = jax.random.PRNGKey(42)
model = AdaptiveGaussianLayer(max_gaussians=30, x_min=-5.0, x_max=5.0, max_sigma=0.5, key=key)
