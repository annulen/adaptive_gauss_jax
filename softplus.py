#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def softplus(x):
    return np.log(1 + np.exp(x))

x = np.linspace(-3, 3, 200)
y0 = np.clip(x, a_min=0, a_max=None)

# softplus is a smooth approximation of np.clip(x, min=0)
y1 = softplus(x)

# scaling x makes approximation more precise
y2 = softplus(x * 10) / 10

fig, ax = plt.subplots()
ax.grid(True, which="major", axis="both")
ax.plot(x, y0, label="clip(x, a_min=0)")
ax.plot(x, y1, label="softplus(x)")
ax.plot(x, y2, label="softplus(x*10)/10")
plt.legend()
plt.show()

