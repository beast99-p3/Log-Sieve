"""
FedProx-style local solve for binary logistic regression (NumPy).

Sklearn's ``LogisticRegression`` does not expose a proximal term
``(μ/2)||w - w_global||²``. This module runs gradient descent on the sum of
weighted binary cross-entropy (class-balanced) and the proximal penalty.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def fedprox_binary_logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    coef_global: np.ndarray,
    intercept_global: np.ndarray,
    mu: float,
    *,
    n_steps: int = 800,
    lr: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize (approx.) sum_i w_i * BCE(y_i, σ(Xw+b)) + (μ/2)(||w-w_g||² + (b-b_g)²).

    Parameters
    ----------
    coef_global
        Shape ``(1, n_features)`` or ``(n_features,)``.
    intercept_global
        Shape ``(1,)`` or scalar broadcast.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y_raw = np.asarray(y, dtype=np.float64).ravel()
    classes = np.unique(y_raw)
    if classes.size != 2:
        raise ValueError("fedprox_binary_logistic_fit requires exactly two classes in y.")
    y_lo, y_hi = float(classes.min()), float(classes.max())
    y = (y_raw == y_hi).astype(np.float64)

    cg = np.asarray(coef_global, dtype=np.float64).reshape(-1)
    bg = float(np.asarray(intercept_global, dtype=np.float64).ravel()[0])
    coef = cg.copy() + rng.normal(0, 1e-6, size=cg.shape)
    b = bg + float(rng.normal(0, 1e-6))

    classes_arr = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes_arr, y=y)
    class_to_w = {float(c): float(w) for c, w in zip(classes_arr, cw)}
    sw = np.array([class_to_w[float(yi)] for yi in y], dtype=np.float64)
    sw /= float(np.mean(sw))

    for _ in range(n_steps):
        z = X @ coef + b
        p = _sigmoid(z)
        err = (p - y) * sw
        scale = float(sw.sum())
        grad_c = (X.T @ err) / scale + mu * (coef - cg)
        grad_b = float(np.sum(err)) / scale + mu * (b - bg)
        coef -= lr * grad_c
        b -= lr * grad_b

    return coef.reshape(1, -1), np.array([b], dtype=np.float64)
