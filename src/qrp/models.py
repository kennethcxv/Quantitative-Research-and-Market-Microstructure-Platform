"""Baseline forecasters: zero, training mean, and ridge regression on standardised features.

The zero forecast ("no change") is the benchmark for returns; a model that cannot beat it on
out-of-sample squared error has no usable predictive content, whatever its rank IC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

FloatArray = npt.NDArray[np.float64]


class ModelSpec(BaseModel):
    name: str
    kind: Literal["zero", "mean", "ridge"]
    features: list[str] = []
    alpha: float = 0.0  # ridge penalty per sample on standardised features; 0 = OLS


@dataclass(frozen=True)
class FittedModel:
    spec: ModelSpec
    intercept: float
    coef: FloatArray
    center: FloatArray
    scale: FloatArray

    def predict(self, x: FloatArray) -> FloatArray:
        if not self.spec.features:
            return np.full(len(x), self.intercept)
        return self.intercept + ((x - self.center) / self.scale) @ self.coef


def fit(spec: ModelSpec, x: FloatArray, y: FloatArray) -> FittedModel:
    k = len(spec.features)
    if x.shape[1] != k:
        raise ValueError(f"{spec.name}: expected {k} feature columns, got {x.shape[1]}")
    empty = np.zeros(k)
    if spec.kind == "zero":
        return FittedModel(spec, 0.0, empty, empty, np.ones(k))
    y_mean = float(y.mean())
    if spec.kind == "mean":
        return FittedModel(spec, y_mean, empty, empty, np.ones(k))
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0  # a constant feature gets a zero coefficient instead of a NaN
    z = (x - center) / scale
    # Ridge as an augmented least-squares problem; lstsq's minimum-norm solution also handles
    # collinear features (and alpha = 0) without a singular-matrix failure.
    penalty = np.sqrt(spec.alpha * len(y)) * np.eye(k)
    design = np.vstack([z, penalty])
    target = np.concatenate([y - y_mean, np.zeros(k)])
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    return FittedModel(spec, y_mean, coef, center, scale)
