"""Development-only state, geometry, and noise ridge baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


class StateFeatureRidge:
    """Ridge linear-probability model using no model-derived score.

    Raw physical features use fixed scales: 1.5 for positions and distances,
    1.5 for velocities, and 0.02 for noise. Wall distance is the Euclidean
    distance to the nearest point on a solid wall interval. Gap signed distance
    is ``y - gap_center``. Directed speeds are velocity dot products with unit
    vectors toward the nearest solid-wall point and the gap center. Interactions
    multiply the corresponding normalized distance and directed-speed features.
    These fixed scalings precede development-data standardization in ``fit``.
    """

    POSITION_SCALE = 1.5
    VELOCITY_SCALE = 1.5
    NOISE_SCALE = 0.02
    REGIMES = ("R0", "R1", "R2")
    FEATURE_NAMES = (
        "x",
        "y",
        "vx",
        "vy",
        "wall_distance",
        "gap_signed_distance",
        "gap_width",
        "goal_distance",
        "speed",
        "wall_directed_speed",
        "gap_directed_speed",
        "noise_scale",
        "regime_R0",
        "regime_R1",
        "regime_R2",
        "wall_distance_x_wall_speed",
        "gap_distance_x_gap_speed",
    )

    @staticmethod
    def _field(checkpoint: Any, name: str, default: Any = None) -> Any:
        if isinstance(checkpoint, Mapping):
            return checkpoint.get(name, default)
        return getattr(checkpoint, name, default)

    @classmethod
    def _state(cls, checkpoint: Any) -> tuple[float, float, float, float]:
        state = cls._field(checkpoint, "state")
        if state is None:
            values = [cls._field(checkpoint, name) for name in ("x", "y", "vx", "vy")]
        elif isinstance(state, Mapping) or hasattr(state, "x"):
            values = [cls._field(state, name) for name in ("x", "y", "vx", "vy")]
        else:
            values = list(state)
        if len(values) != 4 or any(value is None for value in values):
            raise ValueError("checkpoint state must provide x, y, vx, and vy")
        return tuple(float(value) for value in values)

    @classmethod
    def _point(cls, value: Any, name: str) -> tuple[float, float]:
        if isinstance(value, Mapping) or hasattr(value, "x"):
            point = (cls._field(value, "x"), cls._field(value, "y"))
        elif value is not None:
            point = tuple(value)
        else:
            point = ()
        if len(point) != 2 or any(coordinate is None for coordinate in point):
            raise ValueError(f"{name} must contain two coordinates")
        return float(point[0]), float(point[1])

    @classmethod
    def _gap_value(cls, checkpoint: Any, name: str) -> Any:
        direct = cls._field(checkpoint, f"gap_{name}")
        if direct is not None:
            return direct
        gap = cls._field(checkpoint, "gap")
        if gap is None:
            raise ValueError(f"checkpoint must provide gap_{name}")
        value = cls._field(gap, name)
        if value is None:
            raise ValueError(f"checkpoint gap must provide {name}")
        return value

    @staticmethod
    def _regime_index(regime: Any) -> int:
        if isinstance(regime, (int, np.integer)) and not isinstance(regime, bool):
            index = int(regime)
        else:
            text = str(regime).upper()
            if text not in {"R0", "R1", "R2"}:
                raise ValueError("regime must be R0, R1, R2, or integer 0, 1, 2")
            index = int(text[1])
        if index not in {0, 1, 2}:
            raise ValueError("regime must be R0, R1, R2, or integer 0, 1, 2")
        return index

    @classmethod
    def feature_vector(cls, checkpoint: Any) -> np.ndarray:
        """Extract the fixed 17-feature vector from a mapping or state object."""
        x, y, vx, vy = cls._state(checkpoint)
        gap_center = float(cls._gap_value(checkpoint, "center"))
        gap_width = float(cls._gap_value(checkpoint, "width"))
        if gap_width < 0.0:
            raise ValueError("gap_width must be nonnegative")
        goal = cls._point(cls._field(checkpoint, "goal", (1.5, 0.0)), "goal")
        wall_x = float(cls._field(checkpoint, "wall_x", 0.0))
        wall_y_min = float(cls._field(checkpoint, "wall_y_min", -1.5))
        wall_y_max = float(cls._field(checkpoint, "wall_y_max", 1.5))
        if wall_y_min >= wall_y_max:
            raise ValueError("wall_y_min must be less than wall_y_max")
        noise_scale = float(cls._field(checkpoint, "noise_scale", 0.02))
        regime = cls._field(checkpoint, "regime")
        if regime is None:
            raise ValueError("checkpoint must provide a regime")
        regime_index = cls._regime_index(regime)

        gap_bottom = gap_center - gap_width / 2.0
        gap_top = gap_center + gap_width / 2.0
        lower_end = min(max(gap_bottom, wall_y_min), wall_y_max)
        upper_start = min(max(gap_top, wall_y_min), wall_y_max)
        nearest_y_values: list[float] = []
        if wall_y_min <= lower_end:
            nearest_y_values.append(float(np.clip(y, wall_y_min, lower_end)))
        if upper_start <= wall_y_max:
            nearest_y_values.append(float(np.clip(y, upper_start, wall_y_max)))
        if not nearest_y_values:
            raise ValueError("the wall must contain at least one solid interval")
        nearest_wall_y = min(nearest_y_values, key=lambda value: abs(y - value))

        wall_vector = np.array([wall_x - x, nearest_wall_y - y], dtype=float)
        gap_vector = np.array([wall_x - x, gap_center - y], dtype=float)
        velocity = np.array([vx, vy], dtype=float)
        wall_distance = float(np.linalg.norm(wall_vector))
        gap_euclidean_distance = float(np.linalg.norm(gap_vector))
        wall_speed = (
            float(np.dot(velocity, wall_vector) / wall_distance)
            if wall_distance > 0.0
            else 0.0
        )
        gap_speed = (
            float(np.dot(velocity, gap_vector) / gap_euclidean_distance)
            if gap_euclidean_distance > 0.0
            else 0.0
        )
        normalized = np.array(
            [
                x / cls.POSITION_SCALE,
                y / cls.POSITION_SCALE,
                vx / cls.VELOCITY_SCALE,
                vy / cls.VELOCITY_SCALE,
                wall_distance / cls.POSITION_SCALE,
                (y - gap_center) / cls.POSITION_SCALE,
                gap_width / cls.POSITION_SCALE,
                float(np.hypot(x - goal[0], y - goal[1])) / cls.POSITION_SCALE,
                float(np.hypot(vx, vy)) / cls.VELOCITY_SCALE,
                wall_speed / cls.VELOCITY_SCALE,
                gap_speed / cls.VELOCITY_SCALE,
                noise_scale / cls.NOISE_SCALE,
            ],
            dtype=float,
        )
        one_hot = np.zeros(3, dtype=float)
        one_hot[regime_index] = 1.0
        interactions = np.array(
            [normalized[4] * normalized[9], normalized[5] * normalized[10]],
            dtype=float,
        )
        features = np.concatenate((normalized, one_hot, interactions))
        if not np.all(np.isfinite(features)):
            raise ValueError("checkpoint produced non-finite baseline features")
        return features

    @staticmethod
    def _fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return mean, scale

    @staticmethod
    def _ridge_fit(
        features: np.ndarray, outcomes: np.ndarray, ridge: float
    ) -> tuple[np.ndarray, float]:
        feature_center = np.mean(features, axis=0)
        outcome_center = float(np.mean(outcomes))
        centered = features - feature_center
        gram = centered.T @ centered
        gram = gram + ridge * np.eye(gram.shape[0])
        coefficients = np.linalg.solve(gram, centered.T @ (outcomes - outcome_center))
        intercept = float(outcome_center - feature_center @ coefficients)
        return coefficients, intercept

    @classmethod
    def _check_development(cls, checkpoint: Any) -> None:
        for field in ("split", "partition"):
            value = cls._field(checkpoint, field)
            if value is not None and str(value).lower() in {
                "test",
                "confirm",
                "confirmation",
                "confirmatory",
            }:
                raise ValueError("StateFeatureRidge.fit accepts development data only")
        is_development = cls._field(checkpoint, "is_development")
        if is_development is False:
            raise ValueError("StateFeatureRidge.fit accepts development data only")

    def fit(
        self,
        checkpoints: Iterable[Any],
        outcomes: Any,
        seed_groups: Any,
        lambdas: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    ) -> "StateFeatureRidge":
        """Select ridge strength by leave-one-development-seed-out CV and refit."""
        checkpoint_list = list(checkpoints)
        if not checkpoint_list:
            raise ValueError("checkpoints must be non-empty")
        for checkpoint in checkpoint_list:
            self._check_development(checkpoint)
        features = np.vstack([self.feature_vector(item) for item in checkpoint_list])
        targets = np.asarray(outcomes, dtype=float)
        groups = np.asarray(seed_groups)
        if targets.ndim != 1 or groups.ndim != 1:
            raise ValueError("outcomes and seed_groups must be one-dimensional")
        if len(checkpoint_list) != targets.size or targets.size != groups.size:
            raise ValueError("checkpoints, outcomes, and seed_groups must align")
        if not np.all(np.isfinite(targets)) or np.any((targets < 0.0) | (targets > 1.0)):
            raise ValueError("outcomes must be finite probabilities in [0, 1]")
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            raise ValueError("grouped CV requires at least two development seeds")
        lambda_values = sorted({float(value) for value in lambdas})
        if not lambda_values or any(
            not np.isfinite(value) or value <= 0.0 for value in lambda_values
        ):
            raise ValueError("lambdas must be positive finite values")

        self.cv_folds_: list[dict[str, Any]] = []
        self.cv_scores_: dict[float, float] = {}
        for ridge in lambda_values:
            fold_losses: list[float] = []
            for held_out in unique_groups:
                validation = groups == held_out
                training = ~validation
                mean, scale = self._fit_scaler(features[training])
                train_features = (features[training] - mean) / scale
                coefficients, intercept = self._ridge_fit(
                    train_features, targets[training], ridge
                )
                validation_features = (features[validation] - mean) / scale
                prediction = validation_features @ coefficients + intercept
                loss = float(np.mean((prediction - targets[validation]) ** 2))
                fold_losses.append(loss)
                self.cv_folds_.append(
                    {
                        "lambda": ridge,
                        "held_out_group": held_out,
                        "train_groups": unique_groups[unique_groups != held_out].copy(),
                        "feature_mean": mean.copy(),
                        "feature_scale": scale.copy(),
                        "n_validation": int(np.count_nonzero(validation)),
                        "mse": loss,
                    }
                )
            self.cv_scores_[ridge] = float(np.mean(fold_losses))

        self.lambda_ = min(lambda_values, key=lambda value: self.cv_scores_[value])
        self.feature_mean_, self.feature_scale_ = self._fit_scaler(features)
        standardized = (features - self.feature_mean_) / self.feature_scale_
        self.coef_, self.intercept_ = self._ridge_fit(
            standardized, targets, self.lambda_
        )
        self.feature_names_ = self.FEATURE_NAMES
        self.n_features_in_ = features.shape[1]
        self.development_seed_groups_ = unique_groups.copy()
        self.raw_development_predictions_ = standardized @ self.coef_ + self.intercept_
        self.development_predictions_ = np.clip(
            self.raw_development_predictions_, 0.0, 1.0
        )
        return self

    def _prediction_features(self, checkpoints: Iterable[Any] | Any) -> np.ndarray:
        if isinstance(checkpoints, Mapping) or hasattr(checkpoints, "state"):
            checkpoint_list = [checkpoints]
        else:
            checkpoint_list = list(checkpoints)
        if not checkpoint_list:
            raise ValueError("checkpoints must be non-empty")
        return np.vstack([self.feature_vector(item) for item in checkpoint_list])

    def predict_raw(self, checkpoints: Iterable[Any] | Any) -> np.ndarray:
        """Return unbounded linear-probability predictions without clipping."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before prediction")
        features = self._prediction_features(checkpoints)
        standardized = (features - self.feature_mean_) / self.feature_scale_
        return standardized @ self.coef_ + self.intercept_

    def predict(self, checkpoints: Iterable[Any] | Any) -> np.ndarray:
        """Return risk scores clipped to [0, 1], retaining ``predict_raw`` separately."""
        return np.clip(self.predict_raw(checkpoints), 0.0, 1.0)


def closed_form_ridge_fixture() -> dict[str, Any]:
    """Return the plan's one-feature ridge arithmetic and clipping check."""
    train_z = np.array([-1.0, 1.0])
    train_y = np.array([0.0, 1.0])
    coefficients, intercept = StateFeatureRidge._ridge_fit(
        train_z.reshape(-1, 1), train_y, 1.0
    )
    z = np.array([1.0, 2.0])
    raw_predictions = intercept + z * coefficients[0]
    return {
        "train_z": train_z,
        "train_y": train_y,
        "lambda": 1.0,
        "intercept": float(intercept),
        "slope": float(coefficients[0]),
        "z": z,
        "raw_predictions": raw_predictions,
        "clipped_predictions": np.clip(raw_predictions, 0.0, 1.0),
    }


__all__ = ["StateFeatureRidge", "closed_form_ridge_fixture"]
