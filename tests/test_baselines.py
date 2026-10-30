from types import SimpleNamespace

import numpy as np
import pytest

from counterworld.baselines import StateFeatureRidge, closed_form_ridge_fixture


def checkpoint(**changes):
    value = {
        "state": (-1.0, 0.5, 1.0, -0.5),
        "gap_center": 0.0,
        "gap_width": 0.6,
        "goal": (1.5, 0.0),
        "wall_x": 0.0,
        "wall_y_min": -1.5,
        "wall_y_max": 1.5,
        "noise_scale": 0.04,
        "regime": "R2",
        "split": "development",
    }
    value.update(changes)
    return value


def test_feature_vector_normalization_geometry_regime_and_interactions():
    features = StateFeatureRidge.feature_vector(checkpoint())
    assert features.shape == (17,)
    values = dict(zip(StateFeatureRidge.FEATURE_NAMES, features))
    assert values["x"] == pytest.approx(-1.0 / 1.5)
    assert values["y"] == pytest.approx(0.5 / 1.5)
    assert values["vx"] == pytest.approx(1.0 / 1.5)
    assert values["vy"] == pytest.approx(-0.5 / 1.5)
    assert values["wall_distance"] == pytest.approx(1.0 / 1.5)
    assert values["gap_signed_distance"] == pytest.approx(0.5 / 1.5)
    assert values["gap_width"] == pytest.approx(0.6 / 1.5)
    assert values["goal_distance"] == pytest.approx(np.hypot(2.5, 0.5) / 1.5)
    assert values["speed"] == pytest.approx(np.hypot(1.0, 0.5) / 1.5)
    assert values["wall_directed_speed"] == pytest.approx(1.0 / 1.5)
    expected_gap_speed = (1.25 / np.sqrt(1.25)) / 1.5
    assert values["gap_directed_speed"] == pytest.approx(expected_gap_speed)
    assert values["noise_scale"] == pytest.approx(2.0)
    assert [values[f"regime_R{index}"] for index in range(3)] == [0.0, 0.0, 1.0]
    assert values["wall_distance_x_wall_speed"] == pytest.approx(1.0 / 1.5**2)
    assert values["gap_distance_x_gap_speed"] == pytest.approx(
        (0.5 / 1.5) * expected_gap_speed
    )

    object_features = StateFeatureRidge.feature_vector(SimpleNamespace(**checkpoint()))
    np.testing.assert_allclose(object_features, features)


def test_feature_extraction_ignores_model_scores_and_outcome_fields():
    clean = checkpoint(model_score=-1e100, csr=-100.0, outcome=0.0)
    contaminated = checkpoint(model_score=1e100, csr=100.0, outcome=1.0)
    np.testing.assert_array_equal(
        StateFeatureRidge.feature_vector(clean),
        StateFeatureRidge.feature_vector(contaminated),
    )
    assert not any("score" in name for name in StateFeatureRidge.FEATURE_NAMES)


def test_fit_rejects_explicit_confirmatory_data():
    checkpoints = [checkpoint(split="confirm"), checkpoint()]
    with pytest.raises(ValueError, match="development data only"):
        StateFeatureRidge().fit(checkpoints, [0.0, 1.0], [0, 1])


def test_grouped_cv_uses_training_fold_scalers_only():
    x_values = [-1.0, -0.8, -0.6, -0.4, 100.0, 102.0]
    checkpoints = [checkpoint(x=x, state=(x, 0.5, 1.0, -0.5)) for x in x_values]
    outcomes = np.array([0.0, 0.0, 0.2, 0.2, 1.0, 1.0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    model = StateFeatureRidge().fit(checkpoints, outcomes, groups)

    assert len(model.cv_folds_) == 4 * 3
    for ridge in model.cv_scores_:
        folds = [fold for fold in model.cv_folds_ if fold["lambda"] == ridge]
        assert {int(fold["held_out_group"]) for fold in folds} == {0, 1, 2}
    extreme_fold = next(
        fold
        for fold in model.cv_folds_
        if fold["lambda"] == 0.01 and fold["held_out_group"] == 2
    )
    expected_mean = np.mean(
        np.vstack([StateFeatureRidge.feature_vector(item) for item in checkpoints[:4]]),
        axis=0,
    )
    np.testing.assert_allclose(extreme_fold["feature_mean"], expected_mean)
    np.testing.assert_array_equal(extreme_fold["train_groups"], [0, 1])
    assert extreme_fold["n_validation"] == 2
    np.testing.assert_array_equal(model.development_seed_groups_, [0, 1, 2])
    assert model.coef_.shape == (17,)


def test_raw_predictions_are_retained_and_predict_clips():
    first = checkpoint(state=(-1.0, 0.5, 0.0, 0.0), goal=(0.0, 0.5))
    second = checkpoint(state=(-1.0, 0.5, 0.0, 0.0), goal=(2.0, 0.5))
    extrapolated = checkpoint(state=(-1.0, 0.5, 0.0, 0.0), goal=(4.0, 0.5))
    model = StateFeatureRidge().fit(
        [first, second], [0.0, 1.0], [0, 1], lambdas=(1.0,)
    )
    assert model.predict_raw([second])[0] == pytest.approx(5.0 / 6.0)
    assert model.predict_raw([extrapolated])[0] > 1.0
    assert model.predict([extrapolated])[0] == 1.0
    np.testing.assert_allclose(
        model.development_predictions_,
        np.clip(model.raw_development_predictions_, 0.0, 1.0),
    )


def test_closed_form_ridge_fixture_matches_plan_exactly():
    fixture = closed_form_ridge_fixture()
    np.testing.assert_array_equal(fixture["train_z"], [-1.0, 1.0])
    np.testing.assert_array_equal(fixture["train_y"], [0.0, 1.0])
    assert fixture["lambda"] == 1.0
    assert fixture["intercept"] == pytest.approx(1.0 / 2.0)
    assert fixture["slope"] == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(fixture["z"], [1.0, 2.0])
    np.testing.assert_allclose(fixture["raw_predictions"], [5.0 / 6.0, 7.0 / 6.0])
    np.testing.assert_allclose(fixture["clipped_predictions"], [5.0 / 6.0, 1.0])
