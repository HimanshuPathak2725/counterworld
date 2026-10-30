from counterworld.pilot import grouped, power_summary
from counterworld.protocol import PilotConfig


def synthetic_rows():
    rows = []
    for seed in (0, 1):
        for checkpoint_id, probability in enumerate((0.0, 0.25, 0.75, 1.0)):
            rows.append({
                "model_seed": seed,
                "regime": "R0",
                "k": 4,
                "checkpoint_id": checkpoint_id,
                "collision_probability": probability,
                "csr": probability,
                "ensemble_return_variance": 0.0,
                "one_step_error": 1.0 - probability,
                "heldout_return": -probability,
                "state_feature_baseline": probability,
            })
    return rows


def test_grouped_supports_compound_keys():
    rows = synthetic_rows()
    assert len(grouped(rows, "regime", "k")[("R0", 4)]) == 8


def test_power_summary_uses_within_seed_deltas_and_structured_counts():
    report = power_summary(synthetic_rows(), PilotConfig(), k=4)
    regime = report["regimes"]["R0"]
    assert regime["ensemble_return_variance"]["mean_delta"] == 1.0
    assert regime["one_step_error"]["mean_delta"] == 2.0
    assert regime["heldout_return"]["mean_delta"] == 0.0
    assert regime["state_feature_baseline"]["mean_delta"] == 0.0
    assert "required_n_uncapped" in regime["ensemble_return_variance"]
    assert "required_n_max" in report
