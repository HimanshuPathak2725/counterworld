import numpy as np

from counterworld.datasets import make_checkpoint_panel, make_training_dataset, stratified_subset, with_gap_width
from counterworld.environment import DoubleIntegratorGapEnv


def test_r1_r2_share_physical_checkpoints_but_change_noise():
    r1 = make_checkpoint_panel("R1", 500)
    r2 = make_checkpoint_panel("R2", 500)
    for first, second in zip(r1, r2):
        np.testing.assert_array_equal(first.state, second.state)
        assert first.gap_center == second.gap_center
        assert first.checkpoint_id == second.checkpoint_id
        assert first.sigma_v == 0.02
        assert second.sigma_v == 0.04


def test_r0_has_three_equal_strata_and_deterministic_panels():
    first = make_checkpoint_panel("R0", 500)
    repeated = make_checkpoint_panel("R0", 500)
    for left, right in zip(first, repeated):
        np.testing.assert_array_equal(left.state, right.state)
        assert left.gap_center == right.gap_center
        assert left.checkpoint_id == right.checkpoint_id
    assert len(first) == 48
    assert max(point.state[0] for point in first[:16]) <= -0.8
    assert -0.8 <= min(point.state[0] for point in first[16:32])
    assert min(point.state[0] for point in first[32:]) >= -0.4


def test_subset_is_stratified_and_width_change_preserves_states():
    panel = make_checkpoint_panel("R0", 500)
    subset = stratified_subset(panel, 12)
    assert len(subset) == 12
    changed = with_gap_width(subset, 0.4)
    for before, after in zip(subset, changed):
        np.testing.assert_array_equal(before.state, after.state)
        assert after.gap_width == 0.4


def test_training_collection_is_safe_and_episode_split_is_whole():
    dataset = make_training_dataset(DoubleIntegratorGapEnv(), 100)
    assert np.all(dataset.next_states[:, 0] < -0.25)
    assert set(dataset.episode_ids) == set(range(64))
    train = np.isin(dataset.episode_ids, range(52))
    holdout = np.isin(dataset.episode_ids, range(52, 64))
    assert not np.any(train & holdout)
    assert np.any(train) and np.any(holdout)
