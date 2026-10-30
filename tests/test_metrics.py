from dataclasses import dataclass

import numpy as np

from counterworld.environment import Checkpoint as EnvironmentCheckpoint, DoubleIntegratorGapEnv
from counterworld.metrics import candidate_coverage, evaluate_paid_scores, evaluate_target
from counterworld.noise import NoiseRegistry


@dataclass
class Record:
    total_return: float
    states: list
    actions: list
    collision: bool = False
    goal: bool = False
    real_environment_steps: int = 1
    model_calls: int = 1


class Model:
    members = [object(), object()]

    def predict_next(self, state, action, canonical_noise, member_index=None):
        return np.array([1.0, 0.0, 2.0, 0.0])


class Registry:
    def normal(self, *args, shape):
        return np.zeros(shape)


class Planner:
    def __init__(self):
        self.model = Model()
        self.selection_replicates = 4

    def run_episodes(self, checkpoint, controller, execution, noise_registry, stream, checkpoint_id, k, member_index=None):
        if stream == "outcome-evaluation":
            return [Record(-4, [], [], collision=index % 2 == 0) for index in range(k)]
        if stream == "heldout-return":
            return [Record(-4, [[0, 0, 0, 0], [0.75, 0, 0, 0]], [2]) for _ in range(k)]
        if controller == "reference":
            return [Record(0.49, [], []) for _ in range(k)]
        if execution == "model":
            return [Record(0.99, [], []) for _ in range(k)]
        return [Record(-4, [], []) for _ in range(k)]


class Checkpoint:
    checkpoint_id = 7


def test_closed_form_candidate_metrics_match_research_fixture():
    scores = evaluate_paid_scores(None, Planner(), Checkpoint(), Registry(), k=4)
    assert scores["csr"] == 4.49
    assert scores["peg"] == 4.99
    assert scores["heldout_return"] == -4
    assert scores["one_step_error"] == 4.0625


def test_target_uses_independent_odd_even_splits():
    target = evaluate_target(None, Planner(), Checkpoint(), Registry(), episode_count=6)
    assert target["collision_probability"] == 0.5
    assert target["collision_probability_a"] == 1.0
    assert target["collision_probability_b"] == 0.0


def test_candidate_coverage_detects_safe_and_colliding_candidates():
    class CandidatePlanner:
        selection_replicates = 4

        def __init__(self, action):
            self.action = action

        def candidate_actions(self, registry, checkpoint_id, episode_id, time_step):
            return np.broadcast_to(self.action, (4, 12, 2)).copy()

    env = DoubleIntegratorGapEnv()
    checkpoint = EnvironmentCheckpoint(
        state=np.array([-1.0, 0.0, 0.0, 0.0]),
        gap_center=1.0,
        gap_width=0.6,
        regime="R0",
        checkpoint_id=1,
    )
    safe = candidate_coverage(env, CandidatePlanner(np.array([0.0, 0.0])), checkpoint, NoiseRegistry(1))
    collision = candidate_coverage(env, CandidatePlanner(np.array([1.0, 0.0])), checkpoint, NoiseRegistry(1))
    assert safe["strictly_collision_free_candidates"] == 4
    assert not safe["candidate_starved"]
    assert collision["strictly_collision_free_candidates"] == 0
    assert collision["candidate_starved"]
