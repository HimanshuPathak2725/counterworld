# Counterworld

Counterworld is a **development-stage, NumPy-only research toolkit** for evaluating when model-based controllers fail because of missing constraints, candidate starvation, or model mismatch.

The current testbed is an explicit-state, explicit-noise double integrator with a randomized wall gap. The true environment contains the wall; every learned ensemble member deletes it.

**This is not a production RL framework or a confirmatory result.** There is no server, UI, sandbox, Hugging Face adapter, neural model, or claim of robot safety.

## What is implemented

- Deterministic named random streams and canonical Gaussian noise coupling
- Scalar and vectorized double-integrator transitions with continuous wall-collision detection
- Episode-bootstrap linear ridge ensembles with frozen residual covariance factors
- Random-shooting model-predictive control with shared candidate sets
- Closed-loop model and candidate-set-reference controllers
- Candidate-Set Regret (CSR), Paired Exploitability Gap (PEG), held-out return, one-step error, and ensemble return variance
- Development-only state-feature ridge baseline with seed-grouped cross-validation
- Nested evaluation budgets \(k\in\{4,8,16,32\}\)
- Collision variance, endpoint, and split-half reliability gates
- Per-seed Spearman statistics, cluster bootstrap utilities, and pilot power calculations
- Deterministic checkpoint-level parallel execution
- JSONL/JSON artifacts with configuration and file hashes

## Repository structure

```text
.
├── src/counterworld/
│   ├── environment.py   # Explicit state, dynamics, gaps, collisions, rewards
│   ├── noise.py         # Stable random streams and covariance factors
│   ├── models.py        # Ridge ensemble and imagined rollouts
│   ├── planner.py       # Candidate generation and closed-loop MPC
│   ├── metrics.py       # CSR, PEG, return, error, coverage, target estimates
│   ├── statistics.py    # Spearman, bootstrap, reliability, power, gates
│   ├── baselines.py     # Development-only state-feature ridge baseline
│   ├── datasets.py      # Training data and crossed checkpoint panels
│   ├── parallel.py      # Deterministic checkpoint worker execution
│   ├── pilot.py         # Development-only staged experiment runner
│   ├── protocol.py      # Frozen protocol and budget configuration
│   └── artifacts.py     # Provenance, canonical JSON, and hashes
├── tests/               # Analytic, integration, and equivalence tests
├── docs/                # Research-plan revisions and review constraints
├── pyproject.toml       # Package and test configuration
└── .gitignore           # Excludes dependencies and generated artifacts
```

## Requirements

- Python 3.11 or newer
- NumPy 2.0 or newer
- pytest 9 or newer for development

Install the package in editable mode:

```bash
python -m pip install -e '.[dev]'
```

The core runtime depends only on NumPy and Python’s standard library.

## Verification

```bash
python -m pytest
python -m pip wheel . --no-deps -w dist
```

Current verification status: **51 tests pass**, and the wheel builds successfully. Tests include the analytic CSR/PEG fixture, scalar/batched rollout equivalence, noise coupling, candidate coverage, baseline leakage prevention, and power-statistic edge cases.

## Development commands

Use small smoke runs first:

```bash
PYTHONPATH=src python -m counterworld.pilot \
  --stage targets \
  --output artifacts/smoke-targets \
  --seeds 0 \
  --max-checkpoints 2 \
  --workers 4 \
  --random-goals \
  --r0-speed-range 0.75 1.5
```

Run the full development target distribution:

```bash
PYTHONPATH=src python -m counterworld.pilot \
  --stage targets \
  --output artifacts/development-targets-r0-broadened \
  --workers 16 \
  --random-goals \
  --r0-speed-range 0.75 1.5
```

Run the nested paid-score sweep:

```bash
PYTHONPATH=src python -m counterworld.pilot \
  --stage sweep \
  --output artifacts/development-sweep-r0-broadened \
  --workers 8 \
  --random-goals \
  --r0-speed-range 0.75 1.5
```

For a shared server, reduce workers explicitly, for example `--workers 4`. No cloud compute is required.

The power-analysis stage is available through `--stage power` and requires a reviewed `--primary-k`. It should run only after the sweep’s reliability results have been reviewed.

## Artifacts and provenance

Generated files are excluded from Git:

- `artifacts/` — JSONL scores, summaries, and manifests
- `dist/` — wheel and build outputs
- `__pycache__/`, `.pytest_cache/`, and virtual environments

Each pilot manifest records protocol version, configuration hash, Python and NumPy versions, file hashes, and development-only status. Smaller-\(k\) runs are prefixes of larger-\(k\) runs, using nested deterministic streams.

Noise pairing uses a canonical standard-normal vector \(z\):

\[
\eta^E=L^Ez,\qquad \eta^{M_j}=L^{M_j}z.
\]

This is a common-random-number coupling, **not a claim that model and environment noise are identical when their covariance estimates differ**.

## Current development status

- The original R0 target was empirically degenerate: all development checkpoints had collision probability zero.
- The amended R0 design randomizes goal height and broadens wall-directed speed. The amended R0, R1, and R2 targets pass the registered variance, endpoint, interior-support, and reliability gates.
- All four required development seeds for the nested paid-score sweep are complete.
- Development power analysis completed at \(k=16\). CSR did not beat the required baselines: state-feature and one-step-error baselines were stronger in all three regimes, and the required seed count was unbounded for those comparisons. The pre-registered thesis is rejected; no confirmatory run is planned.
- The DYNA teaching application and unrelated materials are not part of this repository.

## Visual result showcase

The `showcase/` directory contains an interactive negative-result report, an eight-page PDF, standalone SVG figures, and a captioned 60-second explainer video. See `showcase/README.md` for open/rebuild instructions and verification results.

## Research documents

- [Research plan v0.1](docs/COUNTERWORLD_RESEARCH_PLAN.md)
- [Research plan v0.2](docs/COUNTERWORLD_RESEARCH_PLAN_V0_2.md)

These documents define the review boundaries, substitutions, estimands, baselines, and pre-registered analysis rules. Protocols remain unstable 0.x contracts.
