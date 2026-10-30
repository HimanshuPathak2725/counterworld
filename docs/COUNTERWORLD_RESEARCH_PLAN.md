# Counterworld v0.1 — Research Plan

**Status:** Review draft only. No implementation is authorized by this document.  
**Protocol family:** `CW-0.x`; all environment, probe, and artifact protocols are unstable until independently reviewed.  
**Research question:** Do paired, decision-focused probes provide failure information that standard model and policy metrics do not already provide?

## 0. Declared substitutions and narrowing

These choices are made before any result is examined:

1. The broad Counterworld concept is narrowed to **one explicit-state, explicit-noise, two-dimensional double-integrator testbed with a wall present in the environment but deleted from every model**. A separate one-dimensional analytic case is used to verify metric arithmetic.
2. **No metric novelty is claimed.** The paired exploitability gap is an instance of the existing XGap idea; paired optimization regret is an MPC instance of established decision regret. The proposed contribution is a reproducible probe protocol, analytic oracles, and a pre-registered matched-budget evaluation of their incremental usefulness.
3. Only **paired optimization regret (POR)** is confirmatory. The paired exploitability gap (PEG) and first-action flip rate are secondary diagnostics; they are not required to beat the baselines.
4. The claim is limited to **failure localization at explicit checkpoints in this testbed**. It is not a claim about robots, images, latent models, general world models, online deployment, or compute-efficient planning.
5. All interventions are executed in a simulator with explicit state and noise. There is **no observational causal-inference claim** and no attempt to infer unobserved counterfactuals from logged data.
6. **“Action-information leakage” is dropped from v0.** There is no critic, baseline, or auxiliary hindsight channel in this protocol, and the phrase has no unique estimand here. A future study must define the random variables and either an information measure or a specific estimator bias before reintroducing it.
7. The core is **NumPy-only**, with Python’s standard library permitted. There is no server, UI, sandbox, Hugging Face adapter, neural model, plugin stability promise, or general training framework in v0.
8. POR uses an environment oracle and is an **offline diagnostic**, not an online planning method. Real-transition budgets are matched across scores; model/planner compute is measured separately, and no compute-dominance claim is made.

## 1. Thesis and kill criterion

### Falsifiable thesis

**Under matched real-transition budgets in the stochastic double-integrator deleted-wall testbed, paired optimization regret ranks checkpoint-level collision risk more accurately than ensemble return variance, planner-trajectory one-step error, or held-out MPC return, achieving at least a 0.10 Spearman-rank improvement over every baseline in at least two of three pre-registered regimes.**

### Kill criterion

Stop expansion and reject the thesis if, for at least two of the three regimes, the minimum POR-minus-baseline Spearman improvement across the three baselines is either **less than 0.10** or has a **Holm-adjusted 95% cluster-bootstrap lower confidence bound less than or equal to zero**.

A pass supports only this protocol and testbed. Any broader claim requires a new pre-registration. A failed thesis is not to be rescued by renaming a post-hoc metric as confirmatory.

## 2. Provisional user personas

These are **design personas, not verified user research**. Their pains and workarounds are hypotheses to validate through interviews; no adoption claim is made.

### Persona A — Model-based RL researcher

- **Specific pain today:** A planner fails near a constraint even though held-out one-step error is low and ensemble members agree. The researcher cannot efficiently determine which checkpoints and candidate actions make the model decision-relevantly wrong.
- **Current workaround:** Bespoke notebooks for reset-matched rollouts, manual seed handling, hand-picked failure visualizations, and one-off scripts that are difficult to reproduce or compare across model versions.
- **Desired outcome:** A checkpoint table with POR/PEG, independent collision outcomes, exact budgets, and replayable probe provenance.

### Persona B — Controls/RL engineer evaluating a model-based controller

- **Specific pain today:** Before using a model near a contact or boundary condition, the engineer needs to know whether confidence and average held-out performance survive a shift in speed, starting state, or noise. Aggregate return does not localize the risky operating region.
- **Current workaround:** Large end-to-end evaluation batches, manual inspection of failures, and conservative safety rules that may hide rather than diagnose model mismatch.
- **Desired outcome:** A matched-budget risk-ranking diagnostic with explicit limitations, not a safety certificate.

### Persona C — Open-source maintainer or reviewer of RL evaluation code

- **Specific pain today:** Contributors report different seeds, rollout budgets, and aggregate scores, making alleged improvements expensive to verify and easy to overstate.
- **Current workaround:** CI smoke tests, manual plot review, and rerunning author-specific scripts. Analytic expected values and confirmatory comparisons are usually missing.
- **Desired outcome:** A small protocol with known analytic answers, explicit seed splits, pre-registered baselines, and machine-readable provenance.

## 3. Targeted prior-art analysis

This is a **targeted rather than systematic review**. “Unserved” below means that the exact combination required here is not supplied by that cited source; it is not a claim that no related work exists elsewhere. In particular, XGap and decision regret are prior art, and the plan does not depend on inventing them.

| Related tool or paper | What it measures or provides | What it does not directly provide for this plan | What remains unserved in the reviewed scope |
|---|---|---|---|
| [Yu et al., *How Should World Models Be Evaluated for Embodied Decision-Making?* (2026)](https://arxiv.org/html/2606.15032v2) | An L0–L7 evaluation framework covering intervention fidelity, closed-loop validity, policy ranking, optimization, calibration, and an exploitability gap, **XGap = model-predicted value minus real value for model-optimized behavior**. | It does not supply the explicit-noise NumPy estimator, analytic deleted-wall oracle, or pre-registered checkpoint-level incremental-value test proposed here. PEG is explicitly an instance of XGap, not a new metric. | An open, small reference protocol that tests whether a paired XGap/regret estimator adds information beyond strong baselines under matched real-transition budgets. |
| [Jiang et al., *World-in-World: World Models in a Closed-Loop World* (ICLR 2026)](https://arxiv.org/html/2510.18135v2) and [code](https://github.com/World-In-World/world-in-world) | Closed-loop embodied task success across four tasks, a unified action API, heterogeneous visual world models, and planning/inference scaling. | Its primary outcome is end-to-end task success rather than explicit-state, reset-matched decision regret or noise-paired checkpoint attribution. | A complementary low-level diagnostic for why a particular checkpoint is risky, with analytic expected values and independent outcome streams. |
| [Elmachtoub & Grigas, *Smart “Predict, then Optimize”*](https://arxiv.org/abs/1710.08005) | Decision error/regret induced by predictions in predict-then-optimize problems and the SPO+ surrogate. | It does not provide a stochastic closed-loop MPC probe protocol, wall-deletion testbed, or comparison against ensemble variance and held-out MPC return. POR is an instantiation of this established decision-regret idea. | Evidence about when regret probes are worth their oracle cost for RL failure localization, rather than another generic regret definition. |
| [Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* / PETS](https://arxiv.org/abs/1805.12114) | Probabilistic ensemble dynamics, uncertainty propagation, trajectory sampling, planning performance, and sample efficiency. | The algorithm and benchmark returns do not by themselves diagnose a structurally omitted constraint at a checkpoint or establish that ensemble disagreement is a sufficient failure warning. | A controlled test in which ensemble members can agree because the same wall is absent from every member, with an explicit decision-relevant comparator. |
| [Janner et al., *When to Trust Your Model: Model-Based Policy Optimization* / MBPO](https://arxiv.org/abs/1906.08253) | Model usage in policy optimization, short branched rollouts, model bias, asymptotic return, and sample efficiency. | It does not supply a per-checkpoint attribution of model optimism, oracle decision regret, or a matched-budget failure-localization score. | A diagnostic complement that distinguishes average model usefulness from localized planner exposure to missing constraints. |
| [Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*](https://arxiv.org/abs/1612.01474) | Predictive uncertainty quality, calibration, accuracy, and behavior under distribution shift. | Predictive uncertainty and ensemble spread do not, by definition, measure the return lost by following a model-selected plan or cover alternatives absent from the model family. | A direct test of whether ensemble return variance ranks collision risk as well as decision-focused probes in a shared-misspecification case. |
| [Ovadia et al., *Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*](https://arxiv.org/abs/1906.02530) | Accuracy and calibration of probabilistic predictors under dataset shift. | It does not evaluate action-sequence selection, executed planner regret, or constraint collisions in closed-loop control. | A shift evaluation whose target is a downstream decision outcome, not only predictive calibration. |
| [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://proceedings.neurips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) and [RLiable](https://github.com/google-research/rliable) | Robust aggregate performance, performance profiles, and stratified-bootstrap interval estimates across runs/tasks. | Aggregate policy performance does not identify model-specific failure mechanisms or checkpoint-level decision regret. | The present plan adopts interval estimation rather than treating RLiable as a competing metric; the open need is reliable uncertainty statements for clustered probe data. |

**Prior-art conclusion:** The field is not missing decision-aware evaluation or exploitability concepts. The narrower unserved artifact is an **analytically checkable, explicitly paired, open reference experiment that tests their incremental failure-localization value under matched budgets**. Whether that artifact is useful remains unverified and is exactly what the kill criterion tests.

## 4. Metric specifications

### 4.1 Shared definitions

- \(E\): true environment containing the wall.
- \(M\): the eight-member ensemble used by the planner through its mean predicted return. Every member has identical free-dynamics and reward interfaces but **no wall**.
- \(s\): fully observed checkpoint state.
- \(H=12\): planning/evaluation horizon; discount \(\gamma=0.99\).
- \(\mathcal U_s\): one shared set of 512 candidate action sequences, sampled before model and oracle selection. This removes candidate-set differences.
- \(\pi^M_s\): the candidate with highest ensemble-mean model-estimated return.
- \(\pi^E_s\): the candidate with highest environment-oracle-estimated return.
- Selection uses **four noise replicates per candidate**, shared by model and oracle selection but independent of metric-evaluation and outcome-evaluation streams.
- \(J^W_H(s,\pi)=\mathbb E[\sum_{t=0}^{H-1}\gamma^t r_t]\) for \(W\in\{E,M\}\), with terminal transitions included and no bootstrap beyond the horizon.
- \(C_H(s)\): collision probability for the model-based MPC controller from checkpoint \(s\), evaluated under an independent outcome stream.
- Candidate ties are resolved by the lowest candidate index, fixed before execution.

POR and PEG are computed from **16 paired environment rollouts per selected plan**. For each evaluation index \(j\), the same explicit noise vector is used for the model-selected and oracle-selected plans, and for the model prediction when evaluating PEG. Selection noise is never reused for metric evaluation; outcome noise is fresh and independent.

### 4.2 Candidate and baseline metrics

| Metric | Formal definition | Pre-registered estimator | Units and interpretation | Closed-form value in the analytic case below |
|---|---|---|---|---|
| **Paired Optimization Regret (POR)** | \(D_H(s)=J^E_H(s,\pi^E_s)-J^E_H(s,\pi^M_s)\) | Mean of 16 paired differences: oracle-plan return minus model-plan return on shared fresh noise. Nonnegative up to Monte Carlo error. | **Reward units.** Return lost by using the model’s selected plan instead of the environment oracle’s selected plan. | **4.49** |
| **Paired Exploitability Gap (PEG)** | \(G_H(s)=J^M_H(s,\pi^M_s)-J^E_H(s,\pi^M_s)\) | Mean of 16 paired differences: model-predicted minus environment-executed return on the model-selected plan. | **Reward units, signed.** Positive means model optimism; negative means pessimism. This is a paired estimator of XGap. | **+4.99** |
| **First-action flip rate (DFR)** | \(F=N^{-1}\sum_{i=1}^{N}\mathbf 1[\pi^E_{s_i,0}\neq\pi^M_{s_i,0}]\) | Fraction of checkpoints whose first selected action differs between model and oracle, using the fixed tie rule. | **Probability/proportion in \([0,1]\).** A mechanism diagnostic, not the primary score. | **1** |
| **Ensemble return variance (baseline)** | \(V_H(s)=\frac1{m-1}\sum_{k=1}^{m}(\hat J_{H,k}(s,\pi^M_s)-\overline{\hat J})^2\), \(m=8\) | Sample variance of ensemble-predicted returns over 16 model rollouts per member, with the same action plan and noise across members. | **Reward units².** High values are treated as higher risk. It is not automatically calibrated confidence. | **0** |
| **Planner-trajectory one-step error (baseline)** | \(E_1=\mathbb E[\|\tilde f_M(s,a,\xi)-\tilde f_E(s,a,\xi)\|_2^2]\), where \(\tilde f\) is the normalized next state | Mean squared error over transitions visited in the 32 held-out MPC rollouts; positions and velocities are normalized by fixed pre-registered scales before squaring. | **Normalized-state units².** Lower is treated as better; high values are treated as higher risk. | **4.0625** |
| **Held-out MPC return (baseline)** | \(R_{\mathrm{held}}(s)=J^E_{H,\mathrm{MPC}}(s,\pi^M)\) | Mean return of 32 independent held-out MPC episodes from the checkpoint. The risk score is its negation. | **Reward units.** This is the strong end-to-end baseline. | **0.99 in the wall-free held-out world** |
| **Independent collision outcome (target, not a candidate score)** | \(p_C(s)=\Pr(\text{wall collision within }H\mid s,\pi^M_{\mathrm{MPC}})\) | Fraction of 32 fresh MPC outcome rollouts with a collision. These rollouts are not used to compute any candidate score. | **Probability in \([0,1]\).** Used only for independent evaluation of risk ranking. | **1 under the wall-containing environment** |

No candidate score is computed from the outcome-evaluation stream. This separation prevents the metric from simply restating its target.

### 4.3 Analytic oracle: one-dimensional special case

This is an **analytic calibration case, not an empirical result**.

- State \(x=0, v=0\); horizon \(H=1\); no noise; discount is irrelevant.
- Actions \(a\in\{0,1,2\}\), with \(x^+=x+v+0.5a\) and \(v^+=v+a\).
- Goal \(x=2\); wall \(x=0.75\); collision terminates, clamps position to the wall, sets velocity to zero, and gives reward \(-4\) instead of the ordinary reward.
- Ordinary one-step reward is progress minus step cost: \((x^+-x)-0.01\).
- The model deletes the wall but otherwise uses identical dynamics and reward rules.

| Action | Free next position | Model return | Environment return |
|---|---:|---:|---:|
| 0 | 0 | -0.01 | -0.01 |
| 1 | 0.5 | 0.49 | 0.49 |
| 2 | 1.0 | 0.99 | **-4** |

The model selects \(a=2\); the environment oracle selects \(a=1\). Therefore:

- **POR:** \(0.49-(-4)=\mathbf{4.49}\).
- **PEG:** \(0.99-(-4)=\mathbf{4.99}\).
- **DFR:** the first action differs, so **1**.
- **Ensemble return variance:** all deleted-wall members predict the same return, so **0**, despite the structural error.
- **One-step error:** with unit normalization for this analytic check, \((1.0-0.75)^2+(2-0)^2=\mathbf{4.0625}\).
- **Held-out return:** on the wall-free held-out world, the model-selected action returns **0.99**; under the wall-containing evaluation environment, collision probability is **1**.

This case supplies known expected values for unit and metamorphic tests. It does not make the confirmatory hypothesis true.

## 5. Provenance for portfolio-derived design inputs

Only the following portfolio-derived inputs are used. Each has verbatim provenance. No other statement about Himanshu’s experiments is assumed.

| Design input | Verbatim quoted text and source | How it is used |
|---|---|---|
| Missing constraints and fixed-budget comparison | “The missing-obstacle example is a small version of a broader problem. Better search cannot recover a constraint the model never represents. A useful next experiment would change what the model knows, hold the planning budget fixed, and measure whether the executed behavior improves.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Motivates the deleted-wall construction, fixed planner budget, and executed outcome. It does not supply the new POR estimator or study results. |
| Closed-loop replanning is not a sufficient repair | “That does not make MPC immune to a missing obstacle. If every new plan uses a model that excludes an important constraint, the planner can repeat the same mistake.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Justifies measuring independent collision outcomes under closed-loop MPC rather than assuming replanning solves the deletion. |
| Matched protocol and exploitability warning | “That comparison needs a fixed environment, action space, planning horizon, and data budget. Its outcome is the behavior in the environment. A lower predicted cost alone could mean the planner has become better at exploiting the model’s errors.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Fixes environment/action/horizon/budgets and treats lower model cost as insufficient evidence. |
| Reproducibility requirements | “A useful report contains the hypothesis, implementation, dataset and evaluation splits, matched budgets, seeds, uncertainty in the estimates, and representative failures.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Defines the minimum artifact and reporting contents. |
| Narrow intervals can be wrong under shift | “A shrinking band was an accurate description of the model’s calculation. It was poor evidence that the calculation was adequate outside the observed region.” — [What uncertainty can tell a planner](https://new-portfolio-henna-ten.vercel.app/posts/what-uncertainty-can-tell-a-planner/) | Prevents treating narrow ensemble spread as evidence of correctness under shift. |
| Shared misspecification cannot be repaired by adding identical members | “If every ensemble member extrapolates as a line, adding members will not introduce our missing bending hypotheses.” — [What uncertainty can tell a planner](https://new-portfolio-henna-ten.vercel.app/posts/what-uncertainty-can-tell-a-planner/) | Motivates the shared-deleted-wall ensemble baseline. |
| Limits of synthetic evidence | “This is a synthetic test with a deliberately chosen extrapolation failure. It reports neither real-robot performance nor a replication of a published benchmark.” — [What uncertainty can tell a planner](https://new-portfolio-henna-ten.vercel.app/posts/what-uncertainty-can-tell-a-planner/) | Supplies the pattern for this plan’s explicit synthetic-test limitation. |

**Unverified items:** the personas, adoption potential, any claim of exhaustive coverage beyond the cited review, and the thesis itself. No numerical result outside Section 4.3 is claimed as observed evidence.

## 6. v0 protocol and first testbed

### 6.1 Protocol boundaries

Protocol identifiers are **`CW-ENV-0.1-draft`**, **`CW-PROBE-0.1-draft`**, and **`CW-ARTIFACT-0.1-draft`**. Any may change without compatibility promises during 0.x review.

The environment protocol requires:

1. **Explicit state:** a reset returns the complete numerical state; no observation encoder, hidden memory, or partial observability.
2. **Explicit noise:** each transition consumes a fully specified noise vector drawn from a declared distribution. The dynamics function does not consume hidden randomness.
3. **Independent streams:** training, planner selection, metric evaluation, outcome evaluation, and bootstrap resampling use separately identified streams. Reusing a stream across these roles is a protocol violation.
4. **Reset-matched intervention:** a probe restores the same explicit checkpoint and evaluates different action sequences with paired noise where required.
5. **Provenance:** every transition stores protocol version, environment/configuration hash, model/oracle identifier, candidate-set identifier, selection/evaluation stream identifiers, action, reward, terminal/collision flags, and real-transition/model-call counters.
6. **No hidden extensibility:** there is no adapter, server, UI, sandbox, Hugging Face integration, neural network, or stable plugin API in v0.

### 6.2 Environment: deleted-wall double integrator

- State \(s=(x,y,v_x,v_y)\); action \(a=(a_x,a_y)\) from the nine-element grid \(\{-1,0,1\}^2\); time step \(\Delta t=1\).
- Free dynamics:
  \[
  p_{t+1}=p_t+v_t+\tfrac12a_t+\sigma_p\varepsilon_{p,t},\qquad
  v_{t+1}=v_t+a_t+\sigma_v\varepsilon_{v,t}.
  \]
- \(\varepsilon_{p,t}\) and \(\varepsilon_{v,t}\) are independent two-dimensional standard-normal vectors. Training and nominal noise use \(\sigma_p=\sigma_v=0.02\); the noise-shift regime uses \(0.04\).
- Wall: the line segment \(x=0,\ |y|\leq0.75\). Collision uses segment intersection, so tunneling between discrete positions is not allowed. Collision terminates the episode, clamps position to the intersection, zeroes velocity, and returns \(-4\) instead of ordinary reward.
- Goal: \(g=(1.5,0)\), radius \(0.1\). Reaching it terminates and adds \(+1\).
- Ordinary reward: \(-0.01+\|p_t-g\|_2-\|p_{t+1}-g\|_2\). The same reward interface is used by model and environment.
- The model deletes the wall and its collision branch. Free dynamics and noise distributions are otherwise identical.

### 6.3 Model, data, and planner

- **Model family:** eight linear ridge-dynamics members fitted with NumPy on eight bootstrap episode samples. Residual covariance is estimated from held-out episodes. No member receives a wall feature or collision class; deletion is structural.
- **Training data:** 64 episodes × 64 maximum transitions, with episode-level rather than transition-level train/held-out splitting (52/12 episodes). Collection starts with \(x\in[-2,-1]\), \(y\in[-0.75,0.75]\), zero velocity, and terminates before data can cross \(x=-0.25\). This creates a safe, left-of-wall training distribution.
- **Planner:** random-shooting MPC, 512 candidate action sequences, horizon 12, discount 0.99, four selection-noise replicates per candidate, first-action execution with replanning at every environment step. The same candidate set is used for model and oracle selection at each checkpoint.
- **Oracle:** identical planner and candidate set, but queries the wall-containing environment. It exists only for offline evaluation and is not a deployable controller.

### 6.4 Checkpoint regimes

Each confirmatory model seed contributes 24 stratified checkpoints per regime:

1. **R0 — nominal:** eight far \((x\in[-1.5,-0.8])\), eight middle \((x\in[-0.8,-0.4])\), and eight near-wall \((x\in[-0.4,-0.1])\) checkpoints; \(y\in[-0.5,0.5]\), \(v_x\in[0,0.5]\), \(v_y\in[-0.25,0.25]\), and nominal noise.
2. **R1 — near-wall/high-speed:** \(x\in[-0.4,-0.1]\), \(y\in[-0.5,0.5]\), \(v_x\in[0.75,1.5]\), \(v_y\in[-0.25,0.25]\), and nominal noise.
3. **R2 — noise shift:** the same checkpoint states as R1 with position and velocity noise doubled to \(0.04\).

This deliberately tests structural misspecification. It does not test wall discovery, representation learning, or general intelligence.

## 7. Flagship pre-registered analysis plan

### 7.1 Seed and checkpoint allocation

- **Development seeds:** 8 seeds (`0–7`) for implementation checks and freezing planner constants. They are excluded from confirmatory estimates.
- **Confirmatory seeds:** 32 independent model/training seeds (`100–131`), each with eight bootstrap ensemble members.
- **Checkpoints:** 24 per seed per regime, giving \(32\times3\times24=\mathbf{2,304}\) checkpoints.
- **Planner selection:** four noise replicates per candidate, with one shared model/oracle selection stream that is independent of metric and outcome streams.
- **Metric evaluation:** 16 fresh paired rollouts for the model-selected plan and 16 for the oracle-selected plan per checkpoint.
- **Held-out-return baseline:** 32 held-out MPC rollouts per checkpoint.
- **Independent outcome evaluation:** 32 additional fresh MPC rollouts per checkpoint, used only to estimate collision probability.
- **Bootstrap:** 10,000 stratified resamples of the 32 confirmatory seed clusters, with a separately recorded bootstrap seed.

The large checkpoint count is not treated as 2,304 independent observations. The **model/training seed is the inference cluster**. The 32-cluster allocation is a design constraint, not a completed power analysis; statistical power is **unverified** and will not be used to strengthen the claim.

### 7.2 Matched information and real-transition budgets

| Score | Real environment rollouts available per checkpoint | Additional model/oracle computation |
|---|---:|---|
| POR + PEG | 16 model-plan + 16 oracle-plan = **32** | Model-predicted returns for PEG and oracle planning; calls recorded separately |
| Held-out MPC return | **32** | Model-based MPC planning |
| One-step error | Reuses transitions from those **32** held-out MPC rollouts | Model one-step predictions |
| Ensemble return variance | **0 additional real rollouts** | Eight members × 16 model rollouts; calls recorded |
| Independent collision target | **32 fresh rollouts for every method**, never reused by a score | Model-based MPC planning |

POR and PEG share their rollouts and receive no extra environment information. The independent outcome budget is identical for all methods. Planner/model calls are **not** claimed to be compute-matched; they are reported as a separate cost.

### 7.3 Primary endpoint and comparisons

For each regime independently:

1. Compute each candidate score at every checkpoint.
2. Compute independent collision probability from the outcome stream.
3. Measure **Spearman rank correlation** between each score and collision probability. Higher ensemble variance, one-step error, PEG/POR, and **negative** held-out return are all oriented so that larger values mean higher predicted risk.
4. Compute POR-minus-baseline correlation differences against all three baselines: ensemble return variance, planner-trajectory one-step error, and negative held-out MPC return.
5. Estimate 95% percentile cluster-bootstrap intervals by resampling model/training seeds, preserving every checkpoint and regime within a resampled seed.
6. Apply **Holm adjustment across the three baseline comparisons within each regime**. Development seeds are never pooled into these estimates.

The confirmatory requirement is the thesis threshold: at least **0.10 Spearman improvement over every baseline in at least two regimes**, with adjusted lower confidence bounds above zero.

### 7.4 Secondary and diagnostic analyses

These are explicitly secondary and do not rescue a failed primary result:

- AUROC for the pre-registered high-risk label \(p_C(s)\geq0.5\).
- PEG sign and magnitude by distance-to-wall stratum.
- First-action flip rate and its relationship to POR.
- Sensitivity to 8, 16, and 32 metric-evaluation rollouts, using nested pre-generated streams.
- Sensitivity to 256 versus 512 planner candidates, fixed before confirmatory execution and reported separately.
- Reporting of model/oracle planning calls and wall-clock cost, without an online-efficiency claim.

### 7.5 Protocol discipline

- No silent seed removal, outlier deletion, or selective reporting of regimes. Numerical failures are logged and counted as protocol failures.
- Planner constants, normalization scales, checkpoint strata, and analysis decisions are frozen after development and before confirmatory execution.
- This document, the frozen configuration, and the analysis plan are hashed before confirmatory execution.
- Any change after seeing confirmatory outcomes is labeled **exploratory**, with the original analysis retained.
- Report all seed-level results, intervals, failed runs, and representative failure traces, including negative outcomes.

## 8. Review boundary

The proposed v0 is deliberately small: **one structural failure, one explicit environment, one model family, one planner, three shift regimes, and one confirmatory comparison**. It should survive only if decision-focused probes demonstrate measurable incremental value over the strong baselines under the stated budgets.

No code has been written for this research plan. Review stops here.
