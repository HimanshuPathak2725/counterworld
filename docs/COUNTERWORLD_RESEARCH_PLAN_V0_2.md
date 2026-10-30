# Counterworld v0.2 — Research Plan for Review

**Status:** Development implementation in progress; confirmatory execution is not authorized.  
**Supersedes:** v0.1.  
**Protocol family:** `CW-0.2-draft`; all protocols remain unstable 0.x contracts.

> **Empirical-status declaration:** Only development and smoke artifacts are being produced. No confirmatory sample size is fixed, and no pilot result is treated as confirmatory evidence.

> **Development amendment (unapproved):** Early target distributions showed R0 collision probability identically zero across completed development seeds. The current development variant therefore samples R0 goal height uniformly from [-1, 1] and R0 wall-directed speed uniformly from [0.75, 1.5], while R1/R2 remain unchanged. This substitution was selected using target variance, interior support, and reliability—not CSR-minus-baseline effects. Original and amended runs are stored under distinct artifact paths and must not be pooled as a post-hoc success claim.

## 0. Substitutions and withdrawals declared before design

1. **Solid wall → randomized-gap wall.** The v0.1 solid wall is replaced by a wall containing one gap whose center varies across checkpoints. The goal remains fixed for comparability. This substitution is intended to create nondegenerate collision-probability variation rather than preserve an attractive but rank-deficient target.
2. **Open-loop POR/oracle → closed-loop Candidate-Set Regret (CSR).** The reference is no longer called an oracle and is not a global optimum. It is a candidate-set reference controller using the true environment. CSR is defined over closed-loop replanning trajectories and can be negative.
3. **Fixed N = 32 → development-determined N.** No confirmatory seed count is fixed until collision gates, split-half reliability, and a development-only pilot power analysis are complete.
4. **Held-out-return ambiguity withdrawn.** The baseline runs in the **wall-containing true environment**, not in the wall-free model and not merely in the training distribution.
5. **Open-loop regret remains only as an analytic special case.** The one-step analytic oracle has horizon one, so closed-loop and open-loop definitions coincide there; it is not used to redefine the main estimand.
6. **Only CSR is confirmatory.** PEG, first-action flip rate, and additional sensitivity analyses remain secondary. “Action-information leakage” remains dropped because it has no defined estimand in this study.
7. **No metric novelty is claimed.** CSR is a closed-loop, candidate-set-conditional decision-regret estimator; PEG is an instance of the existing XGap concept. The new evaluation is the protocol and its incremental evidence, not a new name for existing ideas.
8. **Empirical pilot work is explicitly pending.** Requested development distributions and power analysis are specified below but not fabricated. They must be reviewed before confirmatory execution.

## 1. Revised falsifiable thesis and kill rules

### Thesis

**In the randomized-gap, explicit-noise double-integrator testbed, closed-loop Candidate-Set Regret has higher mean per-seed Spearman correlation with independently estimated checkpoint collision probability than ensemble return variance, planner-trajectory one-step error, held-out wall-environment return, or a development-fitted state-feature baseline, with the one-sided lower confidence bound for each mean per-seed difference exceeding 0.10 in at least two of three pre-registered regimes.**

### Kill rules

The thesis and expansion plan are rejected if any of the following occurs:

1. A regime fails the pre-registered collision-variance or target-reliability gate after the permitted development-only redesign sequence.
2. The development power analysis shows that the required sample size exceeds the pre-registered cap, or that the planning effect is too small for a finite margin test.
3. On confirmatory data, fewer than two regimes have **all four baseline comparisons** satisfying a one-sided, multiplicity-adjusted lower confidence bound of **mean per-seed Spearman difference > 0.10**.

Passing supports only this testbed and protocol. It does not establish robot safety, general world-model quality, or superiority in other environments.

## 2. Development collision distributions and variance gate

### 2.1 Redesigned testbed geometry

- Position \(p=(x,y)\), velocity \(v=(v_x,v_y)\), goal \(g=(1.5,0)\) with radius \(0.1\).
- The wall lies at \(x=0\) and spans \(y\in[-1.5,1.5]\), except for one open interval
  \[
  (c-w/2,\;c+w/2),
  \]
  where \(c\sim\mathrm{Uniform}[-0.45,0.45]\) is the checkpoint’s gap center and \(w=0.6\) is the default gap width.
- The wall and gap are present in the true environment and absent from every ensemble member. Checkpoint identity includes the full state, gap center, gap width, goal, and noise regime.
- The original action grid, horizon, reward, and collision rules are retained unless explicitly changed below. A collision uses continuous segment intersection, so a trajectory cannot tunnel through a solid wall interval between adjacent states.

The fixed goal is intentional: the randomized gap, rather than additional goal randomness, is the only environmental substitution in this revision.

### 2.2 Required development-only distribution report

Before any confirmatory run, produce a report for **16 development seeds** and **48 crossed checkpoints per regime**. Each checkpoint receives **64 independent closed-loop outcome episodes**.

For each regime, report:

1. The empirical collision-probability histogram and a checkpoint-level table, not only a mean.
2. The fraction with \(\hat p_C=0\), the fraction with \(\hat p_C=1\), and the fraction with \(0<\hat p_C<1\).
3. The fraction in the pre-registered interior band \(0.05<\hat p_C<0.95\).
4. Variance of checkpoint \(\hat p_C\), median and 10th/90th percentiles, and the same quantities after odd/even split-half estimation.
5. The fraction of zero-variance score/target vectors, since each directly affects the Spearman convention below.

Finite rollout estimates of exactly zero or one are reported as **\(\hat p_C=0\) or \(\hat p_C=1\)**, not as proof that the underlying probabilities are exactly zero or one.

**Current result:** pending. No fraction is claimed until this report exists.

### 2.3 Pre-registered variance and target-quality gate

Every primary regime must satisfy all of the following on development data:

- **Variance:** \(\operatorname{Var}(\hat p_C)\geq 0.02\), in probability² units.
- **Interior support:** at least **20%** of checkpoints satisfy \(0.05<\hat p_C<0.95\).
- **Endpoint limit:** no more than **70%** of checkpoints have \(\hat p_C\in\{0,1\}\).
- **Target reliability:** mean odd/even split-half Spearman–Brown reliability is at least **0.75**.

If the default \(w=0.6\) design fails, at most **three** development-only configurations may be evaluated, in this fixed order: \(w\in\{0.6,0.4,0.8\}\). Configuration selection uses only target variance, interior support, endpoint fraction, and reliability—never CSR-minus-baseline effects. All configurations and outcomes remain in the artifact. If all three fail, the study stops rather than tuning until a favorable result appears.

## 3. Free/paid baselines and the complete k sweep

### 3.1 Environment used by held-out return

Held-out return is the mean closed-loop return of the **model-based controller executing in the true, wall- and gap-containing environment \(E\)**. “Held-out” refers to independent checkpoint panels and outcome episodes, not to a wall-free environment. It is a paid end-to-end baseline, not a prediction on the model’s own imagined transitions.

### 3.2 Free versus paid classification

“Free” means no new true-environment interaction is required at the checkpoint being scored. Development labels used to fit a baseline are accounted separately and are not hidden inside the free category.

| Score | Class | Data used at checkpoint evaluation | Environment interaction per checkpoint |
|---|---|---|---|
| Ensemble return variance | **Free** | Eight model members’ predicted returns under shared canonical noise | None; model calls are recorded separately |
| Development-fitted state-feature baseline | **Free at confirmation** | State, gap, regime, and noise features only; parameters are frozen on development seeds | None at confirmation; its development outcome labels are reported as a separate one-time paid cost |
| Held-out wall-environment return | **Paid** | Closed-loop model-controller episodes in \(E\) | **2k** episodes |
| Planner-trajectory one-step error | **Paid, shared** | Transitions from the same held-out return episodes | Shares the **2k** episodes; no additional episodes |
| Closed-loop Candidate-Set Regret | **Paid** | \(k\) model-controller and \(k\) candidate-set-reference episodes in \(E\) | **2k** episodes total |
| PEG, secondary | **Paid** | Environment episodes plus model-predicted returns under explicit noise coupling | **2k** environment episodes; model calls reported separately |
| Independent collision target | **Paid outcome, not a candidate score** | Fresh closed-loop episodes used only for \(\hat p_C\) and split halves | **64** episodes; not reused by any score |

### 3.3 Nested sweep over k

All paid scores are computed for every \(k\in\{4,8,16,32\}\), using nested pre-generated streams so smaller \(k\) is a strict prefix of larger \(k\).

- CSR uses \(k\) episodes from each controller, preserving the total **2k** environment budget.
- Held-out return uses **2k** model-controller episodes.
- PEG uses **2k** environment episodes with corresponding model calls.
- One-step error reuses the held-out return episodes for that exact \(k\).
- Free scores do not vary with \(k\), but are reported alongside every row for comparison.
- The independent collision target remains fixed at 64 episodes during the score sweep so the target does not change with the score budget.

For every paid score and \(k\), report odd/even split-half reliability and Monte Carlo uncertainty. The primary \(k\) is selected on development data as the **smallest** value whose CSR and held-out-return Spearman–Brown reliabilities are both at least **0.80** in every regime and whose power requirement remains within the cap. If several qualify, choose the one with the smallest maximum required \(N\); if still tied, choose the largest minimum reliability. The complete sweep is reported, including the unselected budgets.

## 4. Development-only state-feature baseline

A strong geometry-only baseline is necessary because collision risk may be predictable without inspecting a model’s regret or uncertainty.

### 4.1 Features and fitting

The baseline uses only:

- Normalized position and velocity: \(x,y,v_x,v_y\).
- Distance to the next solid wall interval, signed distance to the gap center, gap width, and goal distance.
- Speed, velocity toward the wall, velocity toward the gap center, and the noise scale.
- Regime indicators and pre-registered interactions: wall distance × wall-directed speed, and gap distance × gap-directed speed.

It excludes CSR, PEG, ensemble variance, model predictions, and all confirmatory outcomes.

The estimator is a **ridge linear-probability model** for \(\hat p_C\):

1. Standardize features using development means and standard deviations only.
2. Select ridge strength from \(\{0.01,0.1,1,10\}\) using grouped cross-validation with development model seed as the held-out group.
3. Refit on all development seeds, leave the intercept unpenalized, and freeze coefficients before confirmation.
4. Clip predictions to \([0,1]\) only when forming the risk score; retain unclipped values in the artifact.

All development fitting labels come from the 64-episode outcome protocol. Confirmatory checkpoints, collision outcomes, and model parameters are never used to fit or tune this baseline.

### 4.2 Closed-form check of the ridge estimator

For a one-feature fixture \((z,y)\in\{(-1,0),(1,1)\}\), ridge strength \(1\), and an unpenalized intercept, the intercept is \(1/2\) and the slope is
\[
\beta=\frac{(-1)(0-1/2)+(1)(1-1/2)}{(-1)^2+1^2+1}=\frac13.
\]
At \(z=1\), the raw prediction is \(1/2+1/3=\mathbf{5/6}\). At \(z=2\), the raw prediction is \(1/2+2/3=7/6\), and the clipped risk score is **1**. This fixture checks the estimator arithmetic and clipping; it is not evidence about actual predictive performance.

## 5. Closed-loop Candidate-Set Regret and candidate coverage

### 5.1 Closed-loop estimand

At every MPC step \(t\), the model controller and candidate-set reference controller:

1. Receive their own current true state, which may differ after earlier actions.
2. Receive candidate action sequences generated by the same checkpoint/time-indexed random stream.
3. Use four selection-noise replicates per candidate, with shared canonical noise between controllers and no reuse of metric/outcome noise.
4. Select a first action using, respectively, ensemble-mean model predictions or true-environment predictions.
5. Execute that action in the true environment, observe the resulting true state, and replan.

For checkpoint \(s\), controllers \(\mathcal C^M\) and \(\mathcal C^E\), horizon \(H\), and discount \(\gamma\),
\[
\operatorname{CSR}_H(s)=J^E_H(s,\mathcal C^E)-J^E_H(s,\mathcal C^M).
\]

The estimator averages paired closed-loop return differences over \(k\) episodes per controller. The controllers share exogenous evaluation-noise identifiers, but after their trajectories diverge, each candidate set is generated relative to its own current state using the same checkpoint/time-indexed action-sequence stream.

**Interpretation limits:** the reference is only the best behavior found by the same candidate process with true-environment knowledge. It is not a global optimum, an upper bound, or an oracle controller. CSR includes model error, candidate starvation, and closed-loop search effects and **may be negative**.

### 5.2 Candidate feasibility report

For every checkpoint, evaluate the fixed set of 512 candidate action sequences under four selection-noise replicates in the true environment. A candidate is called **strictly collision-free** only if all four open-loop rollouts are collision-free.

Report, per regime:

- The fraction of checkpoints in which **zero of 512 candidates** is strictly collision-free.
- The same fraction conditional on a deterministic gap-aware geometric detour being collision-free under zero noise.
- The distribution of strictly collision-free candidate counts, including its median and 10th percentile.

The geometric detour distinguishes an infeasible physical checkpoint from starvation of the random candidate set. It is not inserted into the main candidate set and does not rescue the reference controller. Candidate starvation is part of CSR’s estimand, not silently excluded.

### 5.3 Updated analytic oracle

The v0.1 one-dimensional case remains valid with one change: held-out return is now evaluated in the wall-containing environment. With \(H=1\), no noise, actions \(\{0,1,2\}\), wall at \(0.75\), goal at \(2\), and model/environment returns

| Action | Model return | Environment return |
|---|---:|---:|
| 0 | -0.01 | -0.01 |
| 1 | 0.49 | 0.49 |
| 2 | 0.99 | **-4** |

closed-loop and open-loop selection coincide because there is only one action step. Therefore:

- **CSR:** \(0.49-(-4)=\mathbf{4.49}\).
- **PEG:** \(0.99-(-4)=\mathbf{4.99}\).
- **First-action flip:** **1**.
- **Held-out wall-environment return:** **-4**; the separate wall-free value of 0.99 is not the baseline.
- **Ensemble return variance:** **0** when every member predicts the same deleted-wall return.
- **One-step squared error:** \((1-0.75)^2+(2-0)^2=\mathbf{4.0625}\).
- **Strict candidate coverage:** action 1 is collision-free, so the no-collision-free-candidate fraction is **0** in this fixture.

## 6. Statistical specification

### 6.1 Crossed checkpoint design

- Use **48 fixed checkpoint configurations per regime**, evaluated for every model seed. Checkpoints are not resampled independently for each model.
- All model seeds therefore see the **same crossed checkpoint panel**. Model training data remain nested within model seed; checkpoint identity is crossed with model seed.
- R1 and R2 use the **same physical checkpoint states and gaps**, with noise scale crossed over those checkpoints. R0 uses its separately stratified panel.
- Candidate action streams are indexed by checkpoint and MPC time, not by model seed, so comparisons are not driven by different candidate luck.

### 6.2 Per-seed Spearman statistic and ties

For each model seed \(m\), regime \(r\), and score \(b\):

1. Compute Spearman correlation \(\rho_{m,r,b}\) across the 48 checkpoints using **average/midrank ranks for ties**, including tied collision probabilities at 0 or 1. No jitter is added and no tied checkpoint is dropped.
2. Compute CSR-minus-baseline differences **within the same seed and regime**: \(\Delta_{m,r,b}=\rho_{m,r,\mathrm{CSR}}-\rho_{m,r,b}\).
3. Average correlations and differences over model seeds. **Pooled Spearman correlation across all seeds is not used.**

Mathematical Spearman correlation is undefined when a vector is constant. For the decision rule only, a constant score or target is assigned **zero ranking information, \(\rho=0\)**, and every occurrence is reported. A sensitivity analysis that excludes undefined cases is also reported; it cannot be used to rescue the primary result.

### 6.3 Bootstrap and lower-bound margin test

- Resample model/training seeds with replacement **10,000 times**, keeping the complete crossed checkpoint panel, both R1/R2 observations, and all per-seed score/target values together.
- Recompute per-seed correlations and mean differences within each bootstrap replicate. Do not bootstrap checkpoints independently.
- For each of the four baselines, form a **one-sided 98.75% lower confidence bound** for mean per-seed \(\Delta\), applying the Bonferroni adjustment \(1-0.05/4\) to preserve a 95% familywise bound across baselines.
- The confirmatory margin is **lower bound > 0.10**, in at least two of three regimes. A point estimate above 0.10 with a lower bound at or below 0.10 fails.

### 6.4 Split-half reliability of the collision target

For each checkpoint, split the 64 independent outcome episodes by pre-generated odd/even episode indices into two 32-episode halves.

- Estimate \(\hat p_C^{(A)}\) and \(\hat p_C^{(B)}\) independently.
- For each model seed and regime, compute Pearson correlation \(r\) between the two halves across the 48 checkpoints, then apply Spearman–Brown correction \(R=2r/(1+r)\).
- Average reliability over seeds and report its distribution. A constant half is assigned reliability zero for gate purposes and reported explicitly.
- If mean reliability is below **0.75**, increase the development target budget from 64 to **128** episodes and recompute the gate. The confirmatory target budget is frozen only after this check; it is never chosen using CSR effects.

## 7. Development-only pilot power analysis

### 7.1 Status and purpose

The pilot has **not been run**. It is a required gate before fixing \(N\), not a result available for this review. The former 32-seed allocation and “power unverified” note are replaced by the following calculation rule.

### 7.2 Estimates used

For every regime, baseline, and \(k\), compute from development seeds only:

- Mean per-seed Spearman difference \(\hat\Delta_{r,b,k}\).
- Standard deviation of per-seed differences \(\hat\sigma_{r,b,k}\).
- Reliability, target variance, endpoint fractions, and candidate-starvation rates required above.

To avoid taking the optimistic pilot point estimate at face value, use the pre-registered planning values
\[
\Delta_{\mathrm{plan}}=0.8\hat\Delta,\qquad \sigma_{\mathrm{plan}}=1.2\hat\sigma.
\]

### 7.3 Sample-size rule

For an 80%-power, one-sided 95% margin test with four-baseline Bonferroni allocation, the required number of independent model-seed clusters per regime is
\[
N_{r,b,k}=\left\lceil
\left[\frac{(z_{1-0.05/4}+z_{0.80})\,\sigma_{\mathrm{plan}}}{\Delta_{\mathrm{plan}}-0.10}\right]^2
\right\rceil.
\]
Here \(z_{1-0.05/4}\approx2.241\) and \(z_{0.80}\approx0.842\).

- If \(\Delta_{\mathrm{plan}}\leq0.10\), the required margin cannot be demonstrated with finite \(N\); the thesis stops for that regime rather than increasing seeds indefinitely.
- Choose \(N\) as the **maximum** requirement over all four baselines and all three regimes at the selected \(k\). Do not select only favorable regimes to reduce \(N\).
- The cap is **200 independent confirmatory model seeds**. If any required \(N_{r,b,k}>200\), stop or redesign before confirmation; do not report an underpowered result as success.
- Confirmatory seed identifiers begin at `1000` and do not overlap development seeds. The exact final count remains **pending the pilot**.

The required pilot artifact contains the full \(N_{r,b,k}\) table, planning mean/SD values, and selected \(k\). No empirical power claim is made here.

## 8. Overlap check: arXiv 2609.05834

[Zhang, *Counterfactual Latent World Models* (arXiv:2609.05834)](https://arxiv.org/abs/2609.05834) is directly relevant but does not use the estimand proposed here.

### What it measures

- **Counterfactual separability (CFS):** AUROC for whether latent distance between imagined action-conditioned futures separates pairs whose privileged intervention-outcome labels differ:
  \[
  \mathrm{CFS}=\operatorname{AUROC}\left(d_z(\Phi(h,\alpha),\Phi(h,\alpha')),\mathbf1[y(h,\alpha)\neq y(h,\alpha')]\right).
  \]
- **Task performance:** episode success on Occluded Push and Aliased Maze and normalized return on Deferred Kitchen.
- **Model-exploitation rate:** planning failures associated with high predicted value but low realized value.
- **Evaluation scale in the paper:** five random seeds and 100 fixed test episodes per method. The authors report CFS–success correlations of at least \(r=0.94\) across five baseline model classes and explicitly exclude CLWM’s own metric-aligned point from independent metric validation.
- **Limit stated by the authors:** large-scale pretrained encoders are not evaluated; the CFS relationship is established for models trained from scratch.

### Overlap and remaining distinction

The overlap is substantial at the problem-framing level: counterfactual action consequences, intervention labels, and model exploitation all matter. CFS also supplies an AUROC-based diagnostic rather than relying only on return.

The distinction is equally important:

| CFS / CLWM | Counterworld v0.2 |
|---|---|
| Latent, partially observed representations and imagined open-loop futures | Explicit full state and explicit exogenous noise in a two-dimensional double integrator |
| AUROC for latent-distance separation of intervention-outcome classes | Closed-loop candidate-set regret and its rank relationship with independently estimated collision probability |
| Representation learning and counterfactual contrastive training | No representation learning; structural wall deletion in a linear ensemble |
| Five training seeds and fixed test episodes | Development-gated variance/reliability, nested \(k\) sweep, crossed checkpoints, cluster bootstrap, and development-determined \(N\) |
| Privileged outcome labels for intervention pairs | Independent collision outcomes and explicit canonical-noise coupling |

**Conclusion:** CFS is prior art for decision-relevant counterfactual evaluation and must be cited, but it does not replace CSR, candidate-starvation reporting, matched paid-score budgets, or the explicit-noise coupling analysis. CFS is not added as a v0.2 baseline because v0.2 has no latent representation; adding one would be a new study, not a silent extension.

## 9. Valid paired-noise mapping with estimated model covariance

Raw environment-noise values must **not** be reused as if the estimated model covariance were identical to the environment covariance. Pairing is defined on a shared canonical probability space instead.

### 9.1 Canonical noise representation

For each transition, draw an independent standard-normal vector
\[
z_t\sim\mathcal N(0,I_4),
\]
with components ordered as \((z_x,z_y,z_{v_x},z_{v_y})\). The environment applies
\[
\eta_t^E=L^Ez_t,
\]
where \(L^E\) is diagonal with entries \((\sigma_p,\sigma_p,\sigma_v,\sigma_v)\). Model member \(j\) applies
\[
\eta_t^{M_j}=L^{M_j}z_t,
\]
where \(L^{M_j}\) is a Cholesky factor of its frozen residual covariance estimate.

The same \(z_t\) therefore gives a **deterministic common-random-number coupling**, not identical physical noise when \(L^{M_j}\neq L^E\). Each marginal distribution remains correct: model member \(j\) has covariance \(L^{M_j}(L^{M_j})^\top\), and the environment has covariance \(L^E(L^E)^\top\).

### 9.2 Why the pairing is valid

For PEG, the estimand is the expectation of the coupled difference
\[
\mathbb E_z\left[R^M(z;L^M)-R^E(z;L^E)\right],
\]
not a claim that model and environment experienced the same physical disturbance. This estimand is well-defined when the covariance estimates differ and includes noise-model mismatch as part of model error.

- Model covariance is fitted on development/training data and frozen before evaluation, so evaluation \(z\) is independent of the estimated factor.
- If a covariance matrix is numerically non-positive-definite, eigenvalues are clipped at \(10^{-8}\) before factorization and every occurrence is logged. This is a numerical rule, not a hidden model improvement.
- Ensemble members use the same canonical \(z\) for a paired comparison, while each applies its own \(L^{M_j}\). This preserves member-specific marginals and makes disagreement decomposition reproducible.
- During closed-loop environment execution, both CSR controllers receive environment noise \(L^Ez\). During model-based planning, the model uses its own estimated factor. Selection noise is never reused for metric or outcome evaluation.

Future non-Gaussian mappings are out of scope for v0.2. No claim is made that common random numbers guarantee lower variance for every trajectory or controller.

## 10. Frozen v0.2 environment, model, and protocol scope

### 10.1 Dynamics and rewards

- State \((x,y,v_x,v_y)\), nine actions \((a_x,a_y)\in\{-1,0,1\}^2\), \(\Delta t=1\), horizon \(H=12\), discount \(\gamma=0.99\).
- Free dynamics:
  \[
  p_{t+1}=p_t+v_t+\tfrac12a_t+\eta_{p,t},\qquad
  v_{t+1}=v_t+a_t+\eta_{v,t}.
  \]
- Ordinary reward is \(-0.01+\|p_t-g\|_2-\|p_{t+1}-g\|_2\); goal completion adds \(+1\); collision instead returns \(-4\), clamps to the intersection, zeroes velocity, and terminates.
- Nominal environment noise scales are \(\sigma_p=\sigma_v=0.02\); R2 doubles them to \(0.04\).

### 10.2 Model and data

- Eight linear ridge-dynamics members fitted with NumPy on eight bootstrap episode samples. No member receives wall/gap features or a collision class.
- Residual covariance is estimated on held-out episodes and frozen before evaluation.
- Training uses 64 episodes of at most 64 transitions, split 52/12 by whole episode. Initial states have \(x\in[-2,-1]\), \(y\in[-0.75,0.75]\), zero velocity; collection terminates before crossing \(x=-0.25\).
- Planner: random-shooting MPC, 512 candidates, horizon 12, four selection-noise replicates, first-action execution, and replanning after every true transition.

### 10.3 Regimes

Each regime uses 48 crossed checkpoint configurations per model seed:

1. **R0 nominal:** 16 far \((x\in[-1.5,-0.8])\), 16 middle \((x\in[-0.8,-0.4])\), and 16 near-wall \((x\in[-0.4,-0.1])\); \(y\in[-0.5,0.5]\), \(v_x\in[0,0.5]\), \(v_y\in[-0.25,0.25]\), nominal noise.
2. **R1 near-wall/high-speed:** \(x\in[-0.4,-0.1]\), \(y\in[-0.5,0.5]\), \(v_x\in[0.75,1.5]\), \(v_y\in[-0.25,0.25]\), nominal noise.
3. **R2 noise shift:** identical checkpoint states and gaps as R1 with noise scale \(0.04\).

### 10.4 Remaining exclusions

NumPy and Python’s standard library are the only permitted core dependencies. There is no server, UI, sandbox, Hugging Face adapter, neural model, latent CFS baseline, plugin stability promise, or observational causal-inference claim. All protocols remain `CW-0.2-draft`.

## 11. Provenance and remaining prior art

### 11.1 Portfolio-derived inputs

The following are the only portfolio-derived design inputs, with verbatim text. Nothing is attributed to Himanshu’s experiments beyond these quotations.

| Input | Verbatim source | Use |
|---|---|---|
| Missing constraints and fixed-budget comparison | “The missing-obstacle example is a small version of a broader problem. Better search cannot recover a constraint the model never represents. A useful next experiment would change what the model knows, hold the planning budget fixed, and measure whether the executed behavior improves.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Motivates the deleted constraint and fixed planner budget, not the CSR estimator. |
| Replanning limitation | “That does not make MPC immune to a missing obstacle. If every new plan uses a model that excludes an important constraint, the planner can repeat the same mistake.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Motivates the closed-loop collision target and CSR. |
| Matched protocol and exploitability | “That comparison needs a fixed environment, action space, planning horizon, and data budget. Its outcome is the behavior in the environment. A lower predicted cost alone could mean the planner has become better at exploiting the model’s errors.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Motivates matched budgets and executed outcomes. |
| Reporting requirements | “A useful report contains the hypothesis, implementation, dataset and evaluation splits, matched budgets, seeds, uncertainty in the estimates, and representative failures.” — [The World Models Today](https://new-portfolio-henna-ten.vercel.app/posts/the-world-models-today/) | Defines the artifact requirements. |
| Uncertainty under shift | “A shrinking band was an accurate description of the model’s calculation. It was poor evidence that the calculation was adequate outside the observed region.” — [What uncertainty can tell a planner](https://new-portfolio-henna-ten.vercel.app/posts/what-uncertainty-can-tell-a-planner/) | Prevents interpreting low ensemble variance as correctness. |
| Shared misspecification | “If every ensemble member extrapolates as a line, adding members will not introduce our missing bending hypotheses.” — [What uncertainty can tell a planner](https://new-portfolio-henna-ten.vercel.app/posts/what-uncertainty-can-tell-a-planner/) | Motivates the shared deleted-wall ensemble baseline. |
| Synthetic-evidence limits | “This is a synthetic test with a deliberately chosen extrapolation failure. It reports neither real-robot performance nor a replication of a published benchmark.” — [What uncertainty can tell a planner](https://new-portfolio-henna-ten.vercel.app/posts/what-uncertainty-can-tell-a-planner/) | Pattern for the explicit synthetic-test limitation. |

### 11.2 Prior-art position carried forward

- [Yu et al. (2026)](https://arxiv.org/html/2606.15032v2) already defines XGap and decision-centric world-model evaluation. PEG is not presented as new.
- [World-in-World](https://arxiv.org/html/2510.18135v2) measures closed-loop embodied task success and planning scaling, not candidate-set-conditional regret with explicit noise coupling.
- [Elmachtoub & Grigas, SPO](https://arxiv.org/abs/1710.08005) establishes decision regret; CSR applies that idea to candidate-limited closed-loop control rather than claiming a new regret family.
- [PETS](https://arxiv.org/abs/1805.12114) supplies uncertainty-aware ensemble planning, while [MBPO](https://arxiv.org/abs/1906.08253) studies model usage and short branched rollouts. Neither supplies this exact checkpoint-level matched-budget diagnostic.
- [Deep Ensembles](https://arxiv.org/abs/1612.01474) and [Ovadia et al.](https://arxiv.org/abs/1906.02530) motivate the uncertainty baselines and shifted evaluation but do not replace downstream collision/regret measurements.
- [Agarwal et al. and RLiable](https://proceedings.neurips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) motivate interval reporting. The present cluster bootstrap is adapted to crossed checkpoints and per-seed differences rather than treating checkpoints as independent runs.

This remains a targeted, not systematic, review. Exhaustive unserved-status claims are **unverified**.

### 11.3 Personas, unchanged and still provisional

- **Model-based RL researcher:** currently uses bespoke reset-matched notebooks to diagnose low-one-step-error but high-regret failures.
- **Controls/RL engineer:** currently uses large end-to-end batches and manual failure inspection to assess shifted operating regimes.
- **Open-source maintainer/reviewer:** currently reconciles inconsistent seed/budget reports and lacks analytic fixtures for review.

These are design hypotheses, not verified user research. Interview validation remains pending.

## 12. Review and execution gates

Before implementation review can become confirmatory approval, the following artifacts must exist and be reviewable:

1. Per-regime collision histograms, exact \(\hat p_C\in\{0,1\}\) fractions, variance, and interior support.
2. Split-half target reliability and the selected target budget.
3. The full \(k\in\{4,8,16,32\}\) paid-score sweep and reliability table.
4. Development-fitted state-feature coefficients and grouped-CV provenance.
5. Closed-loop CSR definitions in the artifact and the 512-candidate starvation report.
6. Crossed checkpoint manifest, tie/constant-vector counts, and per-seed statistic code paths reviewed against this specification.
7. Development power tables and the resulting \(N\), or a stop decision.
8. arXiv 2609.05834 overlap statement retained with the report.
9. Canonical-noise mapping and covariance-factor provenance for every model member.

No code has been written for v0.2. Review stops here.
