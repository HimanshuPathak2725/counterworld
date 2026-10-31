# Counterworld visual research report

**When the map is wrong: why the better warning sign lost.**

This is a development-only, evidence-backed presentation of a **negative research result**. It explains why Candidate-Set Regret did not outperform the required collision-risk baselines in the deleted-wall double-integrator testbed.

## Deliverables

- `index.html` — interactive visual report with regime selection, trajectory animation, source hashes, and a video dialog
- `report.pdf` — eight-page printable report
- `figures/*.svg` — standalone collision, reliability, metric, wall, and trajectory diagrams
- `assets/evidence.json` — immutable evidence bundle with artifact hashes and actual trajectory records
- `assets/app.js` — interactive behavior; no framework or external runtime dependency
- `video/counterworld-explainer.mp4` — 60-second H.264 explainer at 1280×720
- `video/captions.vtt` — WebVTT captions
- `video/transcript.md` — plain-text transcript
- `video/poster.png` — poster frame
- `build_evidence.py`, `render_figures.py`, `render_video.py` — deterministic generators

The video is silent by design and includes on-screen text and captions.

## Public access

The complete presentation is available publicly at [HimanshuPathak2725/counterworld](https://github.com/HimanshuPathak2725/counterworld). Anyone with the link can review the thesis result without an interactive deployment:

- Open `report.pdf` for the printable report.
- Open `video/counterworld-explainer.mp4` for the explainer video.
- Open `figures/` for the diagrams.
- Open `assets/evidence.json` for the evidence bundle and actual trajectories.
- Open `video/captions.vtt` or `video/transcript.md` for accessible text alternatives.

## Open the interactive report locally

The optional interactive report can be served from this directory:

```bash
python -m http.server 4180
```

Then open <http://localhost:4180>.

For a PDF, open <http://localhost:4180/report.pdf>.

## Rebuild from artifacts

Run from the `research/` directory:

```bash
PYTHONPATH=src python showcase/build_evidence.py
python showcase/render_figures.py
python showcase/render_video.py
```

The evidence bundle is generated from the original and amended target summaries, the nested paid-score sweep, the final power analysis, and a deterministic representative trajectory. The checkpoint selection rule is included in the evidence bundle.

## Verification completed

- 51 core project tests pass
- Regime tabs show the correct CSR and baseline values
- All eight SVG figures load successfully
- The video dialog reports a 60-second duration
- Trajectory controls and source-hash list work
- No browser console errors
- Desktop and mobile layouts have no horizontal overflow
- PDF renders as eight pages with searchable verdict text
- All nine video scenes pass contact-sheet review

Generated frame sequences in `video/frames/` and temporary screenshots are excluded from version control.

## Interpretation

The presentation does not claim CSR is universally useless. It shows that, in this testbed, one-step error and the development-fitted state-feature baseline were much stronger collision-risk predictors. The pre-registered kill rule was triggered, so no confirmatory experiment was run.
