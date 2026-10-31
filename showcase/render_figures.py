"""Render standalone SVG figures from the immutable evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "showcase" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EVIDENCE = json.loads((ROOT / "showcase" / "assets" / "evidence.json").read_text())

NAVY = "#18213d"
TEAL = "#1aa3a1"
RUST = "#c0652d"
AMBER = "#d98d22"
MUTED = "#696d74"
LINE = "#cec0ad"
PAPER = "#f6eddd"
PALE = "#e7ddce"
FONT = "Barlow, 'DejaVu Sans', sans-serif"
MONO = "'JetBrains Mono', 'DejaVu Sans Mono', monospace"


def document(title, body, width=1000, height=520, description=""):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title><desc id="desc">{description}</desc>
  <rect width="100%" height="100%" fill="{PAPER}"/>
  <style>
    text {{ font-family: {FONT}; fill: {NAVY}; }}
    .title {{ font-size: 26px; font-weight: 700; }}
    .subtitle {{ font-size: 14px; fill: {MUTED}; }}
    .label {{ font-size: 12px; }}
    .small {{ font-size: 10px; fill: {MUTED}; }}
    .mono {{ font-family: {MONO}; }}
    .axis {{ stroke: {LINE}; stroke-width: 1; }}
    .grid {{ stroke: {LINE}; stroke-width: 1; stroke-dasharray: 3 6; opacity: .65; }}
  </style>
  {body}
</svg>'''


def text(x, y, value, cls="label", anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{value}</text>'


def rect(x, y, width, height, fill, extra=""):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" {extra}/>'


def line(x1, y1, x2, y2, cls="axis", extra=""):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" {extra}/>'


def path(d, color, width=3, extra=""):
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'


def histogram(regime):
    rows = []
    for seed in range(16):
        path = ROOT / "artifacts" / "development-targets" / f"targets_seed_{seed}.jsonl"
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    rows = [row for row in rows if row["regime"] == regime]
    bins = [0.0] * 11
    for row in rows:
        bins[min(10, int(row["collision_probability"] * 10))] += 1
    return bins


def histogram_figure():
    original = histogram("R0")
    amended_rows = []
    for seed in range(16):
        path = ROOT / "artifacts" / "development-targets-r0-broadened" / f"targets_seed_{seed}.jsonl"
        amended_rows.extend(json.loads(line) for line in path.read_text().splitlines())
    amended = [0.0] * 11
    for row in amended_rows:
        if row["regime"] == "R0":
            amended[min(10, int(row["collision_probability"] * 10))] += 1
    maximum = max(original + amended)
    body = text(48, 48, "The original target could not rank risk", "title") + text(48, 75, "R0: all 768 checkpoints had collision probability zero", "subtitle")
    for index, (before, after) in enumerate(zip(original, amended)):
        x = 58 + index * 85
        before_height = 0 if maximum == 0 else before / maximum * 230
        after_height = after / maximum * 230
        body += rect(x, 340 - before_height, 32, before_height, NAVY, 'rx="4"')
        body += rect(x + 36, 340 - after_height, 32, after_height, TEAL, 'rx="4"')
        body += text(x + 34, 365, f"{index / 10:.1f}", "small", "middle")
    body += rect(58, 405, 16, 16, NAVY) + text(82, 418, "Original R0", "label")
    body += rect(190, 405, 16, 16, TEAL) + text(214, 418, "Amended R0", "label")
    body += text(48, 470, "A useful target needs safe, risky, and uncertain checkpoints.", "subtitle")
    return document("Original and amended collision distributions", body, description="Histograms of checkpoint collision probabilities for original and amended R0")


def reliability_figure():
    colors = {"csr": RUST, "heldout_return": NAVY, "peg": TEAL}
    labels = {"csr": "CSR", "heldout_return": "Held-out return", "peg": "PEG"}
    body = text(48, 45, "Reliability improves with evaluation budget", "title") + text(48, 70, "Split-half Spearman–Brown reliability by regime and rollout budget k", "subtitle")
    for panel, regime in enumerate(["R0", "R1", "R2"]):
        origin_x = 55 + panel * 315
        origin_y = 120
        width, height = 250, 275
        body += text(origin_x, 105, regime, "title")
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = origin_y + height - tick * height
            body += line(origin_x, y, origin_x + width, y, "grid") + text(origin_x - 10, y + 4, f"{tick:.2f}", "small", "end")
        body += line(origin_x, origin_y + height - 0.8 * height, origin_x + width, origin_y + height - 0.8 * height, "axis", 'stroke-dasharray="6 5"')
        body += text(origin_x + width - 4, origin_y + height - 0.8 * height - 7, "0.80 gate", "small", "end")
        for metric, color in colors.items():
            points = []
            for k in [4, 8, 16, 32]:
                value = EVIDENCE["sweep"][f"{regime}/k={k}"][f"{metric}_reliability"]["mean"]
                x = origin_x + (k / 32) * width
                y = origin_y + height - value * height
                points.append((x, y, k, value))
            d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y, _, _) in enumerate(points))
            body += path(d, color, 3)
            for x, y, k, value in points:
                body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
                if k == 32:
                    body += text(x - 2, y - 10, f"{value:.2f}", "small", "end")
        for k in [4, 8, 16, 32]:
            x = origin_x + (k / 32) * width
            body += text(x, origin_y + height + 25, str(k), "small", "middle")
        body += text(origin_x + width / 2, origin_y + height + 48, "k", "small", "middle")
    for index, (metric, color) in enumerate(colors.items()):
        x = 85 + index * 180
        body += line(x, 493, x + 28, 493, "axis", f'stroke="{color}" stroke-width="3"') + text(x + 37, 497, labels[metric], "label")
    return document("Score reliability by rollout budget", body, description="CSR, held-out return, and PEG reliability for k 4, 8, 16, and 32 in R0, R1, and R2")


METRICS = [
    ("ensemble_return_variance", "Ensemble variance", NAVY),
    ("heldout_return", "Held-out return", TEAL),
    ("one_step_error", "One-step error", AMBER),
    ("state_feature_baseline", "State-feature baseline", "#3f7f8e"),
    ("csr", "CSR", RUST),
]


def spearman_figure(regime):
    data = EVIDENCE["power"]["regimes"][regime]
    body = text(48, 45, f"{regime}: risk-ranking comparison", "title") + text(48, 70, "Mean per-seed Spearman correlation with independent collision probability · higher is better", "subtitle")
    origin_y, height = 105, 300
    for tick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        y = origin_y + (1.0 - tick) / 2 * height
        body += line(65, y, 955, y, "grid") + text(54, y + 4, f"{tick:.1f}", "small", "end")
    zero_y = origin_y + height / 2
    body += line(65, zero_y, 955, zero_y, "axis")
    for index, (key, label, color) in enumerate(METRICS):
        value = data["heldout_return"]["csr_mean_spearman"] if key == "csr" else data[key]["baseline_mean_spearman"]
        x = 85 + index * 175
        bar_height = abs(value) * height / 2
        y = zero_y - bar_height if value >= 0 else zero_y
        body += rect(x, y, 105, bar_height, color, 'rx="5"')
        body += text(x + 52, y - 9 if value >= 0 else y + bar_height + 18, f"{value:.3f}", "label", "middle")
        words = label.split(" ")
        body += text(x + 52, 440, " ".join(words[:-1]) if len(words) > 1 else label, "small", "middle")
        body += text(x + 52, 457, words[-1], "small", "middle")
    return document(f"{regime} metric comparison", body, description=f"Spearman correlations for five metrics in {regime}")


def required_figure():
    body = text(48, 45, "The kill criterion is explicit", "title") + text(48, 72, "Required model seeds by regime and baseline · cap 200 · ∞ means no finite margin", "subtitle")
    for panel, regime in enumerate(["R0", "R1", "R2"]):
        data = EVIDENCE["power"]["regimes"][regime]
        y = 112 + panel * 125
        body += text(60, y, regime, "title")
        body += line(60, y + 13, 945, y + 13, "axis")
        for index, (key, label, _) in enumerate(METRICS[:-1]):
            value = data[key]["required_n"]
            x = 70 + index * 222
            body += text(x, y + 45, label, "small")
            color = RUST if value == "Infinity" else TEAL
            body += text(x, y + 84, str(value), "title", extra=f'fill="{color}"')
            body += text(x, y + 105, "required N", "small")
    body += text(48, 500, "Overall required N: ∞ · the thesis cannot proceed to confirmation", "subtitle")
    return document("Required sample sizes", body, description="Required model seed counts for ensemble variance, held-out return, one-step error, and state-feature baseline in all three regimes")


def wall_figure():
    body = text(48, 44, "The map is missing the wall", "title") + text(48, 70, "A missing constraint changes the decision, not just the prediction error", "subtitle")
    body += rect(90, 135, 300, 220, "#efe3d1", 'rx="16"') + text(115, 165, "MODEL'S MAP", "label")
    body += line(225, 195, 225, 310, "axis", 'stroke-dasharray="7 7"')
    body += text(250, 250, "wall deleted", "small")
    body += f'<circle cx="155" cy="275" r="12" fill="{TEAL}"/>' + text(115, 330, "agent", "small")
    body += f'<circle cx="340" cy="205" r="12" fill="{AMBER}"/>' + text(300, 185, "goal", "small")
    body += path("M155,275 Q250,225 340,205", RUST, 4, 'stroke-dasharray="7 6"')
    body += rect(490, 135, 300, 220, "#efe3d1", 'rx="16"') + text(515, 165, "TRUE ENVIRONMENT", "label")
    body += line(625, 195, 625, 240, "axis") + line(625, 280, 625, 310, "axis")
    body += text(648, 265, "gap", "small")
    body += f'<circle cx="555" cy="275" r="12" fill="{TEAL}"/>' + text(515, 330, "agent", "small")
    body += f'<circle cx="740" cy="205" r="12" fill="{AMBER}"/>' + text(700, 185, "goal", "small")
    body += path("M555,275 Q640,230 625,255", RUST, 4) + f'<circle cx="625" cy="255" r="10" fill="{RUST}"/>' + text(645, 300, "collision", "small")
    body += path("M130,420 Q430,360 790,420", NAVY, 3, 'stroke-dasharray="4 8"')
    body += text(48, 475, "Same action. Different world. The regret metric must prove it adds useful warning information.", "subtitle")
    return document("Missing wall model versus true environment", body, description="Diagram of a wall-deleted model and the true wall with a gap")


def trajectory_figure():
    records = EVIDENCE["trajectories"]
    all_states = [state for record in records.values() for state in record["states"]]
    xs = [state[0] for state in all_states] + [1.5]
    ys = [state[1] for state in all_states] + [0.0]
    xmin, xmax = min(xs) - 1, max(xs) + 1
    ymin, ymax = min(ys) - 1, max(ys) + 1

    def point(state):
        x = 100 + (state[0] - xmin) / (xmax - xmin) * 800
        y = 390 - (state[1] - ymin) / (ymax - ymin) * 250
        return round(x, 2), round(y, 2)

    body = text(48, 44, "A real trajectory, not a stylized success", "title") + text(48, 70, f"Checkpoint {EVIDENCE['selection']['checkpoint_id']} · actual model execution in the true environment", "subtitle")
    body += line(120, 400, 900, 400, "grid")
    for name, color in [("model_in_imagination", RUST), ("model_in_environment", TEAL), ("reference_in_environment", NAVY)]:
        states = records[name]["states"]
        points = [point(state) for state in states]
        d = " ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(points))
        body += path(d, color, 3)
        body += f'<circle cx="{points[0][0]}" cy="{points[0][1]}" r="7" fill="{color}"/>'
        body += f'<circle cx="{points[-1][0]}" cy="{points[-1][1]}" r="7" fill="{color}"/>'
    body += text(110, 450, "start", "small")
    body += rect(720, 405, 16, 16, RUST) + text(744, 418, "Imagined", "label")
    body += rect(835, 405, 16, 16, TEAL) + text(859, 418, "Actual", "label")
    body += rect(720, 445, 16, 16, NAVY) + text(744, 458, "Reference", "label")
    return document("Actual and imagined trajectories", body, description="Actual model, imagined model, and reference trajectories for the selected checkpoint")


for name, figure in [
    ("collision-distribution.svg", histogram_figure()),
    ("reliability.svg", reliability_figure()),
    ("required-n.svg", required_figure()),
    ("missing-wall.svg", wall_figure()),
    ("trajectory.svg", trajectory_figure()),
]:
    (OUT / name).write_text(figure)
for regime in ["R0", "R1", "R2"]:
    (OUT / f"spearman-{regime}.svg").write_text(spearman_figure(regime))
print(f"Wrote {len(list(OUT.glob('*.svg')))} SVG figures")
