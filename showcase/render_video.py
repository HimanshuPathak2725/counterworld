"""Render a captioned 60-second Counterworld explainer video from evidence."""

from __future__ import annotations

import json
import math
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "showcase" / "video"
FRAMES = OUT / "frames"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(exist_ok=True)
EVIDENCE = json.loads((ROOT / "showcase" / "assets" / "evidence.json").read_text())
W, H = 1280, 720
FPS = 8
NAVY = "#18213d"
TEAL = "#1aa3a1"
RUST = "#c0652d"
AMBER = "#d98d22"
MUTED = "#696d74"
PAPER = "#f6eddd"
CARD = "#f9f2e5"
LINE = "#cec0ad"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size, bold=False, mono=False):
    return ImageFont.truetype(BOLD if bold else MONO if mono else FONT, size)


def text(draw, xy, value, size=24, fill=NAVY, bold=False, mono=False, anchor=None):
    draw.text(xy, value, font=font(size, bold, mono), fill=fill, anchor=anchor)


def wrap(draw, xy, value, width, size=24, fill=NAVY, bold=False, mono=False, spacing=8, anchor=None):
    lines = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
    draw.multiline_text(xy, "\n".join(lines), font=font(size, bold, mono), fill=fill, spacing=spacing, anchor=anchor)


def base(kicker, caption=""):
    image = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((32, 24, 210, 62), radius=8, fill="#e9dcca")
    text(draw, (48, 34), "COUNTERWORLD", 18, NAVY, mono=True)
    if kicker:
        text(draw, (48, 90), kicker.upper(), 19, MUTED, mono=True)
    if caption:
        draw.rounded_rectangle((28, H - 84, W - 28, H - 24), radius=12, fill=NAVY)
        wrap(draw, (52, H - 70), caption, 90, 22, "#fdf8ef", mono=False)
    return image, draw


def card(draw, box, fill=CARD, outline=LINE, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def histogram(draw, values, box, color, title):
    x0, y0, x1, y1 = box
    card(draw, box)
    text(draw, (x0 + 24, y0 + 18), title, 22, NAVY, bold=True)
    maximum = max(values) or 1
    usable_height = y1 - y0 - 110
    for index, value in enumerate(values):
        x = x0 + 35 + index * 72
        height = value / maximum * usable_height
        draw.rounded_rectangle((x, y1 - 42 - height, x + 42, y1 - 42), radius=4, fill=color)
        text(draw, (x + 21, y1 - 28), f"{index / 10:.1f}", 13, MUTED, anchor="mm")


def line_chart(draw, series, box, title, threshold=None):
    x0, y0, x1, y1 = box
    card(draw, box)
    text(draw, (x0 + 24, y0 + 18), title, 22, NAVY, bold=True)
    left, top, width, height = x0 + 60, y0 + 62, x1 - x0 - 90, y1 - y0 - 120
    for tick in [0, .25, .5, .75, 1]:
        y = top + (1 - tick) * height
        draw.line((left, y, left + width, y), fill=LINE, width=1)
        text(draw, (left - 12, y), f"{tick:.2f}", 13, MUTED, anchor="rm")
    if threshold is not None:
        y = top + (1 - threshold) * height
        draw.line((left, y, left + width, y), fill=RUST, width=2)
        text(draw, (left + width - 4, y - 8), f"{threshold:.2f}", 13, RUST, anchor="rs")
    for label, color, points in series:
        for index, (x_value, y_value) in enumerate(points):
            x = left + (x_value / 32) * width
            y = top + (1 - y_value) * height
            if index == 0:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            if index + 1 < len(points):
                next_x, next_y = points[index + 1]
                next_x = left + (next_x / 32) * width
                next_y = top + (1 - next_y) * height
                draw.line((x, y, next_x, next_y), fill=color, width=4)
    for k in [4, 8, 16, 32]:
        x = left + (k / 32) * width
        text(draw, (x, top + height + 24), str(k), 14, MUTED, anchor="mm")


def spearman(draw, regime, box):
    data = EVIDENCE["power"]["regimes"][regime]
    x0, y0, x1, y1 = box
    card(draw, box)
    text(draw, (x0 + 24, y0 + 18), f"{regime} · risk-ranking scores", 24, NAVY, bold=True)
    metrics = [
        ("ensemble_return_variance", "Ensemble", NAVY),
        ("heldout_return", "Held-out", TEAL),
        ("one_step_error", "One-step", AMBER),
        ("state_feature_baseline", "Geometry", "#3f7f8e"),
        ("csr", "CSR", RUST),
    ]
    left, top, width, height = x0 + 58, y0 + 78, x1 - x0 - 86, y1 - y0 - 145
    zero_y = top + height / 2
    draw.line((left, zero_y, left + width, zero_y), fill=NAVY, width=1)
    for tick in [-1, -.5, 0, .5, 1]:
        y = top + (1 - tick) / 2 * height
        draw.line((left, y, left + width, y), fill=LINE, width=1)
        text(draw, (left - 10, y), f"{tick:.1f}", 13, MUTED, anchor="rm")
    for index, (key, label, color) in enumerate(metrics):
        value = data["heldout_return"]["csr_mean_spearman"] if key == "csr" else data[key]["baseline_mean_spearman"]
        x = left + index * (width / 5) + 16
        bar_height = abs(value) * height / 2
        y = zero_y - bar_height if value >= 0 else zero_y
        draw.rounded_rectangle((x, y, x + 70, y + bar_height), radius=6, fill=color)
        text(draw, (x + 35, y - 8 if value >= 0 else y + bar_height + 20), f"{value:.3f}", 16, color, bold=True, anchor="mm")
        text(draw, (x + 35, top + height + 30), label, 15, MUTED, anchor="mm")


def wall_diagram(draw, box):
    x0, y0, x1, y1 = box
    card(draw, box)
    midpoint = (x0 + x1) // 2
    text(draw, (x0 + 28, y0 + 20), "MODEL'S MAP", 22, NAVY, bold=True)
    text(draw, (midpoint + 28, y0 + 20), "TRUE ENVIRONMENT", 22, NAVY, bold=True)
    wall_x = midpoint - 90
    draw.line((wall_x, y0 + 105, wall_x, y1 - 55), fill=RUST, width=5)
    text(draw, (wall_x + 18, y0 + 120), "deleted", 17, RUST)
    true_x = midpoint + 150
    gap_top, gap_bottom = y0 + 155, y0 + 215
    draw.line((true_x, y0 + 70, true_x, gap_top), fill=NAVY, width=10)
    draw.line((true_x, gap_bottom, true_x, y1 - 50), fill=NAVY, width=10)
    draw.line((true_x, gap_top, true_x, gap_bottom), fill=AMBER, width=5)
    text(draw, (true_x + 16, (gap_top + gap_bottom) // 2), "gap", 17, AMBER)
    for x, color in [(x0 + 100, TEAL), (x0 + 280, AMBER), (midpoint + 70, TEAL), (midpoint + 290, AMBER)]:
        draw.ellipse((x - 14, y0 + 100, x + 14, y0 + 128), fill=color)
    draw.arc((x0 + 90, y0 + 110, x0 + 300, y0 + 290), 190, 345, fill=RUST, width=5)
    draw.ellipse((true_x - 12, y0 + 120, true_x + 12, y0 + 144), fill=RUST)
    text(draw, (true_x + 20, y0 + 140), "collision", 17, RUST)


def trajectory(draw, box):
    records = EVIDENCE["trajectories"]
    x0, y0, x1, y1 = box
    card(draw, box)
    states = [state for record in records.values() for state in record["states"]]
    xs = [state[0] for state in states] + [1.5]
    ys = [state[1] for state in states] + [0.0]
    xmin, xmax = min(xs) - 1, max(xs) + 1
    ymin, ymax = min(ys) - .6, max(ys) + .6

    def point(state):
        x = x0 + 45 + (state[0] - xmin) / (xmax - xmin) * (x1 - x0 - 90)
        y = y0 + 65 + (1 - (state[1] - ymin) / (ymax - ymin)) * (y1 - y0 - 110)
        return x, y

    wall_x = point([0, 0])[0]
    draw.line((wall_x, y0 + 50, wall_x, y1 - 45), fill=NAVY, width=8)
    for name, color in [("model_in_imagination", RUST), ("model_in_environment", TEAL), ("reference_in_environment", NAVY)]:
        points = [point(state) for state in records[name]["states"]]
        for index, (x, y) in enumerate(points):
            if index + 1 < len(points):
                draw.line((x, y, *points[index + 1]), fill=color, width=4)
        for x, y in [points[0], points[-1]]:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    goal = point(EVIDENCE["selection"]["goal"])
    draw.ellipse((goal[0] - 13, goal[1] - 13, goal[0] + 13, goal[1] + 13), fill=AMBER)
    text(draw, (goal[0] + 18, goal[1]), "goal", 16, MUTED)


def scene_text(draw, title, subtitle=None, x=58, y=118, width=50, size=50):
    wrap(draw, (x, y), title, width, size, NAVY, bold=True, spacing=10)
    if subtitle:
        wrap(draw, (x, y + size * 1.6), subtitle, width, 25, MUTED, spacing=7)


SCENES = [
    (4, "title", "When the map is wrong", "A negative RL result in 60 seconds"),
    (7, "wall", "The model deletes the wall", "The environment keeps it. Now every plan is suspect."),
    (7, "thesis", "Can regret predict danger better?", "We test Candidate-Set Regret against four simpler warning signs."),
    (8, "broken", "The original experiment had no variance", "All 768 R0 checkpoints were safe. A metric cannot rank risk when nothing differs."),
    (8, "amended", "Random goals and broader speeds fixed the target", "Now there are safe, risky, and uncertain checkpoints. All gates pass."),
    (7, "reliability", "The smallest reliable budget is k = 16", "CSR and held-out return clear the 0.80 reliability gate in every regime."),
    (10, "tournament", "The smarter-looking metric lost", "One-step error and geometry predict collision risk much better than CSR."),
    (6, "verdict", "Required N = infinity", "The pre-registered kill rule is triggered. No confirmatory experiment will run."),
    (3, "closing", "A useful negative", "A rejected thesis is still evidence. The artifacts and analysis are preserved."),
]


def main():
    frame_number = 0
    for scene_index, (seconds, kind, title, subtitle) in enumerate(SCENES):
        for _ in range(seconds * FPS):
            kicker = f"{scene_index + 1:02d} / {len(SCENES)}"
            image, draw = base(kicker, subtitle)
            if kind == "title":
                scene_text(draw, title, None, y=250, width=52, size=60)
            elif kind == "wall":
                wall_diagram(draw, (65, 150, 1215, 560))
            elif kind == "thesis":
                scene_text(draw, title, None, y=205, width=48, size=50)
                draw.rounded_rectangle((64, 360, 690, 405), radius=8, fill="#e9dcca")
                text(draw, (82, 372), "CSR = model-planned return − reference-planned return", 20, NAVY)
            elif kind == "broken":
                histogram(draw, [0] * 11, (80, 150, 1200, 550), NAVY, "ORIGINAL R0 · collision probability = 0 everywhere")
                text(draw, (86, 570), "768 checkpoints · 0 collisions · 0 variance", 23, RUST, bold=True)
            elif kind == "amended":
                values = [0.48, 0.10, 0.07, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04, 0.03, 0.03]
                histogram(draw, values, (80, 150, 1200, 550), TEAL, "AMENDED R0 · safe, risky, and uncertain checkpoints")
                text(draw, (86, 570), "Variance 0.163 · interior 34.2% · endpoints 58.3% · reliability 0.9924", 21, TEAL, bold=True)
            elif kind == "reliability":
                for metric, color, name in [("csr", RUST, "CSR"), ("heldout_return", NAVY, "Held-out return"), ("peg", TEAL, "PEG")]:
                    points = [(k, EVIDENCE["sweep"][f"R0/k={k}"][f"{metric}_reliability"]["mean"]) for k in [4, 8, 16, 32]]
                    line_chart(draw, [(name, color, points)], (70, 145, 1210, 555), "R0 · split-half reliability vs budget k", threshold=.8)
            elif kind == "tournament":
                regime = ["R0", "R1", "R2"][(frame_number // FPS) % 3]
                spearman(draw, regime, (65, 145, 1215, 555))
            elif kind == "verdict":
                scene_text(draw, title, None, y=210, width=50, size=55)
                draw.rounded_rectangle((62, 350, 690, 406), radius=8, fill="#e9dcca")
                text(draw, (82, 366), "No confirmatory experiment will run.", 24, RUST, bold=True)
            elif kind == "closing":
                scene_text(draw, title, None, y=235, width=52, size=56)
            if kind not in {"title", "thesis", "verdict", "closing"}:
                pass
            image.save(FRAMES / f"frame_{frame_number:04d}.png")
            frame_number += 1
            if frame_number == 1:
                image.save(OUT / "poster.png")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(FRAMES / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(OUT / "counterworld-explainer.mp4"),
    ], check=True, capture_output=True)
    captions = ["WEBVTT\n"]
    transcript = ["# Counterworld explainer transcript\n"]
    start = 0.0
    for index, (seconds, kind, title, subtitle) in enumerate(SCENES, 1):
        end = start + seconds
        captions.append(f"{index}\n{format_time(start)} --> {format_time(end)}\n{title}\n")
        transcript.append(f"## {format_time(start)}–{format_time(end)}\n\n**{title}** {subtitle}\n")
        start = end
    (OUT / "captions.vtt").write_text("\n".join(captions))
    (OUT / "transcript.md").write_text("\n".join(transcript) + "\n")
    print(f"Rendered {frame_number} frames to {OUT / 'counterworld-explainer.mp4'}")


def format_time(value):
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


if __name__ == "__main__":
    main()
