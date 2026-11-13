"""Draw the README's per-class IoU chart from a release's recorded test reports.

Standard library only. The numbers come from the committed JSON records, never
from hand-typed values:

    python3 scripts/results_chart.py docs/results/beltwatch-0.1.0/test docs/assets/test-iou.svg
"""

import argparse
import json
from pathlib import Path

CLASSES = [
    ("cardboard", "Cardboard"),
    ("soft_plastic", "Soft plastic"),
    ("rigid_plastic", "Rigid plastic"),
    ("metal", "Metal"),
]
SERIES = [
    ("random-forest.json", "Random forest (tuned)", "rf"),
    ("release.json", "SegFormer-B0 (release)", "nn"),
]
TARGET = 0.40  # proposed planning value for foreground macro IoU, not a result

W, H = 760, 300
LEFT, RIGHT, TOP, BOTTOM = 48, 16, 44, 46


def _values(report: dict) -> list[float]:
    seg = report["segmentation"]
    return [seg["iou"][key] for key, _ in CLASSES] + [seg["foreground_macro_iou"]]


def render(test_dir: Path) -> str:
    reports = [json.loads((test_dir / name).read_text()) for name, _, _ in SERIES]
    n_images = {r["n_images"] for r in reports}
    if len(n_images) != 1 or any(r["split"] != "test" for r in reports):
        raise SystemExit("reports must all be on the same test images")
    data = [_values(r) for r in reports]
    labels = [label for _, label in CLASSES] + ["Foreground macro"]

    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    y_max = 0.6

    def y(v: float) -> float:
        return TOP + plot_h * (1 - v / y_max)

    group_w = plot_w / len(labels)
    bar_w = group_w * 0.3
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" '
        'role="img" aria-label="Per-class test IoU, random forest versus release">',
        "<style>",
        "text{fill:#1f2328;font-size:12px}.muted{fill:#59636e}.grid{stroke:#d1d9e0}",
        ".rf{fill:#8c959f}.nn{fill:#1f6f5c}.target{stroke:#bc4c00;stroke-dasharray:5 4}",
        ".tl{fill:#bc4c00}",
        "@media (prefers-color-scheme:dark){text{fill:#f0f6fc}.muted{fill:#9198a1}",
        ".grid{stroke:#3d444d}.rf{fill:#656c76}.nn{fill:#43b597}.target{stroke:#f0883e}",
        ".tl{fill:#f0883e}}",
        "</style>",
        f'<text x="{LEFT}" y="18" font-weight="600" font-size="13">'
        f"IoU on the locked test split ({n_images.pop()} images)</text>",
    ]
    for i, (_, name, cls) in enumerate(SERIES):
        x = W - RIGHT - 360 + i * 190
        out.append(f'<rect class="{cls}" x="{x}" y="8" width="12" height="12" rx="2"/>')
        out.append(f'<text x="{x + 18}" y="18">{name}</text>')
    for tick in (0.0, 0.2, 0.4, 0.6):
        ty = y(tick)
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{ty:.1f}" y2="{ty:.1f}"/>')
        out.append(
            f'<text class="muted" x="{LEFT - 8}" y="{ty + 4:.1f}" text-anchor="end">'
            f"{tick:.1f}</text>"
        )
    for g, label in enumerate(labels):
        gx = LEFT + g * group_w + group_w / 2
        for s, (_, _, cls) in enumerate(SERIES):
            v = data[s][g]
            bx = gx - bar_w - 2 + s * (bar_w + 4)
            out.append(
                f'<rect class="{cls}" x="{bx:.1f}" y="{y(v):.1f}" width="{bar_w:.1f}" '
                f'height="{y(0) - y(v):.1f}" rx="2"/>'
            )
            out.append(
                f'<text x="{bx + bar_w / 2:.1f}" y="{y(v) - 5:.1f}" text-anchor="middle" '
                f'font-size="11">{v:.2f}</text>'
            )
        weight = ' font-weight="600"' if g == len(labels) - 1 else ""
        out.append(
            f'<text x="{gx:.1f}" y="{H - BOTTOM + 18}" text-anchor="middle"{weight}>{label}</text>'
        )
    # proposed target applies to the foreground macro IoU only
    gx0 = LEFT + (len(labels) - 1) * group_w + 6
    ty = y(TARGET)
    out.append(
        f'<line class="target" x1="{gx0:.1f}" x2="{W - RIGHT - 6:.1f}" '
        f'y1="{ty:.1f}" y2="{ty:.1f}"/>'
    )
    out.append(
        f'<text class="tl" x="{W - RIGHT - 6:.1f}" y="{ty - 6:.1f}" text-anchor="end" '
        'font-size="11">proposed target 0.40</text>'
    )
    out.append(
        f'<text class="muted" x="{LEFT}" y="{H - 8}" font-size="11">Source: '
        f"docs/results/{test_dir.parent.name}/test/ · scored once, "
        "after the release was declared on val</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("test_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.test_dir))


if __name__ == "__main__":
    main()
