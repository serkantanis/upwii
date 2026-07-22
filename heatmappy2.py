"""CSV gaze points -> image heatmaps using heatmappy2.

For every stimulus, this script reads ``<name>.csv`` and overlays its gaze
points on the matching image (jpg/jpeg/png/bmp/webp). The result is written as
``<name>_heatmap.png``.
"""

from __future__ import annotations

import re
from pathlib import Path

# Google Colab / Google Drive folders.
# Plain strings are allowed; they are converted to Path objects below.
CSV_DIR = "/content/drive/MyDrive/convertedeye"
IMAGE_DIR = "/content/drive/MyDrive/convertedeye"
OUTPUT_DIR = "/content/drive/MyDrive/surveyneuro/image_heats"

CSV_DIR = Path(CSV_DIR)
IMAGE_DIR = Path(IMAGE_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)

# None: process every CSV that has a matching image.
# Or use a list such as ["1", "2", "10", "18"].
STIMULI_LIST: list[str] | None = None

# Set this to "yx" only when each CSV row stores y first, then x.
CSV_COORDINATE_ORDER = "xy"

POINT_DIAMETER = 70
POINT_STRENGTH = 0.8
OPACITY = 0.65
NORMALISATION = "raw"
MODE = "colour"  # "colour", "reveal" or "pair"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image_points(csv_path: Path) -> list[tuple[float, float]]:
    """Load comma, semicolon, tab or whitespace-delimited x/y rows."""
    if CSV_COORDINATE_ORDER not in {"xy", "yx"}:
        raise ValueError('CSV_COORDINATE_ORDER must be either "xy" or "yx".')

    points: list[tuple[float, float]] = []
    with csv_path.open("r", encoding="utf-8-sig") as csv_file:
        for line in csv_file:
            values = re.split(r"[,;\s]+", line.split("#", 1)[0].strip())
            if len(values) < 2:
                continue
            try:
                first, second = float(values[0]), float(values[1])
            except ValueError:
                # This also safely skips a possible x/y header row.
                continue

            if CSV_COORDINATE_ORDER == "xy":
                points.append((first, second))
            else:
                points.append((second, first))

    return points


def find_image(stimulus: str) -> Path | None:
    """Find a supported image whose filename matches the stimulus name."""
    for extension in IMAGE_EXTENSIONS:
        candidate = IMAGE_DIR / f"{stimulus}{extension}"
        if candidate.is_file():
            return candidate

        # Also support upper-case extensions on case-sensitive systems.
        candidate = IMAGE_DIR / f"{stimulus}{extension.upper()}"
        if candidate.is_file():
            return candidate
    return None


def build_heatmapper():
    """Create the v2 image heatmapper with the selected appearance."""
    try:
        from heatmappy2.heatmap import Heatmapper
    except ImportError as exc:
        raise SystemExit(
            "heatmappy2 is not installed. Install it with: pip install heatmappy2"
        ) from exc

    kwargs = {
        "point_diameter": POINT_DIAMETER,
        "point_strength": POINT_STRENGTH,
        "normalisation": NORMALISATION,
    }
    if MODE == "colour":
        kwargs["opacity"] = OPACITY
    else:
        kwargs["mode"] = MODE
        if MODE == "pair":
            kwargs["opacity"] = OPACITY

    return Heatmapper(**kwargs)


def to_heatmappy_points(points: list[tuple[float, float]]):
    """Convert plain x/y coordinates to heatmappy2 v2 Point objects."""
    try:
        from heatmappy2.heatmap import Point
    except ImportError as exc:
        raise SystemExit(
            "heatmappy2 is not installed. Install it with: pip install heatmappy2"
        ) from exc

    # Do not use Point.from_tuple(): it sets diameter_pct and would override
    # the configured POINT_DIAMETER value.
    return [Point(x=x, y=y) for x, y in points]


def stimulus_names() -> list[str]:
    if STIMULI_LIST is not None:
        return [str(name) for name in STIMULI_LIST]
    return sorted(path.stem for path in CSV_DIR.glob("*.csv"))


def main() -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "NumPy or Pillow is not installed. Install them with: pip install numpy Pillow"
        ) from exc

    if not CSV_DIR.is_dir():
        raise FileNotFoundError(f"CSV folder not found: {CSV_DIR}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    names = stimulus_names()
    if not names:
        raise ValueError(f"No CSV files found in: {CSV_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmapper = build_heatmapper()
    created = 0

    for stimulus in names:
        csv_path = CSV_DIR / f"{stimulus}.csv"
        image_path = find_image(stimulus)

        if not csv_path.is_file():
            print(f"Skipped {stimulus}: CSV not found")
            continue
        if image_path is None:
            print(f"Skipped {stimulus}: matching image not found")
            continue

        coordinates = load_image_points(csv_path)
        if not coordinates:
            print(f"Skipped {stimulus}: no valid x/y points")
            continue
        points = to_heatmappy_points(coordinates)

        with Image.open(image_path) as source_image:
            # Pillow uses RGB; heatmappy2 v2 expects and returns BGR arrays.
            rgb_image = np.array(source_image.convert("RGB"), dtype=np.uint8)
            bgr_image = rgb_image[:, :, ::-1].copy()
            heatmap_bgr = heatmapper.heatmap_on_img(points, bgr_image)
            heatmap_rgb = heatmap_bgr[:, :, ::-1]
            output_path = OUTPUT_DIR / f"{stimulus}_heatmap.png"
            Image.fromarray(heatmap_rgb).save(output_path)

        created += 1
        print(f"Saved: {output_path} ({len(coordinates)} points)")

    print(f"Completed: {created}/{len(names)} heatmaps created in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
