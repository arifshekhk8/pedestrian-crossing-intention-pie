"""prep_fig2_icons.py — the stage icons used in Figures 2 and 2b.

Every icon is downloaded from **Tabler Icons**, an open-source set drawn by
Paweł Kuna and contributors and released under the MIT licence:

    repository   https://github.com/tabler/tabler-icons
    licence      https://github.com/tabler/tabler-icons/blob/main/LICENSE  (MIT)
    per-icon     https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/<name>.svg
    browse       https://tabler.io/icons

Why this set rather than a stock-icon site. Three things matter for a figure a
reviewer will look at:

  1. One designer, one 24-pixel grid, one stroke width. Icons pulled from
     different authors on a stock site have visibly different weights and corner
     radii, which is exactly the mismatch that makes a diagram look assembled
     rather than drawn.
  2. The MIT licence needs no credit line in the paper. Flaticon's free licence
     requires crediting each icon's author in the publication, which for a set
     this size means several names in the caption or acknowledgements.
  3. Provenance is checkable: every icon is a file in a public git repository
     with authorship history, so "not machine-generated" is verifiable rather
     than asserted.

Nothing here is AI-generated. If the Flaticon versions are wanted instead, the
manifest below is the only thing that has to change; verified equivalents are
`crosswalk_2584882` (Freepik) and `neural-network_18853561` (Nuricon), whose
PNGs come from https://cdn-icons-png.flaticon.com/512/<first-digits>/<id>.png.

The SVGs are rasterised at 256 px, which is roughly ten times the size the icons
print at, and stored as `_icon_<name>.png` next to this file. They are committed,
so the figure builds with no network access.

Run once:  python prep_fig2_icons.py
"""

from pathlib import Path
from urllib.request import urlopen

import cairosvg

HERE = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline"
PX = 256

# (name, what it stands for in the figure)
ICONS = [
    ("crop", "the pedestrian bounding box: the geometric stream"),
    ("gauge", "ego-vehicle speed: the telemetry stream"),
    ("adjustments", "preprocessing: standardise, then project"),
    ("binary-tree", "the temporal encoder, in both figures"),
    ("arrows-join", "the read-out: a sequence collapsed to one logit"),
    ("walk", "the crossing probability, in both figures"),
    ("video", "an incoming video frame"),
    ("scan", "the detector"),
    ("route", "identity association across frames"),
    ("stack-2", "the per-track ring buffer"),
]


def main():
    for name, purpose in ICONS:
        url = f"{BASE}/{name}.svg"
        with urlopen(url, timeout=30) as r:
            svg = r.read().decode()
        # Tabler strokes are `currentColor`, which has no meaning outside a
        # browser; the figure sets every mark in black, so bind it here.
        svg = svg.replace("currentColor", "#000000")
        out = HERE / f"_icon_{name}.png"
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out),
                         output_width=PX, output_height=PX)
        print(f"{name:14s} {out.name:26s} {purpose}")


if __name__ == "__main__":
    main()
