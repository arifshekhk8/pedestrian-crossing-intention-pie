"""drawio_lib.py — a small mxGraph (draw.io) XML writer.

Everything here emits plain `.drawio` files that open, and are fully editable, in
the desktop draw.io app. Nothing is embedded as an image: every box, line, tick
and label is a real shape you can select, drag and restyle.

Coordinates are in points, y increasing downward, which is what draw.io uses.

The chart helpers exist because the figures being reproduced are mostly data
plots. Positions are computed from the actual numbers rather than eyeballed, so
a bar in the draw.io file is the same length as the bar in the paper. Redrawing
a chart by hand and nudging bars until they look right is how a figure quietly
stops matching its own results.
"""

from xml.sax.saxutils import escape

# Palette, identical to figures/figstyle.py so the two sets of figures match.
BILSTM = "#2a78d6"
TRANSF = "#d1622b"
GRU = "#1f9b7a"
RNN = "#8b5cc7"
ACCENT = BILSTM
CONTEXT = "#b9b8b2"
LEAK = "#b4433a"
CLEAN = "#1f9b7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
RULE = "#c3c2b7"
PANEL = "#f4f3ee"
WHITE = "#ffffff"
NONE = "none"


def _sty(d):
    """Build an mxGraph style string. `True` means a valueless flag: draw.io
    wants a bare `ellipse;`, not `ellipse=1;`, and a shape written the wrong way
    silently falls back to a rectangle."""
    out = []
    for k, v in d.items():
        if v is None:
            continue
        out.append(f"{k};" if v is True else f"{k}={v};")
    return "".join(out)


def _label(value):
    """Escape for an XML attribute. draw.io labels are HTML, so a newline has to
    become a <br> or the line break is silently dropped."""
    s = escape(str(value), {'"': "&quot;"})
    return s.replace("\n", "&lt;br&gt;")


class Doc:
    """One draw.io file holding one diagram."""

    # MDPI \textwidth, measured from the compiled manuscript (\the\textwidth).
    PAGE_PT = 394.36
    BASE_FONT = 11.0      # the size ordinary body labels are written at here
    # What BASE_FONT should measure once the figure is fitted to \textwidth.
    # 5.2 is not a free choice: the matplotlib figures author 7.8 pt text on a
    # 511 pt canvas, so they print at 7.8 x 394/511 = 6.0 pt, and the ratio of
    # text size to box size here is what allows. Asking for 8 pt inflates the
    # labels past the boxes that hold them.
    TARGET_PT = 5.2

    def __init__(self, name, width=1200, height=760):
        self.name = name
        self.w, self.h = width, height
        self.cells = []
        self._n = 0
        # A figure this wide gets scaled to \textwidth in the manuscript, which
        # shrinks everything on it by PAGE_PT/width. Left alone, an 11 pt label
        # on a 1360 pt canvas prints at 3.2 pt. Fonts and stroke widths are
        # therefore pre-multiplied so they land where they were designed to.
        self.fk = (self.TARGET_PT / self.BASE_FONT) * (width / self.PAGE_PT)

    # ------------------------------------------------------------ primitives
    def _id(self):
        self._n += 1
        return f"n{self._n}"

    def _vertex(self, x, y, w, h, value, style, parent="1"):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{_label(value)}" style="{style}" '
            f'vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" as="geometry"/></mxCell>')
        return i

    def rect(self, x, y, w, h, value="", *, fill=PANEL, stroke=RULE, font=11,
             color=INK, bold=False, rounded=True, dashed=False, lw=1,
             align="center", valign="middle", opacity=None, radius=6):
        style = _sty({
            "rounded": 1 if rounded else 0,
            "arcSize": radius if rounded else None,
            "whiteSpace": "wrap", "html": 1,
            "fillColor": fill, "strokeColor": stroke,
            "strokeWidth": round(lw * self.fk, 2),
            "dashed": 1 if dashed else 0,
            "dashPattern": "6 4" if dashed else None,
            "fontSize": round(font * self.fk, 1), "fontColor": color,
            "fontStyle": 1 if bold else 0,
            "align": align, "verticalAlign": valign,
            "opacity": opacity, "shadow": 0,
        })
        return self._vertex(x, y, w, h, value, style)

    def text(self, x, y, w, h, value, *, font=11, color=INK, bold=False,
             italic=False, align="left", valign="middle", spacing=0):
        fs = (1 if bold else 0) | (2 if italic else 0)
        style = _sty({
            "text": True, "html": 1, "whiteSpace": "wrap",
            "fillColor": NONE, "strokeColor": NONE,
            "fontSize": round(font * self.fk, 1), "fontColor": color,
            "fontStyle": fs,
            "align": align, "verticalAlign": valign,
            "spacingLeft": spacing or None, "resizable": 0,
        })
        return self._vertex(x, y, w, h, value, style)

    def ellipse(self, x, y, w, h, value="", *, fill=PANEL, stroke=RULE,
                font=11, color=INK, bold=False):
        style = _sty({
            "ellipse": True, "whiteSpace": "wrap", "html": 1,
            "fillColor": fill, "strokeColor": stroke,
            "strokeWidth": round(1.2 * self.fk, 2),
            "fontSize": round(font * self.fk, 1), "fontColor": color,
            "fontStyle": 1 if bold else 0,
        })
        return self._vertex(x, y, w, h, value, style)

    def line(self, x0, y0, x1, y1, *, stroke=RULE, lw=1, dashed=False,
             arrow="none", start_arrow="none", points=None, rounded=False):
        i = self._id()
        style = _sty({
            "endArrow": arrow, "startArrow": start_arrow,
            "endFill": 1 if arrow not in ("none", "open") else 0,
            "startFill": 1 if start_arrow not in ("none", "open") else 0,
            "html": 1, "rounded": 1 if rounded else 0,
            "strokeColor": stroke, "strokeWidth": round(lw * self.fk, 2),
            "dashed": 1 if dashed else 0,
            "dashPattern": "5 4" if dashed else None,
            "edgeStyle": "none", "endSize": 5, "startSize": 5,
        })
        pts = ""
        if points:
            inner = "".join(f'<mxPoint x="{px:.1f}" y="{py:.1f}"/>' for px, py in points)
            pts = f'<Array as="points">{inner}</Array>'
        self.cells.append(
            f'<mxCell id="{i}" style="{style}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x0:.1f}" y="{y0:.1f}" as="sourcePoint"/>'
            f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/>'
            f'{pts}</mxGeometry></mxCell>')
        return i

    def polyline(self, pts, *, stroke=BILSTM, lw=2, dashed=False):
        """A curve. draw.io stores it as an edge with waypoints."""
        if len(pts) < 2:
            return None
        return self.line(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1],
                         stroke=stroke, lw=lw, dashed=dashed,
                         points=pts[1:-1] or None)

    def arrow(self, x0, y0, x1, y1, *, stroke=RULE, lw=1, dashed=False):
        return self.line(x0, y0, x1, y1, stroke=stroke, lw=lw, dashed=dashed,
                         arrow="blockThin")

    # ----------------------------------------------------------------- output
    def xml(self):
        body = "".join(self.cells)
        return (
            '<mxfile host="Electron" agent="pedestrian-thesis figure generator" '
            'type="device">'
            f'<diagram id="{escape(self.name)}" name="{escape(self.name)}">'
            f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
            '</mxGraphModel></diagram></mxfile>')

    def save(self, path):
        path.write_text(self.xml())
        print(f"wrote {path}  ({len(self.cells)} shapes)")


# ---------------------------------------------------------------------------
# Chart scaffolding
# ---------------------------------------------------------------------------

class Axes:
    """Maps data coordinates onto a rectangle of the page.

    `x0,y0` is the top-left of the plotting rectangle in page points.
    Data y increases upward, page y increases downward; `py` handles the flip.
    """

    def __init__(self, doc, x0, y0, w, h, xlim, ylim):
        self.d, self.x0, self.y0, self.w, self.h = doc, x0, y0, w, h
        self.xlim, self.ylim = xlim, ylim

    def px(self, v):
        a, b = self.xlim
        return self.x0 + (v - a) / (b - a) * self.w

    def py(self, v):
        a, b = self.ylim
        return self.y0 + self.h - (v - a) / (b - a) * self.h

    def hgrid(self, values, *, color=GRID, lw=1):
        for v in values:
            self.d.line(self.x0, self.py(v), self.x0 + self.w, self.py(v),
                        stroke=color, lw=lw)

    def vgrid(self, values, *, color=GRID, lw=1):
        for v in values:
            self.d.line(self.px(v), self.y0, self.px(v), self.y0 + self.h,
                        stroke=color, lw=lw)

    def yticks(self, values, labels=None, *, font=10, color=INK_2, width=60):
        # Label boxes scale with the fonts. Left fixed, a two-line tick at 1.7x
        # type overflows its box and lands on whatever is underneath.
        k = self.d.fk
        labels = labels or [str(v) for v in values]
        for v, s in zip(values, labels):
            self.d.text(self.x0 - width * k - 6 * k, self.py(v) - 9 * k,
                        width * k, 18 * k, s, font=font, color=color,
                        align="right")

    def xticks(self, values, labels=None, *, font=10, color=INK_2, width=54,
               dy=6):
        k = self.d.fk
        labels = labels or [str(v) for v in values]
        for v, s in zip(values, labels):
            self.d.text(self.px(v) - width * k / 2, self.y0 + self.h + dy * k,
                        width * k, 24 * k, s, font=font, color=color,
                        align="center")
        # Where an axis label may safely go, below the tallest tick label.
        self.tick_bottom = self.y0 + self.h + dy * k + 26 * k

    def spine_bottom(self, *, color=RULE, lw=1):
        self.d.line(self.x0, self.y0 + self.h, self.x0 + self.w,
                    self.y0 + self.h, stroke=color, lw=lw)

    def spine_left(self, *, color=RULE, lw=1):
        self.d.line(self.x0, self.y0, self.x0, self.y0 + self.h,
                    stroke=color, lw=lw)

    def vbar(self, xc, value, width, *, fill=ACCENT, base=0.0, stroke=NONE):
        top, bot = self.py(max(value, base)), self.py(min(value, base))
        self.d.rect(self.px(xc) - width / 2, top, width, max(bot - top, 0.6),
                    fill=fill, stroke=stroke, rounded=False)

    def hbar(self, yc, value, height, *, fill=ACCENT, base=0.0, stroke=NONE):
        left, right = self.px(min(value, base)), self.px(max(value, base))
        self.d.rect(left, self.py(yc) - height / 2, max(right - left, 0.6),
                    height, fill=fill, stroke=stroke, rounded=False)

    def curve(self, pts, *, stroke=ACCENT, lw=2, dashed=False):
        self.d.polyline([(self.px(a), self.py(b)) for a, b in pts],
                        stroke=stroke, lw=lw, dashed=dashed)

    def marker(self, x, y, *, size=9, fill=ACCENT, stroke=WHITE, lw=1.4):
        size *= self.d.fk          # else a marker is a speck once page-fitted
        self.d.ellipse(self.px(x) - size / 2, self.py(y) - size / 2, size, size,
                       fill=fill, stroke=stroke)


def panel_title(doc, x, y, title, subtitle=None, *, width=900):
    doc.text(x, y, width, 22 * doc.fk, title, font=14, color=INK, bold=True)
    if subtitle:
        doc.text(x, y + 24 * doc.fk, width, 34 * doc.fk, subtitle, font=11,
                 color=MUTED)


def legend(doc, x, y, items, *, font=11, swatch=16, gap=20, vertical=True):
    """items: list of (color, label). Returns the height used.

    Swatch and spacing scale with the page-fit factor for the same reason the
    fonts do; a 16 pt swatch on a 1400 pt canvas prints at 4 pt."""
    swatch *= doc.fk
    gap *= doc.fk
    for i, (color, label) in enumerate(items):
        if vertical:
            cx, cy = x, y + i * gap
        else:
            cx, cy = x + i * 210 * doc.fk, y
        doc.rect(cx, cy + 0.18 * gap, swatch, 0.55 * gap, fill=color,
                 stroke=NONE, rounded=False)
        doc.text(cx + swatch + 0.4 * gap, cy, 420 * doc.fk, gap * 0.9,
                 label, font=font, color=INK_2)
    return len(items) * gap if vertical else gap
