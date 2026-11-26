# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
"""A few SVG helpers for the frame renderer (no dependencies)."""
from xml.sax.saxutils import escape

FONT = "DejaVu Sans"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def rect(self, x, y, w, h, fill="#fff", stroke="#333", sw=1.2, rx=4, dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

    def text(self, x, y, s, size=13, fill="#111", weight="normal", anchor="start", mono=False):
        fam = "DejaVu Sans Mono" if mono else FONT
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')

    def lines(self, x, y, ss, size=13, dy=None, **kw):
        dy = dy or size * 1.32
        for i, s in enumerate(ss):
            self.text(x, y + i * dy, s, size=size, **kw)

    def line(self, x1, y1, x2, y2, stroke="#555", sw=1.5, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
        if arrow:
            import math
            a = math.atan2(y2 - y1, x2 - x1)
            L = 9
            for s in (-0.45, 0.45):
                self.parts.append(f'<line x1="{x2:.1f}" y1="{y2:.1f}" x2="{x2 - L * math.cos(a + s):.1f}" y2="{y2 - L * math.sin(a + s):.1f}" stroke="{stroke}" stroke-width="{sw}"/>')

    def raw(self, s):
        self.parts.append(s)

    def done(self):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><rect width="100%" height="100%" fill="#fbfbf8"/>' + "".join(self.parts) + "</svg>"
