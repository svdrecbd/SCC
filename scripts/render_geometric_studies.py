#!/usr/bin/env python3
"""Draw four geometric cover studies as editable SVGs and raster previews.

The constructions are visual studies, not diagrams of established SCC results.
Only analytic coordinates and ordinary vector/raster drawing operations are used.
"""

import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = REPOSITORY / "deliverables/geometric-studies"
WIDTH, HEIGHT = 800, 620
PAPER = "#E9E5D9"
INK = "#33362F"
CONSTRUCTION = "#9D9D91"


class Drawing:
    """A small shared representation for SVG and supersampled PNG output."""

    def __init__(self, title):
        self.title = title
        self.paths = []

    def line(self, start, end, width=1.1, color=INK):
        self.paths.append(([start, end], width, color))

    def path(self, coordinates, width=1.1, color=INK):
        self.paths.append((coordinates, width, color))

    def export(self, filename):
        elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}">',
                    f'<title>{html.escape(self.title)}</title>',
                    '<desc>Geometric design study. No experimental data or mechanism claim.</desc>']
        scale = 3
        raster = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), PAPER)
        renderer = ImageDraw.Draw(raster)
        for coordinates, width, color in self.paths:
            points = " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates)
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')
            renderer.line([(round(x * scale), round(y * scale)) for x, y in coordinates],
                          fill=color, width=max(1, round(width * scale)), joint="curve")
        elements.append("</svg>")
        (OUTPUT_DIRECTORY / f"{filename}.svg").write_text("\n".join(elements) + "\n")
        raster = raster.resize((WIDTH * 2, HEIGHT * 2), Image.Resampling.LANCZOS)
        raster.save(OUTPUT_DIRECTORY / f"{filename}.png")
        return raster


def interpolate(start, end, fraction):
    return tuple(a + (b - a) * fraction for a, b in zip(start, end))


def shared_lattice():
    drawing = Drawing("Shared lattice")
    left_top, left_bottom = (100, 138), (100, 464)
    middle_top, middle_bottom = (402, 255), (402, 373)
    right_top, right_bottom = (700, 138), (700, 464)
    for outer_top, outer_bottom in [(left_top, left_bottom), (right_top, right_bottom)]:
        drawing.path([outer_top, middle_top, middle_bottom, outer_bottom, outer_top], 1.3)
        for index in range(1, 7):
            fraction = index / 7
            drawing.line(interpolate(outer_top, middle_top, fraction),
                         interpolate(outer_bottom, middle_bottom, fraction), 0.95)
        for fraction in [0.25, 0.5, 0.75]:
            drawing.line(interpolate(outer_top, outer_bottom, fraction),
                         interpolate(middle_top, middle_bottom, fraction), 0.9)
    left_origin, right_origin = (51, 313), (749, 313)
    for index in range(6):
        fraction = index / 5
        drawing.line(left_origin, interpolate(middle_bottom, right_bottom, fraction), 0.9)
        drawing.line(right_origin, interpolate(left_bottom, middle_bottom, fraction), 0.9)
    drawing.line(middle_top, middle_bottom, 1.85)
    for fraction in [0, 0.25, 0.5, 0.75, 1]:
        for start, end in [(left_bottom, middle_bottom), (middle_bottom, right_bottom)]:
            x, y = interpolate(start, end, fraction)
            drawing.line((x, y + 15), (x, y + 25), 0.8)
    return drawing


def projected_sections():
    drawing = Drawing("Projected sections")
    vanishing_point = (716, 274)
    center = (218, 323)
    radius_x, radius_y = 100, 211

    def project(horizontal, vertical, depth):
        scale = 1 / (1 + depth * 0.54)
        return (vanishing_point[0] + (center[0] + horizontal - vanishing_point[0]) * scale,
                vanishing_point[1] + (center[1] + vertical - vanishing_point[1]) * scale)

    for horizontal, vertical in [(-radius_x, -radius_y), (radius_x, -radius_y),
                                  (-radius_x, radius_y), (radius_x, radius_y), (0, 0)]:
        drawing.line(project(horizontal, vertical, 0), vanishing_point, 0.9)
    for depth in [0, 1, 2.5, 4.7, 8.2]:
        drawing.path([project(radius_x * math.cos(index * math.tau / 240),
                              radius_x * math.sin(index * math.tau / 240), depth)
                      for index in range(241)], 1.15)
        corners = [project(x, y, depth) for x, y in [(-radius_x, -radius_y),
                   (radius_x, -radius_y), (radius_x, radius_y), (-radius_x, radius_y),
                   (-radius_x, -radius_y)]]
        drawing.path(corners, 0.8)
        drawing.line(project(0, -radius_y, depth), project(0, radius_y, depth), 0.8)
        bottom = project(0, radius_y, depth)
        drawing.line(bottom, (min(bottom[0] + 96, 705), bottom[1]), 0.75)
    return drawing


def project_isometric(x, y, z):
    return (400 + 73 * (x - y), 320 + 39 * (x + y) - 89 * z)


def intersecting_planes():
    drawing = Drawing("Intersecting planes")
    extent = 1.9
    for index in range(9):
        coordinate = -extent + index * extent / 4
        width = 1.35 if index in [0, 8] else 0.8
        for start, end in [((coordinate, -extent, 0), (coordinate, extent, 0)),
                           ((-extent, coordinate, 0), (extent, coordinate, 0)),
                           ((coordinate, 0, -extent), (coordinate, 0, extent)),
                           ((-extent, 0, coordinate), (extent, 0, coordinate)),
                           ((0, coordinate, -extent), (0, coordinate, extent)),
                           ((0, -extent, coordinate), (0, extent, coordinate))]:
            drawing.line(project_isometric(*start), project_isometric(*end), width)
    for start, end in [((-extent, 0, 0), (extent, 0, 0)),
                       ((0, -extent, 0), (0, extent, 0)),
                       ((0, 0, -extent), (0, 0, extent))]:
        drawing.line(project_isometric(*start), project_isometric(*end), 1.8)
    return drawing


def ruled_surface():
    drawing = Drawing("Ruled surface")

    def project(x, y, z):
        return (400 + 142 * x + 126 * y, 327 + 43 * x - 58 * y - 136 * z)

    corners = [project(x, y, 0) for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]]
    drawing.path(corners, 0.75, CONSTRUCTION)
    for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        drawing.line(project(x, y, 0), project(x, y, x * y), 0.75, CONSTRUCTION)
    for index in range(19):
        value = -1 + index / 9
        width = 1.45 if index in [0, 18] else 0.9
        drawing.line(project(value, -1, -value), project(value, 1, value), width)
        drawing.line(project(-1, value, -value), project(1, value, value), width)
    drawing.line(project(-1, 0, 0), project(1, 0, 0), 1.3)
    drawing.line(project(0, -1, 0), project(0, 1, 0), 1.3)
    return drawing


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    studies = [("01", "Shared lattice", "Two systems with a common structural edge.", shared_lattice),
               ("02", "Projected sections", "Successive sections under one perspective map.", projected_sections),
               ("03", "Intersecting planes", "Three coordinate planes and their shared axes.", intersecting_planes),
               ("04", "Ruled surface", "Two straight-line families forming one continuous surface.", ruled_surface)]
    font_directory = Path(__import__("reportlab").__file__).parent / "fonts"
    title_font = ImageFont.truetype(str(font_directory / "Vera.ttf"), 30)
    label_font = ImageFont.truetype(str(font_directory / "Vera.ttf"), 23)
    caption_font = ImageFont.truetype(str(font_directory / "Vera.ttf"), 17)
    board = Image.new("RGB", (1680, 1570), PAPER)
    renderer = ImageDraw.Draw(board)
    renderer.text((58, 44), "GEOMETRIC STUDIES", fill=INK, font=title_font)
    renderer.text((58, 91), "Cover exploration / drawn in Python / no placement selected", fill=INK, font=caption_font)
    for index, (number, title, caption, constructor) in enumerate(studies):
        drawing = constructor()
        filename = title.lower().replace(" ", "_")
        raster = drawing.export(filename)
        left = 40 + index % 2 * 800
        top = 135 + index // 2 * 690
        board.paste(raster.resize((800, 620), Image.Resampling.LANCZOS), (left, top))
        renderer.text((left + 35, top + 608), f"{number}  {title}", fill=INK, font=label_font)
        renderer.text((left + 35, top + 643), caption, fill=INK, font=caption_font)
    board.save(OUTPUT_DIRECTORY / "geometric_studies.png")
    print(OUTPUT_DIRECTORY / "geometric_studies.png")


if __name__ == "__main__":
    main()
