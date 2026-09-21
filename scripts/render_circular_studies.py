#!/usr/bin/env python3
"""Draw original circular compositions for a separate cover design review."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render_geometric_studies import Drawing, OUTPUT_DIRECTORY, PAPER, INK, CONSTRUCTION


def circle(drawing, center, radius, width=1.25):
    drawing.path([(center[0] + radius * math.cos(index * math.tau / 480),
                   center[1] + radius * math.sin(index * math.tau / 480))
                  for index in range(481)], width)


def circular_sequence():
    drawing = Drawing("Circular sequence")
    angle = -math.pi / 12
    direction = (math.cos(angle), math.sin(angle))
    normal = (-direction[1], direction[0])
    origin = (270, 352)
    sections = [(0, 182), (103, 147), (190, 115), (262, 84), (320, 56)]

    def position(distance, height=0):
        return (origin[0] + direction[0] * distance + normal[0] * height,
                origin[1] + direction[1] * distance + normal[1] * height)

    drawing.line(position(-205), position(401), 0.7, CONSTRUCTION)
    for index, (distance, radius) in enumerate(sections):
        circle(drawing, position(distance), radius)
        drawing.line(position(distance, -4), position(distance, 4), 0.9)
        if index + 1 < len(sections):
            next_distance, next_radius = sections[index + 1]
            separation = next_distance - distance
            chord_distance = (radius * radius - next_radius * next_radius + separation * separation) / (2 * separation)
            half_chord = math.sqrt(radius * radius - chord_distance * chord_distance)
            drawing.line(position(distance + chord_distance, -half_chord),
                         position(distance + chord_distance, half_chord), 0.85)
    return drawing


def common_chord():
    drawing = Drawing("Common chord")
    center_x, center_y, half_chord = 400, 310, 133
    for offset in [-147, -76, 0, 76, 147]:
        circle(drawing, (center_x + offset, center_y), math.hypot(offset, half_chord))
    drawing.line((400, 144), (400, 476), 1.1)
    drawing.line((211, 310), (589, 310), 0.7, CONSTRUCTION)
    for offset in [-147, -76, 0, 76, 147]:
        drawing.line((400 + offset, 306), (400 + offset, 314), 0.8)
    for y in [177, 443]:
        drawing.line((394, y), (406, y), 1.05)
    return drawing


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    font_directory = Path(__import__("reportlab").__file__).parent / "fonts"
    title_font = ImageFont.truetype(str(font_directory / "Vera.ttf"), 27)
    label_font = ImageFont.truetype(str(font_directory / "Vera.ttf"), 23)
    caption_font = ImageFont.truetype(str(font_directory / "Vera.ttf"), 17)
    board = Image.new("RGB", (1680, 880), PAPER)
    renderer = ImageDraw.Draw(board)
    renderer.text((62, 42), "CIRCULAR STUDIES", font=title_font, fill=INK)
    renderer.text((62, 88), "Option 02 / revised geometry", font=caption_font, fill=INK)
    studies = [("02A", "Circular sequence", "Progression through overlapping circular sections.", circular_sequence),
               ("02B", "Common chord", "Five circles sharing the same two intersection points.", common_chord)]
    for index, (number, title, caption, constructor) in enumerate(studies):
        drawing = constructor()
        raster = drawing.export(title.lower().replace(" ", "_"))
        left, top = 40 + index * 800, 115
        board.paste(raster.resize((800, 620), Image.Resampling.LANCZOS), (left, top))
        renderer.text((left + 25, 754), f"{number}  {title}", font=label_font, fill=INK)
        renderer.text((left + 25, 794), caption, font=caption_font, fill=INK)
    board.save(OUTPUT_DIRECTORY / "circular_studies.png")
    print(OUTPUT_DIRECTORY / "circular_studies.png")


if __name__ == "__main__":
    main()
