#!/usr/bin/env python3
"""Rebuild Figure 4.4's compact vector legend above the plot panels.

This post-processes the generated Matplotlib PDF without altering any plotted
data.  It is useful when the source data reader is unavailable but the stored
vector figure still contains the exact legend paths and text.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path


PDF = Path(__file__).resolve().parents[1] / "images" / "ch4_coverage.pdf"


def pdf_objects(data: bytes) -> tuple[bytes, dict[int, bytes]]:
    xref_at = data.rfind(b"\nxref\n") + 1
    if xref_at <= 0:
        raise RuntimeError("xref table not found")
    xref = data[xref_at:].splitlines()
    count = int(xref[1].split()[1])
    offsets: dict[int, int] = {}
    for number, entry in enumerate(xref[2:2 + count]):
        if number and entry.endswith(b" n "):
            offsets[number] = int(entry[:10])
    ordered = sorted((offset, number) for number, offset in offsets.items())
    objects: dict[int, bytes] = {}
    for index, (offset, number) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else xref_at
        objects[number] = data[offset:end].strip()
    header = data[:ordered[0][0]]
    return header, objects


def rebuild_pdf(header: bytes, objects: dict[int, bytes]) -> bytes:
    output = bytearray(header)
    if output and not output.endswith(b"\n"):
        output.extend(b"\n")
    offsets = {0: 0}
    for number in range(1, max(objects) + 1):
        offsets[number] = len(output)
        output.extend(objects[number])
        output.extend(b"\n")
    xref_at = len(output)
    size = max(objects) + 1
    output.extend(f"xref\n0 {size}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for number in range(1, size):
        output.extend(f"{offsets[number]:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {size} /Root 1 0 R /Info 34 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode()
    )
    return bytes(output)


def legend_stream() -> bytes:
    def label(text: str, x: float) -> bytes:
        encoded = text.encode("utf-16-be").hex()
        return (
            f"q\n0 g\n0 G\n1 0 0 1 {x:.3f} 331.500 cm\nBT\n/F1 8.2 Tf\n"
            f"0 0 Td\n<{encoded}> Tj\nET\nQ\n"
        ).encode()

    commands = bytearray(
        b"\nq\n1 g\n0 316.054 426.939375 42 re\nf\nQ\n"
        b"/A4 gs\n"
    )
    colors = (
        (0.2078431373, 0.4941176471, 0.7803921569),
        (0.8862745098, 0.6039215686, 0.0862745098),
        (0.1450980392, 0.6313725490, 0.5568627451),
        (0.7098039216, 0.3568627451, 0.6039215686),
    )
    for x, (red, green, blue) in zip((102, 112, 122, 132), colors):
        commands.extend(
            f"{red} {green} {blue} rg\n{x} 331 8 7 re\nf\n".encode()
        )
    commands.extend(label("closed loop", 146))
    commands.extend(b"0.1411764706 0.1921568627 0.2392156863 RG\n1.3 w\n")
    commands.extend(b"245 331 12 7 re\nS\n")
    commands.extend(label("open loop", 263))
    return bytes(commands)


def main() -> None:
    raw = PDF.read_bytes()
    header, objects = pdf_objects(raw)

    stream_match = re.search(
        rb"stream\r?\n(.*?)\r?\nendstream", objects[9], re.DOTALL
    )
    if not stream_match:
        raise RuntimeError("Figure content stream not found")
    content = zlib.decompress(stream_match.group(1))

    white_band = b"\nq\n1 g\n0 316.054 426.939375 42 re\nf\nQ\n"
    band_at = content.find(white_band)
    if band_at < 0:
        raise RuntimeError("Expected relocated Figure 4.4 legend band was not found")
    content = content[:band_at]

    # Remove the old explanatory sentence while preserving the final graphics
    # state restoration that follows its text object.
    note_transform = b"q\n1 0 -0 1 101.4924375 306.289 cm"
    note_at = content.find(note_transform)
    muted = b"0.3960784314 0.4470588235 0.4941176471 rg"
    if note_at >= 0:
        note_start = content.rfind(muted, 0, note_at)
        note_end_marker = b"ET\nQ\nQ\n"
        note_end = content.find(note_end_marker, note_at)
        if note_start < 0 or note_end < 0:
            raise RuntimeError("Could not isolate the Figure 4.4 explanatory text")
        content = content[:note_start] + content[note_end + len(b"ET\nQ\n"):]
    content += legend_stream()
    compressed = zlib.compress(content, 9)

    objects[9] = (
        b"9 0 obj\n<< /Length 12 0 R /Filter /FlateDecode >>\nstream\n"
        + compressed
        + b"\nendstream\nendobj"
    )
    objects[12] = f"12 0 obj\n{len(compressed)}\nendobj".encode()
    objects[11] = re.sub(
        rb"/MediaBox \[ 0 0 426\.939375 316\.054 \]",
        b"/MediaBox [ 0 0 426.939375 358.054 ]",
        objects[11],
    )

    PDF.write_bytes(rebuild_pdf(header, objects))
    print(f"updated {PDF}")


if __name__ == "__main__":
    main()
