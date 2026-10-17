"""Getting the results out of the workbench and into the tools the office already uses.

Two writers, for the two things a civil engineer actually sends somewhere:

* :func:`write_excel` — every table in the study, one sheet per table, with a frozen bold
  header and sensible column widths.  Excel is where a reviewer will re-check the numbers,
  so the tables have to arrive as data rather than as a picture of data.  Uses
  ``openpyxl``, which is already installed because pandas needs it.
* :func:`write_docx` — a Word document for the submission report: headings, paragraphs,
  bullets, tables and the figures, in order.

**Word is written by hand rather than with a library.**  ``python-docx`` is not installed
and pulling it in would add ``lxml`` to a dependency list that only exists so the venv can
be rebuilt offline.  A ``.docx`` is a zip of XML, and the subset needed here — direct
formatting, borders on tables, inline pictures — is small and does not require a styles
part, which is the part that usually goes wrong.  Bullets are written as a literal "•  "
prefix with an indent for the same reason: real list numbering needs a numbering part and
buys nothing.
"""

from __future__ import annotations

import os
import re
import struct
import zipfile

import pandas as pd

# EMU (English Metric Units) — Word's internal unit: 914400 to the inch.
EMU_PER_INCH = 914400

#: Excel sheet names may not contain these, and cap out at 31 characters.
_BAD_SHEET = re.compile(r"[\[\]:*?/\\]")


def _sheet_name(name: str, taken: set) -> str:
    cleaned = _BAD_SHEET.sub("-", str(name)).strip()[:31] or "Sheet"
    candidate, index = cleaned, 2
    while candidate.lower() in taken:
        suffix = f" ({index})"
        candidate = cleaned[:31 - len(suffix)] + suffix
        index += 1
    taken.add(candidate.lower())
    return candidate


def _safe_value(value):
    """Excel refuses anything it cannot type; turn those cells into text."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:   # NaN
            return ""
        return value
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            item = value.item()
            return "" if isinstance(item, float) and item != item else item
        if isinstance(value, np.ndarray):
            return str(value.tolist())
    except Exception:  # noqa: BLE001
        pass
    return str(value)


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------

def write_excel(path: str, sheets: dict, notes=None, title: str = "ML Workbench results") -> str:
    """Write ``{sheet name: DataFrame}`` to a formatted workbook.

    A leading "Notes" sheet records where each table came from, so a workbook that is
    emailed on its own still says what it is.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)

    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    header_align = Alignment(vertical="center", horizontal="left", wrap_text=True)
    body_font = Font(size=10)
    thin = Side(style="thin", color="D0D0D0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    taken: set = set()
    if notes or title:
        sheet = workbook.create_sheet("Notes")
        taken.add("notes")
        sheet["A1"] = title
        sheet["A1"].font = Font(bold=True, size=14, color="1F3864")
        row = 3
        for line in (notes or []):
            sheet.cell(row=row, column=1, value=str(line)).font = body_font
            row += 1
        sheet.column_dimensions["A"].width = 110

    for name, frame in sheets.items():
        if frame is None:
            continue
        if isinstance(frame, dict):
            frame = pd.DataFrame(frame)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        sheet = workbook.create_sheet(_sheet_name(name, taken))
        columns = [str(c) for c in frame.columns]
        sheet.append(columns)
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = border
        for record in frame.itertuples(index=False):
            sheet.append([_safe_value(v) for v in record])
        for column in sheet.columns:
            letter = get_column_letter(column[0].column)
            longest = 0
            for cell in column:
                cell.font = body_font
                cell.border = border
                text = "" if cell.value is None else str(cell.value)
                longest = max(longest, min(len(text), 60))
            sheet.column_dimensions[letter].width = max(9, min(longest + 2, 46))
        sheet.freeze_panes = "A2"
        if sheet.max_row > 1:
            sheet.auto_filter.ref = (f"A1:{get_column_letter(sheet.max_column)}"
                                     f"{sheet.max_row}")

    if not workbook.sheetnames:
        workbook.create_sheet("Empty")

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    workbook.save(path)
    return path


# --------------------------------------------------------------------------
# Word — a small OOXML writer
# --------------------------------------------------------------------------

def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _png_size(data: bytes) -> tuple[int, int]:
    """Read width and height out of a PNG header, so pictures keep their aspect ratio."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
    except Exception:  # noqa: BLE001
        pass
    return 1000, 600


def _run(text, bold=False, italic=False, size=None, colour=None, mono=False) -> str:
    properties = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    if size:
        properties.append(f'<w:sz w:val="{int(size * 2)}"/>')
    if colour:
        properties.append(f'<w:color w:val="{colour}"/>')
    if mono:
        properties.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
    properties = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    return (f'<w:r>{properties}<w:t xml:space="preserve">{_escape(text)}</w:t></w:r>')


def _paragraph(runs: str, spacing_before: int = 0, spacing_after: int = 100,
               indent: int = 0, align: str | None = None) -> str:
    properties = [f'<w:spacing w:before="{spacing_before}" w:after="{spacing_after}"/>']
    if indent:
        properties.append(f'<w:ind w:left="{indent}"/>')
    if align:
        properties.append(f'<w:jc w:val="{align}"/>')
    return f"<w:p><w:pPr>{''.join(properties)}</w:pPr>{runs}</w:p>"


def _table_xml(frame: pd.DataFrame, header: bool = True, font_size: float = 8.0) -> str:
    columns = [str(c) for c in frame.columns]
    column_count = max(1, len(columns))
    usable = int(6.5 * EMU_PER_INCH)          # a Letter page with 1-inch margins
    width = max(600, usable // column_count // 635)   # twips
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"))

    def cell(text, bold=False, fill=None, align=None):
        shading = f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>' if fill else ""
        return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shading}</w:tcPr>'
                + _paragraph(_run(text, bold=bold, size=font_size), 20, 20, 0, align)
                + "</w:tc>")

    parts = ["<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
             f"<w:tblBorders>{borders}</w:tblBorders>"
             "<w:tblLayout w:type=\"fixed\"/></w:tblPr>"
             f"<w:tblGrid>{'<w:gridCol w:w=\"%d\"/>' % width * column_count}</w:tblGrid>"]
    if header:
        parts.append("<w:tr><w:trPr><w:tblHeader/></w:trPr>"
                     + "".join(cell(c, bold=True, fill="1F3864") for c in columns)
                     + "</w:tr>")
    for record in frame.itertuples(index=False):
        cells = []
        for value in record:
            if isinstance(value, float):
                text = "" if value != value else f"{value:,.6g}"
            else:
                text = "" if value is None else str(value)
            cells.append(cell(text))
        parts.append("<w:tr>" + "".join(cells) + "</w:tr>")
    parts.append("</w:tbl>")
    # Word requires a paragraph after a table, or the next heading can merge into it.
    parts.append(_paragraph("", 0, 60))
    return "".join(parts)


def _image_xml(relationship_id: str, data: bytes, draw_id: int,
               width_inches: float = 6.5) -> str:
    pixel_width, pixel_height = _png_size(data)
    cx = int(width_inches * EMU_PER_INCH)
    cy = int(cx * pixel_height / max(1, pixel_width))
    return (
        "<w:p><w:pPr><w:spacing w:before=\"120\" w:after=\"60\"/>"
        "<w:jc w:val=\"center\"/></w:pPr><w:r><w:drawing>"
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{draw_id}" name="Picture {draw_id}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{draw_id}" name="image{draw_id}.png"/>'
        "<pic:cNvPicPr/></pic:nvPicPr>"
        f'<pic:blipFill><a:blip r:embed="{relationship_id}"/>'
        "<a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
        "<pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/>"
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        "<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr>"
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>")


def write_docx(path: str, blocks, title: str = "") -> str:
    """Write a Word document from a list of blocks.

    A block is one of:

    * ``("heading", level, text)`` — level 1 is the document title, 2 and 3 are sub-headings.
    * ``("paragraph", text)`` — a body paragraph.
    * ``("bullet", item)`` or ``("bullets", [items])`` — a bulleted list.
    * ``("table", DataFrame)`` or ``("table", DataFrame, caption)`` — a bordered table.
    * ``("image", png_bytes)`` or ``("image", png_bytes, caption)`` — a centred picture.
    * ``("pagebreak",)`` — start a new page.
    """
    body: list[str] = []
    media: list[tuple[str, bytes]] = []
    relationships: list[str] = []
    picture_number = 0

    if title:
        body.append(_paragraph(_run(title, bold=True, size=20, colour="1F3864"),
                               0, 240, 0, "center"))

    for block in blocks:
        if not block:
            continue
        kind = str(block[0]).lower()
        payload = block[1] if len(block) > 1 else None
        caption = block[2] if len(block) > 2 else None
        if kind == "heading":
            level = int(payload or 2)
            size = {1: 16, 2: 13, 3: 11}.get(level, 11)
            body.append(_paragraph(_run(caption, bold=True, size=size, colour="1F3864"),
                                   240 if level <= 2 else 180, 100))
        elif kind == "paragraph":
            body.append(_paragraph(_run(payload, size=10), 0, 100))
        elif kind == "bullet":
            body.append(_paragraph(_run("•  " + str(payload), size=10), 0, 40, 340))
        elif kind == "bullets":
            for item in (payload or []):
                body.append(_paragraph(_run("•  " + str(item), size=10), 0, 40, 340))
        elif kind == "table":
            frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
            if caption:
                body.append(_paragraph(_run(caption, bold=True, size=9.5), 120, 40))
            body.append(_table_xml(frame))
        elif kind == "image":
            data = payload or b""
            if not data:
                continue
            picture_number += 1
            name = f"image{picture_number}.png"
            relationship_id = f"rIdImage{picture_number}"
            media.append((f"word/media/{name}", data))
            relationships.append(
                '<Relationship Id="%s" Type="http://schemas.openxmlformats.org/'
                'officeDocument/2006/relationships/image" Target="media/%s"/>'
                % (relationship_id, name))
            body.append(_image_xml(relationship_id, data, picture_number))
            if caption:
                body.append(_paragraph(_run(caption, italic=True, size=9, colour="595959"),
                                       0, 160, 0, "center"))
        elif kind == "pagebreak":
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    section = ('<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
               '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"'
               ' w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>')
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
        ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f"<w:body>{''.join(body)}{section}</w:body></w:document>")

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
        'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')

    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')

    document_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships) + "</Relationships>")

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", document_relationships)
        for name, data in media:
            archive.writestr(name, data)
    return path


def verify_docx(path: str) -> dict:
    """Read a written ``.docx`` back and check the parts Word insists on.

    Word reports a mangled file as "the file is corrupt", which says nothing about what is
    wrong.  This is the check that catches a broken relationship or an unclosed tag before
    the file is emailed to anyone.
    """
    import xml.etree.ElementTree as ElementTree

    report = {"path": path, "ok": True, "parts": [], "problems": []}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        report["parts"] = sorted(names)
        for required in ("[Content_Types].xml", "_rels/.rels", "word/document.xml",
                         "word/_rels/document.xml.rels"):
            if required not in names:
                report["ok"] = False
                report["problems"].append(f"missing part: {required}")
        for name in names:
            if name.endswith(".xml") or name.endswith(".rels"):
                try:
                    ElementTree.fromstring(archive.read(name))
                except ElementTree.ParseError as exc:
                    report["ok"] = False
                    report["problems"].append(f"{name} is not valid XML: {exc}")
        if "word/document.xml" in names:
            document = archive.read("word/document.xml").decode("utf-8")
            relationships = archive.read("word/_rels/document.xml.rels").decode("utf-8")
            for match in re.finditer(r'r:embed="([^"]+)"', document):
                if f'Id="{match.group(1)}"' not in relationships:
                    report["ok"] = False
                    report["problems"].append(
                        f"the document points at {match.group(1)}, which no relationship defines")
            for match in re.finditer(r'Target="media/([^"]+)"', relationships):
                if f"word/media/{match.group(1)}" not in names:
                    report["ok"] = False
                    report["problems"].append(f"missing image: word/media/{match.group(1)}")
    return report


def read_excel_sheets(path: str) -> list[str]:
    """The sheet names in a workbook — used by the self-test to confirm a write worked."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()
