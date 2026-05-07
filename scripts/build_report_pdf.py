from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


REPORT_MD = Path("reports") / "oil_news_database_model_report.md"
OUTPUT_PDF = Path("output") / "pdf" / "oil_news_database_model_report.pdf"


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def parse_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    idx = start
    while idx < len(lines) and lines[idx].strip().startswith("|"):
        raw = lines[idx].strip().strip("|")
        cells = [cell.strip() for cell in raw.split("|")]
        if not all(set(cell) <= {"-", ":", " "} for cell in cells):
            rows.append(cells)
        idx += 1
    return rows, idx


def build_story(markdown: str) -> list:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24))
    styles.add(ParagraphStyle("H1Custom", parent=styles["Heading1"], fontSize=15, leading=19, spaceBefore=12))
    styles.add(ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10))
    styles.add(ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=9.3, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle("BulletCustom", parent=styles["BodyText"], fontSize=9.3, leading=12, leftIndent=14, firstLineIndent=-8))
    styles.add(ParagraphStyle("CodeCustom", parent=styles["Code"], fontName="Courier", fontSize=7.8, leading=9.5))

    story = []
    lines = markdown.splitlines()
    idx = 0
    first_title = True
    in_code = False
    code_lines: list[str] = []

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["CodeCustom"]))
                story.append(Spacer(1, 0.08 * inch))
                code_lines = []
                in_code = False
            else:
                in_code = True
            idx += 1
            continue

        if in_code:
            code_lines.append(line)
            idx += 1
            continue

        if not stripped:
            idx += 1
            continue

        if stripped.startswith("|"):
            rows, idx = parse_markdown_table(lines, idx)
            if rows:
                table_data = [[Paragraph(clean_inline(cell), styles["BodyCustom"]) for cell in row] for row in rows]
                table = Table(table_data, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF6")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C7D0DD")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.12 * inch))
            continue

        if stripped.startswith("# "):
            if first_title:
                story.append(Paragraph(clean_inline(stripped[2:]), styles["TitleCenter"]))
                story.append(Spacer(1, 0.18 * inch))
                first_title = False
            else:
                story.append(PageBreak())
                story.append(Paragraph(clean_inline(stripped[2:]), styles["H1Custom"]))
            idx += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(clean_inline(stripped[3:]), styles["H1Custom"]))
            idx += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(clean_inline(stripped[4:]), styles["H2Custom"]))
            idx += 1
            continue

        if stripped.startswith("- "):
            story.append(Paragraph("• " + clean_inline(stripped[2:]), styles["BulletCustom"]))
            idx += 1
            continue

        story.append(Paragraph(clean_inline(stripped), styles["BodyCustom"]))
        idx += 1

    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5B6472"))
    canvas.drawString(0.72 * inch, 0.45 * inch, "Semantic News and Oil Price Database Project")
    canvas.drawRightString(7.78 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    markdown = REPORT_MD.read_text(encoding="utf-8")
    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Semantic News and Oil Price Database Project Report",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])
    doc.build(build_story(markdown))
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
