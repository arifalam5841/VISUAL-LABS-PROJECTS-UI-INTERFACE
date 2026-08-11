import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


def report_generator_agent(state):

    summary = state["summary"]
    query = state["query"]
    academic_results = state["academic_results"]
    web_results = state["web_results"]

    print("\n📄 Report Generator Agent is creating the report...")

    # --------------------------------------------------
    # REPORT DIRECTORY
    # --------------------------------------------------

    report_directory = Path("reports")

    report_directory.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------
    # SAFE FILE NAME
    # --------------------------------------------------

    safe_topic = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        query
    ).strip("_")

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_filename = (
        f"{safe_topic}_{timestamp}.pdf"
    )

    report_path = (
        report_directory /
        report_filename
    )

    # --------------------------------------------------
    # PDF DOCUMENT
    # --------------------------------------------------

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = styles["Heading2"]

    body_style = styles["BodyText"]

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "AI Research Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Research Topic:</b> "
            f"{escape(query)}",
            body_style
        )
    )

    story.append(
        Spacer(1, 0.25 * inch)
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    sections = summary.split("\n")

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # Remove Markdown bold markers
        section = section.replace("**", "")

        # Check whether the line is a heading
        if re.match(
            r"^[1-9]\.\s+",
            section
        ):

            story.append(
                Paragraph(
                    escape(section),
                    heading_style
                )
            )

        else:

            story.append(
                Paragraph(
                    escape(section),
                    body_style
                )
            )

        story.append(
            Spacer(1, 8)
        )

    # --------------------------------------------------
    # REFERENCES
    # --------------------------------------------------

    story.append(
        Paragraph(
            "References",
            heading_style
        )
    )

    story.append(
        Spacer(1, 8)
    )

    reference_number = 1

    # --------------------------------------------------
    # ACADEMIC REFERENCES
    # --------------------------------------------------

    for paper in academic_results:

        title = paper.get(
            "title",
            "Untitled Paper"
        )

        url = paper.get(
            "url",
            ""
        )

        reference = (
            f"{reference_number}. "
            f"{title} - {url}"
        )

        story.append(
            Paragraph(
                escape(reference),
                body_style
            )
        )

        story.append(
            Spacer(1, 6)
        )

        reference_number += 1

    # --------------------------------------------------
    # WEB REFERENCES
    # --------------------------------------------------

    for result in web_results:

        title = result.get(
            "title",
            "Untitled Web Source"
        )

        url = result.get(
            "url",
            ""
        )

        reference = (
            f"{reference_number}. "
            f"{title} - {url}"
        )

        story.append(
            Paragraph(
                escape(reference),
                body_style
            )
        )

        story.append(
            Spacer(1, 6)
        )

        reference_number += 1

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(story)

    print(
        "\n✅ Report generated successfully:"
        f"\n{report_path}"
    )

    return {
        "final_report": str(report_path)
    }