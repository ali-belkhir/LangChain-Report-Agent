# tools.py
import requests
from bs4 import BeautifulSoup
from docx import Document
import os
import uuid
from langchain_core.tools import tool
from datetime import datetime
from typing import List
from schemas import ReportSummary  # Import your schema classes


# ------------------- Tools -------------------

@tool
def get_website_content(url: str) -> str:
    """Fetches the main text content from a given URL as plain text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script/style
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        return soup.get_text(separator=' ', strip=True)
    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve content from {url}. Reason: {e}"


@tool
def generate_word_report(summary: ReportSummary, output_dir: str = "reports") -> str:
    """
    Generate a structured Word report from a ReportSummary object.
    Returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"report_{uuid.uuid4().hex[:8]}.docx"
    file_path = os.path.join(output_dir, file_name)

    doc = Document()

    # Title
    doc.add_heading(summary.title, level=1)

    # Date
    doc.add_paragraph(f"Date: {summary.date}")

    # Description
    doc.add_heading("Summary", level=2)
    doc.add_paragraph(summary.description)

    # Table of energy prices
    if summary.prices:
        doc.add_heading("Energy Prices Evolution", level=2)
        table = doc.add_table(rows=1, cols=3)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Energy"
        hdr_cells[1].text = "Price"
        hdr_cells[2].text = "Evolution"

        for p in summary.prices:
            row_cells = table.add_row().cells
            row_cells[0].text = p.energy
            row_cells[1].text = str(p.price)
            row_cells[2].text = p.evolution

    doc.save(file_path)
    return file_path


# ------------------- Export list -------------------
tools = [get_website_content, generate_word_report]
