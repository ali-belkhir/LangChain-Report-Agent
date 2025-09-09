# schemas.py
from pydantic import BaseModel, Field
from typing import List

class PriceEvolution(BaseModel):
    energy: str = Field(description="Type of energy (e.g., Oil, Natural Gas, Power)")
    price: float = Field(description="Current price of the energy")
    evolution: str = Field(description="Evolution percentage compared to previous period")

class ReportSummary(BaseModel):
    title: str = Field(description="Title of the report")
    date: str = Field(description="Date of the report")
    description: str = Field(description="Textual description/summary of energy price evolution")
    prices: List[PriceEvolution] = Field(description="Table of prices and evolutions for different energies")

class ReportOutput(BaseModel):
    url: str = Field(description="The source URL")
    report_summary: ReportSummary = Field(description="Structured summary of the extracted content")
    word_file_path: str = Field(description="Path to the generated Word (.docx) file")
