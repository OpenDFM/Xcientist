from typing import List, Dict
from pydantic import BaseModel, Field


class Idea(BaseModel):
    title: str = Field(description="Title of the research proposal")
    description: str = Field(description="High-level description or abstract")
    key_innovations: List[str] = Field(description="List of key innovations")
    methodology: Dict[str, str] = Field(description="Methodology sections")
    expected_outcomes: List[str] = Field(description="List of expected outcomes")


class Proposal(BaseModel):
    idea: Idea = Field(description="The core idea details")
    reference_papers: List[str] = Field(description="List of reference papers")
