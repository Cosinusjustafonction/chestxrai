"""
Guideline Tool — Clinical guideline lookup for detected pathologies.
Simple JSON-based retrieval. No RAG needed — 14 pathologies fit in memory.
"""

import os
import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# ── Load guidelines once ────────────────────────────────────────
GUIDELINES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "guidelines.json"
)

_guidelines = None

def _get_guidelines():
    global _guidelines
    if _guidelines is None:
        with open(GUIDELINES_PATH, "r") as f:
            _guidelines = json.load(f)["pathologies"]
        print(f"[GuidelineTool] Loaded {len(_guidelines)} pathology guidelines")
    return _guidelines


# ── Core lookup ─────────────────────────────────────────────────
def lookup_guideline(pathology: str) -> dict:
    """Look up clinical guidelines for a specific pathology."""
    guidelines = _get_guidelines()

    # Normalize: handle underscores, case
    key = pathology.strip().replace(" ", "_")

    # Try exact match first
    if key in guidelines:
        result = guidelines[key].copy()
        result["pathology"] = key
        result["found"] = True
        return result

    # Try case-insensitive
    for k, v in guidelines.items():
        if k.lower() == key.lower():
            result = v.copy()
            result["pathology"] = k
            result["found"] = True
            return result

    return {
        "pathology": pathology,
        "found": False,
        "message": f"No guideline found for '{pathology}'. "
                   f"Available: {', '.join(guidelines.keys())}"
    }


# ── CrewAI Tool ─────────────────────────────────────────────────
class GuidelineToolInput(BaseModel):
    pathology: str = Field(
        ...,
        description="Name of the pathology to look up (e.g., 'Cardiomegaly', 'Pneumothorax')"
    )

class GuidelineTool(BaseTool):
    name: str = "guideline_lookup"
    description: str = (
        "Look up clinical guidelines for a specific thoracic pathology. "
        "Returns the clinical definition, severity grading criteria, "
        "recommended follow-up actions, urgency classification, "
        "differential diagnoses, and known interactions with other pathologies. "
        "Input: the name of a detected pathology (e.g., 'Cardiomegaly')."
    )
    args_schema: Type[BaseModel] = GuidelineToolInput

    def _run(self, pathology: str) -> str:
        result = lookup_guideline(pathology)
        return json.dumps(result, indent=2)