from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

_REPO_ROOT = Path(__file__).resolve().parent
_RESUME_PARSER_DIR = _REPO_ROOT / "resume_parser"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_RESUME_PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(_RESUME_PARSER_DIR))

from resume_parser.config import LLM_MODEL, get_openai_client
from resume_parser.neo4j_store import get_neo4j_driver, store_in_neo4j
from resume_parser.pipeline import parse_resume_file

load_dotenv()


class GraphState(TypedDict):
    resume_path: str
    resume_id: str | None
    resume_data: dict[str, Any] | None
    ATS_check: str | None
    content_quality: str | None
    format_structure: str | None
    skill_analysis: str | None


def _run_llm_json_prompt(system_prompt: str, resume_data: dict[str, Any]) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Resume data (JSON):\n" + json.dumps(resume_data, indent=2)},
        ],
    )
    return response.choices[0].message.content or "{}"


ATS_SYSTEM_PROMPT = """
You are an ATS (Applicant Tracking System) compatibility expert.
Analyse the provided resume data and evaluate how well it will pass through ATS systems used by recruiters.

You must return ONLY a valid JSON object — no markdown, no explanation — with this exact structure:
{
  "score": <float 0-100>,
  "keyword_match_rate": <float 0.0-1.0>,
  "missing_keywords": ["keyword1", ...],
  "found_keywords": ["keyword1", ...],
  "ats_friendly_sections": ["section_name", ...],
  "ats_issues": ["issue description", ...],
  "recommendations": ["actionable recommendation", ...],
  "raw_feedback": "A 2-3 sentence overall assessment."
}
"""

CONTENT_QUALITY_SYSTEM_PROMPT = """
You are an expert resume content quality reviewer and professional writing coach.
Evaluate the quality of the resume content — specifically bullet points, summary, and overall writing strength.

Return ONLY a valid JSON object with this exact structure:
{
  "score": <float 0-100>,
  "bullet_quality_score": <float 0-100>,
  "summary_score": <float 0-100>,
  "impact_score": <float 0-100>,
  "weak_bullets": ["copy of weak bullet text", ...],
  "strong_bullets": ["copy of strong bullet text", ...],
  "issues": ["specific issue found", ...],
  "recommendations": ["actionable improvement", ...],
  "raw_feedback": "2-3 sentence overall assessment of writing quality."
}
"""

FORMAT_STRUCTURE_SYSTEM_PROMPT = """
You are a professional resume formatting and structure expert.
Evaluate the structural completeness and organisation of the resume.

Return ONLY a valid JSON object with this exact structure:
{
  "score": <float 0-100>,
  "section_completeness_score": <float 0-100>,
  "length_score": <float 0-100>,
  "consistency_score": <float 0-100>,
  "missing_sections": ["section name", ...],
  "present_sections": ["section name", ...],
  "issues": ["specific structural issue", ...],
  "recommendations": ["actionable fix", ...],
  "raw_feedback": "2-3 sentence overall structural assessment."
}
"""

SKILL_GAP_SYSTEM_PROMPT = """
You are a senior technical recruiter and skills market analyst.
Analyse the candidate's skill set against the requirements of their most likely target role.
Identify critical gaps and opportunities.

Return ONLY a valid JSON object with this exact structure:
{
  "score": <float 0-100>,
  "inferred_target_role": "string (e.g. Senior Backend Engineer, Data Scientist)",
  "matched_skills": ["skill", ...],
  "missing_critical_skills": ["skill", ...],
  "missing_nice_to_have": ["skill", ...],
  "outdated_skills": ["skill", ...],
  "recommendations": ["actionable recommendation to close a gap", ...],
  "raw_feedback": "2-3 sentence overall skills gap assessment."
}
"""


def extract_and_store_resume_node(state: GraphState) -> GraphState:
    resume_path = state["resume_path"]
    parsed = parse_resume_file(resume_path)
    resume_id = str(uuid.uuid4())

    try:
        neo4j_driver = get_neo4j_driver()
        store_in_neo4j(neo4j_driver, resume_id, parsed)
        neo4j_driver.close()
    except Exception as exc:
        print(f"Warning: Neo4j storage failed: {exc}")

    state["resume_id"] = resume_id
    state["resume_data"] = parsed.model_dump(mode="json", exclude={"raw_text"})
    return state


def ats_check_node(state: GraphState) -> GraphState:
    resume_data = state.get("resume_data") or {}
    state["ATS_check"] = _run_llm_json_prompt(ATS_SYSTEM_PROMPT, resume_data)
    return state


def content_quality_node(state: GraphState) -> GraphState:
    resume_data = state.get("resume_data") or {}
    state["content_quality"] = _run_llm_json_prompt(CONTENT_QUALITY_SYSTEM_PROMPT, resume_data)
    return state


def format_structure_node(state: GraphState) -> GraphState:
    resume_data = state.get("resume_data") or {}
    state["format_structure"] = _run_llm_json_prompt(FORMAT_STRUCTURE_SYSTEM_PROMPT, resume_data)
    return state


def skill_gap_analysis_node(state: GraphState) -> GraphState:
    resume_data = state.get("resume_data") or {}
    state["skill_analysis"] = _run_llm_json_prompt(SKILL_GAP_SYSTEM_PROMPT, resume_data)
    return state


def build_graph():
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("extract_resume", extract_and_store_resume_node)
    graph_builder.add_node("ats_check", ats_check_node)
    graph_builder.add_node("content_quality", content_quality_node)
    graph_builder.add_node("format_structure", format_structure_node)
    graph_builder.add_node("skill_gap_analysis", skill_gap_analysis_node)

    graph_builder.add_edge(START, "extract_resume")
    graph_builder.add_edge("extract_resume", "ats_check")
    graph_builder.add_edge("ats_check", "content_quality")
    graph_builder.add_edge("content_quality", "format_structure")
    graph_builder.add_edge("format_structure", "skill_gap_analysis")
    graph_builder.add_edge("skill_gap_analysis", END)
    return graph_builder.compile()


def _pretty_print_json_text(label: str, value: str | None) -> None:
    print(f"\n--- {label} ---")
    if not value:
        print("null")
        return
    try:
        parsed = json.loads(value)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        pprint(value, width=100, sort_dicts=False)

def main() -> None:
    resume_path = os.environ.get("RESUME_PATH", "").strip()

    if not resume_path:
        resume_path = input("Enter path to resume file: ").strip().strip('"').strip("'")

    if not resume_path:
        raise ValueError("No resume path provided.")
    if not Path(resume_path).exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    initial_state: GraphState = {
        "resume_path": resume_path,
        "resume_id": None,
        "resume_data": None,
        "ATS_check": None,
        "content_quality": None,
        "format_structure": None,
        "skill_analysis": None,
    }

    graph = build_graph()
    result = graph.invoke(initial_state)

    print("\n========== Pipeline Result ==========")
    print(f"resume_id: {result.get('resume_id')}")
    _pretty_print_json_text("ATS Check", result.get("ATS_check"))
    _pretty_print_json_text("Content Quality", result.get("content_quality"))
    _pretty_print_json_text("Format Structure", result.get("format_structure"))
    _pretty_print_json_text("Skill Gap Analysis", result.get("skill_analysis"))


if __name__ == "__main__":
    main()
