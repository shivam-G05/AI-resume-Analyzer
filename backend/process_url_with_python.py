from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from resume_parser.pipeline import parse_resume_file
from resume_parser.config import LLM_MODEL, get_openai_client
from resume_parser.neo4j_store import get_neo4j_driver, store_in_neo4j  # ← add


def _run_llm_json_prompt(system_prompt: str, resume_data: dict) -> dict:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Resume data (JSON):\n" + json.dumps(resume_data, indent=2)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    raw = content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


ATS_SYSTEM_PROMPT = """
You are an ATS (Applicant Tracking System) compatibility expert.
Analyse the provided resume data and evaluate how well it will pass through ATS systems used by recruiters.
Return ONLY a valid JSON object with this exact structure:
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
Return ONLY a valid JSON object with this exact structure:
{
  "score": <float 0-100>,
  "inferred_target_role": "string",
  "matched_skills": ["skill", ...],
  "missing_critical_skills": ["skill", ...],
  "missing_nice_to_have": ["skill", ...],
  "outdated_skills": ["skill", ...],
  "recommendations": ["actionable recommendation", ...],
  "raw_feedback": "2-3 sentence overall skills gap assessment."
}
"""


def main() -> None:
    try:
        if len(sys.argv) < 2:
            raise ValueError("File path argument is required.")

        file_path = sys.argv[1].strip()

        if not Path(file_path).exists():
            raise ValueError(f"File not found: {file_path}")

        # Step 1 — parse resume into structured data
        parsed = parse_resume_file(file_path)
        resume_id = str(uuid.uuid4())
        resume_data = parsed.model_dump(mode="json", exclude={"raw_text"})

        # Step 2 — store in Neo4j (non-blocking, won't crash the API if it fails)
        neo4j_uri = __import__('os').environ.get("NEO4J_URI", "").strip()
        if neo4j_uri:
            try:
                driver = get_neo4j_driver()
                store_in_neo4j(driver, resume_id, parsed)
                driver.close()
            except Exception as neo4j_err:
                print(f"Warning: Neo4j storage failed: {neo4j_err}", file=sys.stderr)

        # Step 3 — run all 4 analysis prompts
        ats_check        = _run_llm_json_prompt(ATS_SYSTEM_PROMPT, resume_data)
        content_quality  = _run_llm_json_prompt(CONTENT_QUALITY_SYSTEM_PROMPT, resume_data)
        format_structure = _run_llm_json_prompt(FORMAT_STRUCTURE_SYSTEM_PROMPT, resume_data)
        skill_gap        = _run_llm_json_prompt(SKILL_GAP_SYSTEM_PROMPT, resume_data)

        # Step 4 — return everything as one JSON object
        result = {
            "resume_id":        resume_id,
            "parsed_resume":    resume_data,
            "ats_check":        ats_check,
            "content_quality":  content_quality,
            "format_structure": format_structure,
            "skill_gap":        skill_gap,
        }

        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()