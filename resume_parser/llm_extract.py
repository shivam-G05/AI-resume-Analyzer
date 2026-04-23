import json
import io
from .config import LLM_MODEL, get_openai_client
from .schemas import ParsedResume

def extract_text_from_bytes(data: bytes, file_type: str) -> str:
    """Convert raw bytes to plain text based on file type."""
    
    if file_type == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text=[]
        for page in reader.pages:
            content=page.extract_text()
            if content.strip():
                text.append(content)
        return "\n".join(text)

    elif file_type in ("docx",):
        import docx
        doc = docx.Document(io.BytesIO(data))
        text=[]
        for paragraph in doc.paragraphs:
            content=paragraph.text
            if content.strip():
                text.append(content)
        return "\n".join(text)

    elif file_type in ("jpg", "jpeg", "png", "webp", "gif"):
        from PIL import Image
        import pytesseract
        img = Image.open(io.BytesIO(data))
        text=[]
        for img in images:
            content=pytesseract.image_to_string(img)
            if content.strip():
                text.append(content)
        return "\n".join(text)

    elif file_type == "txt":
        return data.decode("utf-8", errors="replace")

    else:
        raise ValueError(f"Unsupported file type: '{file_type}'")


EXTRACTION_SYSTEM_PROMPT = """
You are a precise resume parsing assistant.
Extract ALL information from the resume text and return a single valid JSON object.
Also generate a small query parameter for the job searches based on the resume data.
for example 
-if the resume data contains "Python, JavaScript, React" then the query_parameter should be query
should be  "software engineer jobs in india".
-if the resume data contains "Java, Spring Boot, Hibernate" then the query_parameter should be query
should be  "java developer jobs in india".
-if the resume data contains "React, Node.js, MongoDB" then the query_parameter should be query
should be  "full stack developer jobs in india".
-if the resume data contains "Python, Django, PostgreSQL" then the query_parameter should be query
should be  "django developer jobs in india".
-if the resume data contains "React, Node.js, MongoDB" then the query_parameter should be query
should be  "full stack developer jobs in india".
The JSON must include these standard fields:
{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "location": "string",
  "summary": "string (professional summary or objective; empty string if absent)",
  "skills": ["string", ...],
  "work_experience": [
    {
      "company": "string",
      "role": "string",
      "start_date": "string",
      "end_date": "string or null",
      "bullets": ["string", ...]
    }
  ] or null,
  "education": [
    {
      "institution": "string",
      "degree": "string",
      "field": "string or null",
      "graduation_year": "string or null"
    }
  ],
  "projects": [
    {
      "name": "string",
      "role": "string or null",
      "description": "string",
      "technologies": ["string", ...],
      "bullets": ["string", ...],
      "link": "string or null"
    }
  ] or null,
  "certifications": ["string", ...] or null,
  "languages": ["string", ...] or null,
  "achievements": [
    {
      "title": "string",
      "description": "string or null"
    }
  ] or null
  "query_parameter": "string"
}

IMPORTANT — Extra sections:
- If the resume contains ANY section not listed above (e.g. "Volunteer Work",
  "Publications", "Awards", "Hobbies", "References", "Competitions", etc.),
  include it as an additional key in the JSON using snake_case naming.
- For list-like sections use an array of strings or objects as appropriate.
- Example: a "Volunteer Work" section becomes "volunteer_work": [...]

Rules:
- Output ONLY valid JSON. No markdown fences, no explanation, no preamble.
- If a standard field is missing, use "" for strings, [] for lists, null for optional fields.
- Extract ALL skills mentioned anywhere in the resume. Deduplicate them.
"""




def extract_resume_with_llm(source: str | bytes, file_type: str = None) -> ParsedResume:
    """Accept either raw text string OR bytes with a file_type."""
    
    if isinstance(source, bytes):
        if not file_type:
            raise ValueError("file_type is required when passing bytes.")
        raw_text = extract_text_from_bytes(source, file_type)
    else:
        raw_text = source  # existing behaviour — plain text passed directly, unchanged

    # everything below is your original code, untouched
    response = get_openai_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this resume:\n\n{raw_text[:12000]}"},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty model response (no message content).")
    raw_json = content.strip()

    if raw_json.startswith("```"):
        parts = raw_json.split("```")
        raw_json = parts[1] if len(parts) > 1 else parts[0]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    data = json.loads(raw_json)
    return ParsedResume(**data, raw_text=raw_text)