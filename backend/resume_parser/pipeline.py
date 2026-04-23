from pathlib import Path

from .llm_extract import extract_resume_with_llm
from .parsers.extract import extract_text_from_file
from .schemas import ParsedResume


def parse_resume_file(file_path: str) -> ParsedResume:
    """Parse a resume file into a ``ParsedResume`` (same pipeline as the CLI).

    Other modules can import this and call ``.model_dump(exclude={"raw_text"})``
    for a JSON-serializable dict.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If no text could be extracted or the LLM response is invalid.
        json.JSONDecodeError: If the model returns non-JSON content.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"file not found at '{path}'")
    raw_text = extract_text_from_file(str(path))
    if not raw_text.strip():
        raise ValueError("could not extract any text from the file.")
    return extract_resume_with_llm(raw_text)
