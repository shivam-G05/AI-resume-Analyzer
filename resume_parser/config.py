import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pytesseract

load_dotenv(Path(__file__).with_name(".env"), override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

NEO_CONNECTION_URI = os.getenv("NEO_CONNECTION_URI")
NEO_USERNAME = os.getenv("NEO_USERNAME")
NEO_PASSWORD = os.getenv("NEO_PASSWORD")

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1").strip()

_openai_client: OpenAI | None = None

_tesseract_cmd = (os.getenv("TESSERACT_CMD") or "").strip()
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def get_openai_client() -> OpenAI:
    global _openai_client
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in resume_parser/.env."
        )
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client
