from typing import Any, Optional
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: Optional[str] = None
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    field: Optional[str] = None
    graduation_year: Optional[str] = None


class Project(BaseModel):
    name: str
    role: Optional[str] = None
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    link: Optional[str] = None


class Achievement(BaseModel):
    title: str
    description: Optional[str] = None


class ParsedResume(BaseModel):
    model_config = {"extra": "allow"}   # ← allows any extra fields the LLM finds

    full_name: str
    email: str
    phone: str
    location: str
    summary: str
    skills: list[str]
    work_experience: Optional[list[WorkExperience]] = None
    education: list[Education]
    projects: Optional[list[Project]] = None
    certifications: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    achievements: Optional[list[Achievement]] = None   # ← new
    raw_text: Optional[str] = None