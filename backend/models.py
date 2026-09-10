from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, validator


class Source(BaseModel):
    page: int = Field(..., ge=1)
    text: str
    pdf_name: str = ""


class Message(BaseModel):
    role: str
    content: str
    sources: list[Source] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    name: str
    pages: int = Field(..., ge=0)
    chunks: int = Field(..., ge=0)
    path: str | None = None
    uploaded_at: str | None = None


class Thread(BaseModel):
    id: str
    name: str
    pdf_name: str | None = None
    document: DocumentInfo | None = None
    messages: list[Message] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ThreadCreate(BaseModel):
    name: str | None = Field(default=None, max_length=80)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)

    @validator("question")
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    message: Message


class TranscriptionResponse(BaseModel):
    text: str
