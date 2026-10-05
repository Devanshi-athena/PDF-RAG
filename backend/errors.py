from __future__ import annotations

from fastapi import HTTPException


class BackendError(Exception):
    """An expected, user-facing service error."""

    def __init__(self, message: str, status_code: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class NotFoundError(BackendError):
    def __init__(self, message: str = "Thread not found."):
        super().__init__(message, 404, "not_found")


def http_error(error: BackendError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message},
    )
