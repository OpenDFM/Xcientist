"""Security helpers for experiment-agent path validation."""

from __future__ import annotations

import os


class SecurityError(Exception):
    """Raised when path is outside the allowed workspace."""


class SecurityContext:
    """Stores workspace roots used by lightweight path validation helpers."""

    _project_root: str = ""
    _workspace_root: str = ""

    @classmethod
    def set_roots(cls, project_root: str, workspace_root: str | None = None) -> None:
        cls._project_root = os.path.realpath(os.path.abspath(project_root))
        cls._workspace_root = os.path.realpath(
            os.path.abspath(workspace_root or project_root)
        )

    @classmethod
    def get_project_root(cls) -> str:
        return cls._project_root

    @classmethod
    def get_workspace_root(cls) -> str:
        return cls._workspace_root


class SecurityValidator:
    """Validates paths stay within the declared workspace root."""

    @staticmethod
    def validate_path(path: str, workspace_root: str) -> bool:
        if not path:
            return False
        try:
            abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
            abs_workspace = os.path.realpath(
                os.path.abspath(os.path.expanduser(workspace_root))
            )
        except Exception:
            return False
        return abs_path.startswith(abs_workspace + os.sep) or abs_path == abs_workspace

    @staticmethod
    def validate_or_raise(path: str, workspace_root: str, operation: str = "access") -> None:
        if not SecurityValidator.validate_path(path, workspace_root):
            raise SecurityError(
                f"Cannot {operation} path '{path}': outside allowed workspace '{workspace_root}'"
            )

    @staticmethod
    def resolve_path(path: str, workspace_root: str, project_root: str | None = None) -> str:
        if os.path.isabs(path):
            full_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
        else:
            if workspace_root:
                full_path = os.path.realpath(
                    os.path.abspath(os.path.join(workspace_root, path))
                )
            elif project_root:
                full_path = os.path.realpath(
                    os.path.abspath(os.path.join(project_root, path))
                )
            else:
                full_path = os.path.realpath(os.path.abspath(path))

        SecurityValidator.validate_or_raise(full_path, workspace_root, "access")
        return full_path


__all__ = [
    "SecurityContext",
    "SecurityError",
    "SecurityValidator",
]
