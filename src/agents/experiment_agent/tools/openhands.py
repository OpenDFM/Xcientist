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
    def _trusted_mounts(workspace_root: str) -> tuple[str, ...]:
        workspace_abs = os.path.abspath(os.path.expanduser(workspace_root))
        return (os.path.join(workspace_abs, "model_candidate", "model_share"),)

    @staticmethod
    def validate_path(path: str, workspace_root: str) -> bool:
        if not path:
            return False
        try:
            abs_path = os.path.abspath(os.path.expanduser(path))
            real_path = os.path.realpath(abs_path)
            abs_workspace = os.path.realpath(
                os.path.abspath(os.path.expanduser(workspace_root))
            )
        except Exception:
            return False
        if real_path.startswith(abs_workspace + os.sep) or real_path == abs_workspace:
            return True
        for mount_root in SecurityValidator._trusted_mounts(workspace_root):
            mount_abs = os.path.abspath(os.path.expanduser(mount_root))
            if abs_path.startswith(mount_abs + os.sep) or abs_path == mount_abs:
                return True
        return False

    @staticmethod
    def validate_or_raise(path: str, workspace_root: str, operation: str = "access") -> None:
        if not SecurityValidator.validate_path(path, workspace_root):
            raise SecurityError(
                f"Cannot {operation} path '{path}': outside allowed workspace '{workspace_root}'"
            )

    @staticmethod
    def resolve_path(path: str, workspace_root: str, project_root: str | None = None) -> str:
        if os.path.isabs(path):
            full_path = os.path.abspath(os.path.expanduser(path))
        else:
            if workspace_root:
                full_path = os.path.abspath(os.path.join(workspace_root, path))
            elif project_root:
                full_path = os.path.abspath(os.path.join(project_root, path))
            else:
                full_path = os.path.abspath(path)

        SecurityValidator.validate_or_raise(full_path, workspace_root, "access")
        return full_path


__all__ = [
    "SecurityContext",
    "SecurityError",
    "SecurityValidator",
]
