"""Resource discovery and download tools for prepare-stage workers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

from src.agents.experiment_agent.config import get_api_config

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation


def _json_request(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="replace")
    return json.loads(body)


def _run_subprocess(command: list[str], *, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _project_python(workspace_root: str) -> str:
    project_python = os.path.join(workspace_root, "project", "venv", "bin", "python")
    if os.path.exists(project_python):
        return project_python
    return os.environ.get("PYTHON", "python3")


def _hf_endpoint() -> str:
    endpoint = str(get_api_config().get("huggingface_endpoint") or "https://huggingface.co")
    return endpoint.rstrip("/")


def _render_results(header: dict[str, Any], results: list[dict[str, Any]]) -> str:
    payload = dict(header)
    payload["results"] = results
    return json.dumps(payload, ensure_ascii=False, indent=2)


class ResourceSearchAction(Action):
    query: str = Field(description="Search query string.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum results to return.")
    search_type: str | None = Field(default=None, description="Provider-specific search type.")
    repo: str | None = Field(default=None, description="Optional repo qualifier for code search.")


class ResourceDownloadAction(Action):
    repo_id: str = Field(description="Resource identifier to download.")
    local_dir: str = Field(description="Absolute or workspace-relative destination directory.")
    repo_type: Literal["model", "dataset"] = Field(default="model")
    revision: str | None = Field(default=None)
    filename: str | None = Field(default=None, description="Optional single file to fetch.")


class ResourceObservation(Observation):
    provider: str = Field(description="The backend provider name.")
    resource_type: str | None = Field(default=None)
    raw_output_path: str | None = Field(default=None)

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        lines = [self.text]
        if self.raw_output_path:
            lines.append(f"[Raw output saved to: {self.raw_output_path}]")
        return [TextContent(text="\n".join(part for part in lines if part))]


class GitHubSearchExecutor(ToolExecutor[ResourceSearchAction, ResourceObservation]):
    def __call__(
        self,
        action: ResourceSearchAction,
        conversation: "LocalConversation | None" = None,
    ) -> ResourceObservation:
        token = (
            os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_AI_TOKEN")
            or str(get_api_config().get("github_ai_token") or "")
        )
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        search_type = (action.search_type or "repositories").strip().lower()
        if search_type == "code":
            query = action.query
            if action.repo:
                query = f"{query} repo:{action.repo}"
            url = (
                "https://api.github.com/search/code?"
                + urllib.parse.urlencode({"q": query, "per_page": action.top_k})
            )
            payload = _json_request(url, headers=headers)
            items = payload.get("items") or []
            results = [
                {
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "repo": ((item.get("repository") or {}).get("full_name")),
                    "url": item.get("html_url"),
                    "score": item.get("score"),
                }
                for item in items[: action.top_k]
            ]
        else:
            url = (
                "https://api.github.com/search/repositories?"
                + urllib.parse.urlencode({"q": action.query, "per_page": action.top_k})
            )
            payload = _json_request(url, headers=headers)
            items = payload.get("items") or []
            results = [
                {
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "url": item.get("html_url"),
                    "default_branch": item.get("default_branch"),
                    "stars": item.get("stargazers_count"),
                    "updated_at": item.get("updated_at"),
                }
                for item in items[: action.top_k]
            ]
        text = _render_results(
            {
                "provider": "github",
                "search_type": search_type,
                "query": action.query,
                "top_k": action.top_k,
            },
            results,
        )
        return ResourceObservation.from_text(text=text, provider="github", resource_type=search_type)


class HFHubSearchExecutor(ToolExecutor[ResourceSearchAction, ResourceObservation]):
    def __call__(
        self,
        action: ResourceSearchAction,
        conversation: "LocalConversation | None" = None,
    ) -> ResourceObservation:
        search_type = (action.search_type or "models").strip().lower()
        if search_type not in {"models", "datasets"}:
            search_type = "models"
        path = "models" if search_type == "models" else "datasets"
        url = (
            f"{_hf_endpoint()}/api/{path}?"
            + urllib.parse.urlencode({"search": action.query, "limit": action.top_k})
        )
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = _json_request(url, headers=headers)
        results: list[dict[str, Any]] = []
        for item in (payload or [])[: action.top_k]:
            results.append(
                {
                    "id": item.get("id") or item.get("modelId") or item.get("name"),
                    "downloads": item.get("downloads"),
                    "likes": item.get("likes"),
                    "pipeline_tag": item.get("pipeline_tag"),
                    "last_modified": item.get("lastModified"),
                    "private": item.get("private"),
                }
            )
        text = _render_results(
            {
                "provider": "huggingface",
                "search_type": search_type,
                "query": action.query,
                "top_k": action.top_k,
            },
            results,
        )
        return ResourceObservation.from_text(
            text=text, provider="huggingface", resource_type=search_type
        )


class HFHubDownloadExecutor(ToolExecutor[ResourceDownloadAction, ResourceObservation]):
    def __call__(
        self,
        action: ResourceDownloadAction,
        conversation: "LocalConversation | None" = None,
    ) -> ResourceObservation:
        conv_workspace = getattr(getattr(conversation, "state", None), "workspace", None)
        workspace_root = getattr(conv_workspace, "working_dir", None) or os.getcwd()
        local_dir = action.local_dir
        if not os.path.isabs(local_dir):
            local_dir = os.path.join(workspace_root, local_dir)
        os.makedirs(local_dir, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
        hf_cli = shutil.which("hf")
        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token
        if hf_cli:
            command = [hf_cli, "download", action.repo_id]
            if action.filename:
                command.append(action.filename)
            command.extend(["--repo-type", action.repo_type, "--local-dir", local_dir, "--max-workers", "1"])
            if action.revision:
                command.extend(["--revision", action.revision])
            if token:
                command.extend(["--token", token])
            code, stdout, stderr = _run_subprocess(command, env=env)
        else:
            python_bin = _project_python(workspace_root)
            py_code = (
                "from huggingface_hub import snapshot_download, hf_hub_download; "
                "import os, json; "
                f"repo_id={action.repo_id!r}; repo_type={action.repo_type!r}; local_dir={local_dir!r}; "
                f"revision={action.revision!r}; filename={action.filename!r}; token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN'); "
                "os.makedirs(local_dir, exist_ok=True); "
                "result = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, local_dir=local_dir, revision=revision, token=token) if filename else snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, revision=revision, token=token); "
                "print(json.dumps({'path': result}))"
            )
            code, stdout, stderr = _run_subprocess([python_bin, "-c", py_code], env=env)
        text = _render_results(
            {
                "provider": "huggingface",
                "resource_type": action.repo_type,
                "repo_id": action.repo_id,
                "local_dir": local_dir,
                "revision": action.revision,
                "filename": action.filename,
                "exit_code": code,
            },
            [{"stdout": stdout[-4000:], "stderr": stderr[-4000:]}],
        )
        return ResourceObservation.from_text(
            text=text,
            is_error=code != 0,
            provider="huggingface",
            resource_type=action.repo_type,
        )


class ModelScopeSearchExecutor(ToolExecutor[ResourceSearchAction, ResourceObservation]):
    def __call__(
        self,
        action: ResourceSearchAction,
        conversation: "LocalConversation | None" = None,
    ) -> ResourceObservation:
        search_type = (action.search_type or "models").strip().lower()
        path = "models" if search_type == "models" else "datasets"
        params = {"page_number": 1, "page_size": action.top_k, "name": action.query}
        url = f"https://www.modelscope.cn/api/v1/{path}?" + urllib.parse.urlencode(params)
        token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN") or ""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = _json_request(url, headers=headers)
        items = (
            payload.get("Data")
            or payload.get("data")
            or payload.get("Models")
            or payload.get("Dataset")
            or []
        )
        results = []
        for item in items[: action.top_k]:
            results.append(
                {
                    "id": item.get("Name") or item.get("name") or item.get("Id") or item.get("id"),
                    "path": item.get("Path") or item.get("path"),
                    "downloads": item.get("Downloads") or item.get("downloads"),
                    "likes": item.get("Likes") or item.get("likes"),
                }
            )
        text = _render_results(
            {
                "provider": "modelscope",
                "search_type": search_type,
                "query": action.query,
                "top_k": action.top_k,
            },
            results,
        )
        return ResourceObservation.from_text(
            text=text, provider="modelscope", resource_type=search_type
        )


class ModelScopeDownloadExecutor(ToolExecutor[ResourceDownloadAction, ResourceObservation]):
    def __call__(
        self,
        action: ResourceDownloadAction,
        conversation: "LocalConversation | None" = None,
    ) -> ResourceObservation:
        conv_workspace = getattr(getattr(conversation, "state", None), "workspace", None)
        workspace_root = getattr(conv_workspace, "working_dir", None) or os.getcwd()
        local_dir = action.local_dir
        if not os.path.isabs(local_dir):
            local_dir = os.path.join(workspace_root, local_dir)
        os.makedirs(local_dir, exist_ok=True)
        env = os.environ.copy()
        token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN") or ""
        if token:
            env["MODELSCOPE_API_TOKEN"] = token
            env["MODELSCOPE_TOKEN"] = token
        python_bin = _project_python(workspace_root)
        py_code = (
            "import os, json; "
            "from modelscope.hub.snapshot_download import snapshot_download; "
            f"repo_id={action.repo_id!r}; local_dir={local_dir!r}; revision={action.revision!r}; "
            "os.makedirs(local_dir, exist_ok=True); "
            "result = snapshot_download(model_id=repo_id, cache_dir=local_dir, revision=revision) if "
            f"{action.repo_type!r} == 'model' else snapshot_download(dataset_id=repo_id, cache_dir=local_dir, revision=revision); "
            "print(json.dumps({'path': result}))"
        )
        code, stdout, stderr = _run_subprocess([python_bin, "-c", py_code], env=env)
        text = _render_results(
            {
                "provider": "modelscope",
                "resource_type": action.repo_type,
                "repo_id": action.repo_id,
                "local_dir": local_dir,
                "revision": action.revision,
                "exit_code": code,
            },
            [{"stdout": stdout[-4000:], "stderr": stderr[-4000:]}],
        )
        return ResourceObservation.from_text(
            text=text,
            is_error=code != 0,
            provider="modelscope",
            resource_type=action.repo_type,
        )


class GitHubSearchTool(ToolDefinition[ResourceSearchAction, ResourceObservation]):
    name: ClassVar[str] = "github_search"

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["GitHubSearchTool"]:
        return [
            cls(
                action_type=ResourceSearchAction,
                observation_type=ResourceObservation,
                description="Search GitHub repositories or code snippets relevant to the current prepare-stage task.",
                annotations=ToolAnnotations(
                    title="github_search",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=GitHubSearchExecutor(),
            )
        ]


class HFHubSearchTool(ToolDefinition[ResourceSearchAction, ResourceObservation]):
    name: ClassVar[str] = "hf_hub_search"

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["HFHubSearchTool"]:
        return [
            cls(
                action_type=ResourceSearchAction,
                observation_type=ResourceObservation,
                description="Search HuggingFace model or dataset registries and return structured candidate results.",
                annotations=ToolAnnotations(
                    title="hf_hub_search",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=HFHubSearchExecutor(),
            )
        ]


class HFHubDownloadTool(ToolDefinition[ResourceDownloadAction, ResourceObservation]):
    name: ClassVar[str] = "hf_hub_download"

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["HFHubDownloadTool"]:
        return [
            cls(
                action_type=ResourceDownloadAction,
                observation_type=ResourceObservation,
                description="Download a HuggingFace model or dataset into a prepared local workspace directory. Uses HF_TOKEN from the environment when available.",
                annotations=ToolAnnotations(
                    title="hf_hub_download",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=HFHubDownloadExecutor(),
            )
        ]


class ModelScopeSearchTool(ToolDefinition[ResourceSearchAction, ResourceObservation]):
    name: ClassVar[str] = "modelscope_search"

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["ModelScopeSearchTool"]:
        return [
            cls(
                action_type=ResourceSearchAction,
                observation_type=ResourceObservation,
                description="Search ModelScope model or dataset registries and return structured candidate results.",
                annotations=ToolAnnotations(
                    title="modelscope_search",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=ModelScopeSearchExecutor(),
            )
        ]


class ModelScopeDownloadTool(ToolDefinition[ResourceDownloadAction, ResourceObservation]):
    name: ClassVar[str] = "modelscope_download"

    @classmethod
    def create(cls, conv_state: "ConversationState") -> Sequence["ModelScopeDownloadTool"]:
        return [
            cls(
                action_type=ResourceDownloadAction,
                observation_type=ResourceObservation,
                description="Download a ModelScope model or dataset into a prepared local workspace directory. Uses MODELSCOPE_API_TOKEN or MODELSCOPE_TOKEN when available.",
                annotations=ToolAnnotations(
                    title="modelscope_download",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=ModelScopeDownloadExecutor(),
            )
        ]


_RESOURCE_TOOLS_ENABLED = False


def enable_resource_tools() -> None:
    global _RESOURCE_TOOLS_ENABLED
    if _RESOURCE_TOOLS_ENABLED:
        return
    register_tool(GitHubSearchTool.name, GitHubSearchTool)
    register_tool(HFHubSearchTool.name, HFHubSearchTool)
    register_tool(HFHubDownloadTool.name, HFHubDownloadTool)
    register_tool(ModelScopeSearchTool.name, ModelScopeSearchTool)
    register_tool(ModelScopeDownloadTool.name, ModelScopeDownloadTool)
    _RESOURCE_TOOLS_ENABLED = True


__all__ = [
    "GitHubSearchTool",
    "HFHubSearchTool",
    "HFHubDownloadTool",
    "ModelScopeSearchTool",
    "ModelScopeDownloadTool",
    "enable_resource_tools",
]
