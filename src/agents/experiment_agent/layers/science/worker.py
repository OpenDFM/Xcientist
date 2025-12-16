"""
Experiment Worker Agent - Task Execution (LLM + Tools)

Goal:
- Delegate experiment execution to an agent (similar to CodeWorker).
- The worker agent is responsible for:
  - creating any needed config/data under result_dir
  - running the experiment command
  - handling errors (inspect logs, adjust inputs, re-run)
  - persisting artifacts (stdout/stderr/meta) into result_dir

The manager/integrator stay the same: they read result_dir artifacts for resume and analysis.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.science.schemas.experiment import (
    ExperimentResult,
    ExperimentTask,
    MetricSpec,
)
from src.agents.experiment_agent.shared.tools.core import get_worker_tools
from src.agents.experiment_agent.shared.utils.config import SCIENCE_WORKER_MODEL
from src.agents.experiment_agent.shared.utils.prompts import load_and_render_prompt


logger = logging.getLogger(__name__)


class ExpWorkerAgent(BaseAgent):
    """
    Experiment Worker (LLM + tools).

    Contract:
    - The worker SHOULD only create/edit files under task.result_dir (relative to project_root).
    - It should write the following orchestration artifacts into result_dir:
      - stdout_{task_id}.txt
      - stderr_{task_id}.txt
      - meta_{task_id}.json
      - metrics_{task_id}.json (optional; we also compute metrics from metric_specs as a fallback)
    """

    def __init__(
        self,
        model: str = SCIENCE_WORKER_MODEL,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="ExpWorker",
            model=model,
            max_turns=2000,
            verbose=verbose,
        )

    def _get_tools(self) -> List:
        return get_worker_tools()

    def _safe_task_id(self, task_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id or "task"))

    def _get_abs_result_dir(
        self, task: ExperimentTask, project_root: str
    ) -> Optional[str]:
        rd = str(getattr(task, "result_dir", "") or "").strip()
        if not rd:
            return None
        return os.path.join(project_root, rd)

    def _read_text(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _read_json(self, path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _persist_orchestration_outputs(
        self,
        task: ExperimentTask,
        project_root: str,
        stdout: str,
        stderr: str,
        metrics: Dict[str, float],
        success: bool,
        error: Optional[str],
        returncode: Optional[int] = None,
    ) -> None:
        abs_dir = self._get_abs_result_dir(task, project_root)
        if not abs_dir:
            return

        try:
            os.makedirs(abs_dir, exist_ok=True)
        except Exception:
            return

        safe_id = self._safe_task_id(task.task_id)
        stdout_path = os.path.join(abs_dir, f"stdout_{safe_id}.txt")
        stderr_path = os.path.join(abs_dir, f"stderr_{safe_id}.txt")
        metrics_path = os.path.join(abs_dir, f"metrics_{safe_id}.json")
        meta_path = os.path.join(abs_dir, f"meta_{safe_id}.json")

        try:
            with open(stdout_path, "w", encoding="utf-8") as f:
                f.write(stdout or "")
        except Exception:
            pass

        try:
            with open(stderr_path, "w", encoding="utf-8") as f:
                f.write(stderr or "")
        except Exception:
            pass

        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics or {}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        meta = {
            "task_id": str(task.task_id),
            "description": str(getattr(task, "description", "") or ""),
            "command": str(getattr(task, "command", "") or ""),
            "cwd": str(project_root),
            "success": bool(success),
            "returncode": returncode,
            "error": error,
            "expected_output_files": list(
                getattr(task, "expected_output_files", []) or []
            ),
            "metric_specs_count": len(getattr(task, "metric_specs", None) or []),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _collect_metrics(
        self, task: ExperimentTask, project_root: str
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        specs: List[MetricSpec] = getattr(task, "metric_specs", None) or []
        if not specs:
            return metrics

        for spec in specs:
            file_path = os.path.join(project_root, spec.file_path)
            if not os.path.exists(file_path):
                continue

            if spec.kind == "json_key":
                if not spec.key:
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        continue
                    value = data.get(spec.key)
                    if spec.subkey and isinstance(value, dict):
                        value = value.get(spec.subkey)
                    if isinstance(value, (int, float)):
                        metrics[spec.name] = float(value)
                except Exception:
                    continue

            elif spec.kind == "jsonl_last":
                if not spec.key:
                    continue
                try:
                    where = getattr(spec, "where", None) or {}
                    if where is None:
                        where = {}
                    if not isinstance(where, dict):
                        where = {}

                    last_row = None
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = (line or "").strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            if not isinstance(row, dict):
                                continue
                            ok = True
                            for k, v in where.items():
                                if row.get(k) != v:
                                    ok = False
                                    break
                            if ok:
                                last_row = row

                    if not isinstance(last_row, dict):
                        continue
                    value = last_row.get(spec.key)
                    if spec.subkey and isinstance(value, dict):
                        value = value.get(spec.subkey)
                    if isinstance(value, (int, float)):
                        metrics[spec.name] = float(value)
                except Exception:
                    continue

            elif spec.kind == "regex":
                if not spec.pattern:
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    matches = re.findall(spec.pattern, content)
                    if not matches:
                        continue
                    last = matches[-1]
                    if isinstance(last, tuple):
                        last = last[0] if last else None
                    if last is None:
                        continue
                    metrics[spec.name] = float(last)
                except Exception:
                    continue

        return metrics

    def _build_system_prompt(self, **kwargs) -> str:
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "exp_worker", "system.txt"
        )
        return load_and_render_prompt(
            prompt_path=prompt_path,
            variables={
                "project_root": str(kwargs.get("project_root", "") or ""),
            },
        )

    def _build_user_prompt(
        self,
        task: ExperimentTask,
        project_root: str,
        feedback: str = "",
        **kwargs,
    ) -> str:
        _ = kwargs
        builder = PromptBuilder()

        abs_result_dir = self._get_abs_result_dir(task, project_root) or ""
        safe_id = self._safe_task_id(task.task_id)

        builder.add_header("Experiment Execution Task")
        builder.add_key_value("Task ID", str(task.task_id))
        builder.add_key_value("Project Root", str(project_root))
        builder.add_key_value(
            "Result Dir (relative)", str(getattr(task, "result_dir", "") or "")
        )
        builder.add_key_value("Result Dir (absolute)", str(abs_result_dir))
        builder.add_text("")

        builder.add_header("Description", level=2)
        builder.add_text(str(getattr(task, "description", "") or "").strip())

        builder.add_header("Command", level=2)
        builder.add_code(str(getattr(task, "command", "") or "").strip(), language="")

        builder.add_header("Expected Outputs (relative to project root)", level=2)
        builder.add_list(
            [str(x) for x in (getattr(task, "expected_output_files", []) or [])]
        )

        builder.add_header("Metric Specs (extract after run; no guessing)", level=2)
        try:
            ms = [
                (m.model_dump() if hasattr(m, "model_dump") else dict(m))
                for m in (getattr(task, "metric_specs", None) or [])
            ]
        except Exception:
            ms = []
        builder.add_code(json.dumps(ms, ensure_ascii=False, indent=2), language="json")

        if feedback:
            builder.add_header("Feedback From Manager (previous attempt)", level=2)
            builder.add_text(str(feedback))

        builder.add_separator()
        builder.add_header("Hard Requirements", level=2)
        builder.add_list(
            [
                f"Create the result directory if it does not exist: `{abs_result_dir}`",
                "Run from project root using bash tool (set working_dir to project root).",
                f"Before running, export env var `SCIENCE_RESULT_DIR` to the absolute result dir (`{abs_result_dir}`) for the command.",
                "If the command fails, inspect stdout/stderr, create/fix configs or other inputs under result_dir as needed, and re-run until success or you can provide a clear failure reason.",
                "Do NOT modify project code outside result_dir. Only write under result_dir.",
                "Persist orchestration artifacts under result_dir (per-task filenames):",
                f"- `stdout_{safe_id}.txt`, `stderr_{safe_id}.txt`, `meta_{safe_id}.json`",
            ],
            ordered=False,
        )
        builder.add_text("")
        builder.add_header("Meta file schema", level=3)
        builder.add_text(
            "Write `meta_{task_id}.json` with keys: task_id, success, returncode, error, notes, setup_summary, command_summary."
        )
        builder.add_text("")
        builder.add_text(
            'Output the final line exactly: "EXPERIMENT COMPLETE" when done.'
        )

        return builder.build()

    async def run_task(
        self,
        task: ExperimentTask,
        project_root: str,
        feedback: str = "",
    ) -> ExperimentResult:
        if self.verbose:
            print(f"    ExpWorker: Executing {task.task_id} (agent-driven)...")
            if getattr(task, "result_dir", ""):
                print(f"      Result Dir: {task.result_dir}")

        result_obj = await self._run_agent(
            user_prompt=self._build_user_prompt(
                task=task, project_root=project_root, feedback=feedback
            ),
            system_prompt=self._build_system_prompt(project_root=project_root),
            tools=self._get_tools(),
        )
        _ = result_obj  # output text not relied upon; artifacts are on disk

        # Verify outputs + collect metrics (authoritative)
        missing_files: List[str] = []
        artifacts: List[str] = []
        for expected_file in getattr(task, "expected_output_files", []) or []:
            p = os.path.join(project_root, expected_file)
            if os.path.exists(p):
                artifacts.append(expected_file)
            else:
                missing_files.append(expected_file)

        metrics = self._collect_metrics(task, project_root)

        safe_id = self._safe_task_id(task.task_id)
        abs_dir = self._get_abs_result_dir(task, project_root) or ""
        meta_path = os.path.join(abs_dir, f"meta_{safe_id}.json") if abs_dir else ""
        stdout_path = os.path.join(abs_dir, f"stdout_{safe_id}.txt") if abs_dir else ""
        stderr_path = os.path.join(abs_dir, f"stderr_{safe_id}.txt") if abs_dir else ""

        meta = (
            self._read_json(meta_path)
            if meta_path and os.path.exists(meta_path)
            else None
        )
        stdout = (
            self._read_text(stdout_path)
            if stdout_path and os.path.exists(stdout_path)
            else ""
        )
        stderr = (
            self._read_text(stderr_path)
            if stderr_path and os.path.exists(stderr_path)
            else ""
        )

        meta_success = None
        meta_returncode = None
        meta_error = None
        if isinstance(meta, dict):
            meta_success = meta.get("success")
            meta_returncode = meta.get("returncode")
            meta_error = meta.get("error")

        success = (
            bool(meta_success)
            if meta_success is not None
            else (len(missing_files) == 0)
        )
        error = None
        if not success:
            reason_parts = []
            if missing_files:
                reason_parts.append(
                    "Missing expected output files: "
                    + ", ".join([str(x) for x in missing_files])
                )
            if meta_error:
                reason_parts.append(str(meta_error))
            if stderr:
                reason_parts.append((stderr or "")[:2000])
            error = (
                "\n".join([x for x in reason_parts if x]).strip() or "Experiment failed"
            )

        # Ensure downstream has persisted artifacts, even if the agent forgot some.
        try:
            self._persist_orchestration_outputs(
                task=task,
                project_root=project_root,
                stdout=stdout,
                stderr=stderr,
                metrics=metrics,
                success=success,
                error=error if not success else None,
                returncode=(
                    meta_returncode if isinstance(meta_returncode, int) else None
                ),
            )
        except Exception:
            pass

        return ExperimentResult(
            task_id=str(task.task_id),
            success=bool(success),
            metrics=metrics,
            artifacts=artifacts,
            result_dir=getattr(task, "result_dir", None) or None,
            error=error if not success else None,
            stdout=stdout,
            stderr=stderr,
        )
