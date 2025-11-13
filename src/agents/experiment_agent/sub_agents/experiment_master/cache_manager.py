"""
Cache Manager for Experiment Workflow.

This module provides caching functionality based on experiment IDs.
Each experiment has its own isolated cache directory with a manifest
tracking the complete workflow progress.
"""

import json
import os
from datetime import datetime
from typing import Any, Optional, Dict, List
from pathlib import Path


class CacheManager:
    """Manages experiment caching with experiment ID isolation."""

    def __init__(self, cache_dir: str = "./cached"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory to store all experiment caches
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Current experiment context
        self.current_experiment_id: Optional[str] = None
        self.current_experiment_dir: Optional[Path] = None

    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start or resume an experiment with given ID.

        Args:
            experiment_id: Unique identifier for the experiment

        Returns:
            True if this is a new experiment, False if resuming existing
        """
        self.current_experiment_id = experiment_id
        self.current_experiment_dir = self.cache_dir / experiment_id

        is_new = not self.current_experiment_dir.exists()
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True)

        # Check if manifest exists (to handle cases where directory exists but manifest doesn't)
        manifest_path = self._get_manifest_path()
        manifest_exists = manifest_path.exists()

        if is_new or not manifest_exists:
            # Initialize manifest for new experiment or repair incomplete experiment
            if not is_new and not manifest_exists:
                print(
                    f"[CACHE] Warning: Experiment directory exists but manifest missing. Reinitializing..."
                )
            manifest = {
                "experiment_id": experiment_id,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "status": "initialized",
                "workflow_state": {
                    "current_state": "initial",
                    "current_checklist_step": 0,
                    "completed_checklist_steps": [],
                    "checklist_step_retry_count": 0,
                    "execution_step_counter": 0,
                    "iteration_count": 0,
                },
                "cached_outputs": {},
                "workflow_progress": [],
                "current_stage": None,
            }
            self._save_manifest(manifest)
            print(f"[CACHE] Created new experiment: {experiment_id}")
            return True
        else:
            manifest = self._load_manifest()
            print(f"[CACHE] Resuming experiment: {experiment_id}")
            print(f"[CACHE] Last updated: {manifest.get('last_updated')}")
            print(f"[CACHE] Current stage: {manifest.get('current_stage')}")
            return False

    def _get_manifest_path(self) -> Path:
        """Get path to experiment manifest file."""
        if not self.current_experiment_dir:
            raise ValueError("No experiment started. Call start_experiment() first.")
        return self.current_experiment_dir / "manifest.json"

    def _save_manifest(self, manifest: Dict) -> None:
        """Save experiment manifest."""
        manifest_path = self._get_manifest_path()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _load_manifest(self) -> Dict:
        """Load experiment manifest."""
        manifest_path = self._get_manifest_path()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_agent_cache_path(self, agent_name: str, step_id: int) -> Path:
        """
        Get cache file path for specific agent.

        All agents use the format: step{i}_agent_name.json
        This helps track the complete execution path clearly.

        Args:
            agent_name: Name of the agent
            step_id: Step ID in the execution sequence (required for all agents)

        Returns:
            Path to the cache file
        """
        if not self.current_experiment_dir:
            raise ValueError("No experiment started. Call start_experiment() first.")

        return self.current_experiment_dir / f"step{step_id}_{agent_name}.json"

    def _serialize_output(self, output: Any) -> Any:
        """Serialize agent output to JSON-compatible format."""
        if output is None:
            return None

        # Handle Pydantic models
        if hasattr(output, "model_dump"):
            return output.model_dump()
        elif hasattr(output, "dict"):
            return output.dict()
        # Handle dataclasses
        elif hasattr(output, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(output)
        # Handle dict
        elif isinstance(output, dict):
            return output
        # Handle list
        elif isinstance(output, list):
            return [self._serialize_output(item) for item in output]
        # Handle primitive types
        elif isinstance(output, (str, int, float, bool)):
            return output
        # Fallback: convert to string
        else:
            return str(output)

    def save_cache(
        self,
        agent_name: str,
        input_data: str,
        output: Any,
        step_id: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save agent output to experiment cache.

        Args:
            agent_name: Name of the agent
            input_data: Input string to the agent
            output: Agent output to cache
            step_id: Step ID in the execution sequence (required for all agents)
            metadata: Optional metadata to store
        """
        if not self.current_experiment_id:
            raise ValueError("No experiment started. Call start_experiment() first.")

        cache_path = self._get_agent_cache_path(agent_name, step_id)

        # Serialize output
        serialized_output = self._serialize_output(output)

        # Prepare cache entry
        cache_entry = {
            "experiment_id": self.current_experiment_id,
            "agent_name": agent_name,
            "step_id": step_id,
            "timestamp": datetime.now().isoformat(),
            "input": input_data,  # Store complete input for debugging
            "output": serialized_output,
            "metadata": metadata or {},
        }

        # Save to file
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)

        # Update manifest
        manifest = self._load_manifest()
        manifest["last_updated"] = datetime.now().isoformat()

        # Create a unique key for this cache entry
        cache_key = f"step{step_id}_{agent_name}"
        manifest["current_stage"] = cache_key

        # Update workflow progress
        existing_entry = None
        for entry in manifest["workflow_progress"]:
            if entry.get("cache_key") == cache_key:
                existing_entry = entry
                break

        if existing_entry:
            existing_entry["timestamp"] = datetime.now().isoformat()
            existing_entry["status"] = "updated"
        else:
            manifest["workflow_progress"].append(
                {
                    "cache_key": cache_key,
                    "agent_name": agent_name,
                    "step_id": step_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                }
            )

        self._save_manifest(manifest)

        # Print cache save message
        print(
            f"[CACHE] Saved step{step_id}_{agent_name} output for experiment {self.current_experiment_id}"
        )

    def load_cache(self, agent_name: str, step_id: int) -> Optional[Any]:
        """
        Load agent output from experiment cache if available.

        Args:
            agent_name: Name of the agent
            step_id: Step ID in the execution sequence (required for all agents)

        Returns:
            Agent output or None if not cached
        """
        if not self.current_experiment_id:
            raise ValueError("No experiment started. Call start_experiment() first.")

        cache_path = self._get_agent_cache_path(agent_name, step_id)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_entry = json.load(f)

            output = cache_entry.get("output")
            timestamp = cache_entry.get("timestamp")

            cache_key = f"step{step_id}_{agent_name}"
            print(f"[CACHE] Loaded {cache_key} output (saved: {timestamp})")
            return output

        except Exception as e:
            cache_key = f"step{step_id}_{agent_name}"
            print(f"[CACHE] Error loading cache for {cache_key}: {str(e)}")
            return None

    def has_cache(self, agent_name: str, step_id: int) -> bool:
        """
        Check if cache exists for given agent in current experiment.

        Args:
            agent_name: Name of the agent
            step_id: Step ID in the execution sequence (required for all agents)

        Returns:
            True if cache exists, False otherwise
        """
        if not self.current_experiment_id:
            return False

        cache_path = self._get_agent_cache_path(agent_name, step_id)
        return cache_path.exists()

    def get_experiment_progress(self) -> Dict:
        """
        Get current experiment progress.

        Returns:
            Dictionary with experiment status and progress information
        """
        if not self.current_experiment_id:
            raise ValueError("No experiment started. Call start_experiment() first.")

        manifest = self._load_manifest()

        # Build completed stages list - use cache_key if available, otherwise agent_name
        completed_stages = []
        for entry in manifest["workflow_progress"]:
            if "cache_key" in entry:
                completed_stages.append(entry["cache_key"])
            else:
                # Backward compatibility with old format
                completed_stages.append(entry["agent_name"])

        return {
            "experiment_id": manifest["experiment_id"],
            "created_at": manifest["created_at"],
            "last_updated": manifest["last_updated"],
            "current_stage": manifest["current_stage"],
            "completed_stages": completed_stages,
            "workflow_progress": manifest["workflow_progress"],
        }

    def list_experiments(self) -> List[Dict]:
        """
        List all experiments in cache directory.

        Returns:
            List of experiment information dictionaries
        """
        experiments = []

        for exp_dir in self.cache_dir.iterdir():
            if exp_dir.is_dir():
                manifest_path = exp_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        experiments.append(
                            {
                                "experiment_id": manifest["experiment_id"],
                                "created_at": manifest["created_at"],
                                "last_updated": manifest["last_updated"],
                                "current_stage": manifest.get("current_stage"),
                                "num_stages": len(
                                    manifest.get("workflow_progress", [])
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"[CACHE] Error reading experiment {exp_dir.name}: {e}")

        return sorted(experiments, key=lambda x: x["last_updated"], reverse=True)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cached entries."""
        if not self.current_experiment_id:
            return {"total_experiments": len(list(self.cache_dir.iterdir()))}

        # Count actual cache files in the experiment directory
        cache_files = list(self.current_experiment_dir.glob("*.json"))
        # Exclude manifest.json from count
        cache_files = [f for f in cache_files if f.name != "manifest.json"]

        stats = {
            "experiment_id": self.current_experiment_id,
            "cached_entries": len(cache_files),
            "cache_files": [f.name for f in cache_files],
        }

        return stats

    def replay_experiment(self) -> Dict[str, Any]:
        """
        Replay/load the complete experiment workflow from cache.

        Returns:
            Dictionary mapping cache keys to their cached outputs
        """
        if not self.current_experiment_id:
            raise ValueError("No experiment started. Call start_experiment() first.")

        manifest = self._load_manifest()
        replay_data = {}

        print(f"\n[CACHE REPLAY] Replaying experiment: {self.current_experiment_id}")
        print(f"[CACHE REPLAY] Created: {manifest['created_at']}")
        print(f"[CACHE REPLAY] Last updated: {manifest['last_updated']}\n")

        for entry in manifest["workflow_progress"]:
            agent_name = entry["agent_name"]
            step_id = entry.get("step_id")
            cache_key = entry.get("cache_key", f"step{step_id}_{agent_name}")

            if step_id is None:
                print(
                    f"[CACHE REPLAY] ⚠ {cache_key}: missing step_id (old format, skipping)"
                )
                continue

            output = self.load_cache(agent_name, step_id)
            if output:
                replay_data[cache_key] = output
                print(f"[CACHE REPLAY] ✓ {cache_key}: {entry['timestamp']}")
            else:
                print(f"[CACHE REPLAY] ✗ {cache_key}: cache file missing")

        return replay_data

    def clear_experiment(self, experiment_id: Optional[str] = None) -> None:
        """
        Clear cache for a specific experiment.

        Args:
            experiment_id: ID of experiment to clear (default: current)
        """
        if experiment_id is None:
            experiment_id = self.current_experiment_id

        if not experiment_id:
            raise ValueError("No experiment ID specified")

        exp_dir = self.cache_dir / experiment_id
        if exp_dir.exists():
            import shutil

            shutil.rmtree(exp_dir)
            print(f"[CACHE] Cleared experiment: {experiment_id}")
        else:
            print(f"[CACHE] Experiment not found: {experiment_id}")

    def save_workflow_snapshot(
        self,
        context: "WorkflowContext",
        agent_name: Optional[str] = None,
        agent_output: Any = None,
    ) -> None:
        """
        Save complete workflow snapshot to manifest.

        This should be called after each agent execution to ensure
        the workflow state is always recoverable.

        Args:
            context: Current workflow context
            agent_name: Name of the agent that just executed (optional)
            agent_output: Output from the agent (optional)
        """
        if not self.current_experiment_id:
            raise ValueError("No experiment started. Call start_experiment() first.")

        # Load current manifest
        manifest = self._load_manifest()

        # Update workflow state from context
        manifest["workflow_state"] = {
            "current_state": context.current_state.value,
            "current_checklist_step": context.current_checklist_step,
            "completed_checklist_steps": context.completed_checklist_steps or [],
            "checklist_step_retry_count": context.checklist_step_retry_count,
            "execution_step_counter": context.execution_step_counter,
            "iteration_count": context.iteration_count,
        }

        # Update cached outputs references
        if "cached_outputs" not in manifest:
            manifest["cached_outputs"] = {}

        if agent_name and agent_output is not None:
            # Save reference to the latest output file for each agent type
            step_id = context.execution_step_counter
            cache_filename = f"step{step_id}_{agent_name}.json"
            manifest["cached_outputs"][agent_name] = cache_filename

            # Keep special references for important outputs
            if agent_name == "pre_analysis":
                manifest["cached_outputs"]["last_pre_analysis"] = cache_filename
            elif agent_name == "code_plan":
                manifest["cached_outputs"]["last_code_plan"] = cache_filename
            elif agent_name == "code_implement":
                manifest["cached_outputs"]["last_code_implement"] = cache_filename
            elif agent_name == "code_judge":
                manifest["cached_outputs"]["last_code_judge"] = cache_filename

        # Update metadata
        manifest["last_updated"] = datetime.now().isoformat()
        manifest["status"] = "in_progress"

        # Save manifest
        self._save_manifest(manifest)

    def resume_workflow_context(self, context: "WorkflowContext") -> bool:
        """
        Resume workflow context from cache.

        This restores the complete workflow state including:
        - Current workflow state
        - Checklist progress
        - Execution counter
        - Previously generated outputs (plan, analysis)

        Args:
            context: WorkflowContext object to restore (will be modified in-place)

        Returns:
            True if successfully restored, False if manifest is invalid
        """
        if not self.current_experiment_id:
            raise ValueError("No experiment started. Call start_experiment() first.")

        try:
            manifest = self._load_manifest()

            # Restore workflow state (required)
            from .workflow_state_machine import WorkflowState

            workflow_state = manifest["workflow_state"]
            context.current_state = WorkflowState(workflow_state["current_state"])
            context.current_checklist_step = workflow_state["current_checklist_step"]
            context.completed_checklist_steps = workflow_state[
                "completed_checklist_steps"
            ]
            context.checklist_step_retry_count = workflow_state[
                "checklist_step_retry_count"
            ]
            context.execution_step_counter = workflow_state["execution_step_counter"]
            context.iteration_count = workflow_state["iteration_count"]

            # Restore cached outputs
            import re

            cached_outputs = manifest.get("cached_outputs", {})

            # Load pre_analysis output if exists
            if "last_pre_analysis" in cached_outputs:
                analysis_file = cached_outputs["last_pre_analysis"]
                match = re.search(r"step(\d+)_", analysis_file)
                if match:
                    step_id = int(match.group(1))
                    context.pre_analysis_output = self.load_cache(
                        "pre_analysis", step_id
                    )

            # Load code_plan output if exists
            if "last_code_plan" in cached_outputs:
                plan_file = cached_outputs["last_code_plan"]
                match = re.search(r"step(\d+)_", plan_file)
                if match:
                    step_id = int(match.group(1))
                    context.code_plan_output = self.load_cache("code_plan", step_id)

            # Load code_implement output if exists
            if "last_code_implement" in cached_outputs:
                implement_file = cached_outputs["last_code_implement"]
                match = re.search(r"step(\d+)_", implement_file)
                if match:
                    step_id = int(match.group(1))
                    context.code_implement_output = self.load_cache(
                        "code_implement", step_id
                    )
                    if context.code_implement_output:
                        print(
                            f"[CACHE] Loaded step{step_id}_code_implement output (saved: {cached_outputs.get('last_code_implement_timestamp', 'unknown')})"
                        )

            # Load code_judge output if exists
            if "last_code_judge" in cached_outputs:
                judge_file = cached_outputs["last_code_judge"]
                match = re.search(r"step(\d+)_", judge_file)
                if match:
                    step_id = int(match.group(1))
                    context.code_judge_output = self.load_cache("code_judge", step_id)
                    if context.code_judge_output:
                        print(
                            f"[CACHE] Loaded step{step_id}_code_judge output (saved: {cached_outputs.get('last_code_judge_timestamp', 'unknown')})"
                        )

            print(f"[CACHE] Restored workflow state:")
            print(f"  State: {context.current_state.value}")
            print(f"  Checklist step: {context.current_checklist_step}")
            print(f"  Completed steps: {context.completed_checklist_steps}")
            print(f"  Retry count: {context.checklist_step_retry_count}")
            print(f"  Execution counter: {context.execution_step_counter}")
            print(f"  Has pre_analysis: {context.pre_analysis_output is not None}")
            print(f"  Has code_plan: {context.code_plan_output is not None}")
            print(f"  Has code_implement: {context.code_implement_output is not None}")
            print(f"  Has code_judge: {context.code_judge_output is not None}")

            return True

        except KeyError as e:
            print(f"[CACHE] Invalid manifest format: missing {e}")
            print(
                "[CACHE] This experiment was created with an old format and cannot be resumed"
            )
            print("[CACHE] Please clear the cache directory and start a new experiment")
            return False
        except Exception as e:
            print(f"[CACHE] Error restoring workflow state: {e}")
            import traceback

            traceback.print_exc()
            return False
