"""
Cache utilities for SuperAgent.

Provides:
- Blueprint caching
- Proposal hashing

Thread-safe implementation with file locking.
"""

import os
import json
import hashlib
import threading
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Cache:
    """
    Thread-safe file-based cache for blueprints.
    """

    _instance = None
    _cache_dir = ""
    _enabled = True
    _lock = threading.RLock()

    @classmethod
    def initialize(cls, cache_dir: str, enabled: bool = True):
        """
        Initialize the cache with a directory.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        with cls._lock:
            cls._cache_dir = cache_dir
            cls._enabled = enabled
            if enabled:
                os.makedirs(cache_dir, exist_ok=True)
                os.makedirs(os.path.join(cache_dir, "blueprints"), exist_ok=True)
                logger.debug(f"Cache initialized at {cache_dir}")

    @classmethod
    def _hash_key(cls, key: str) -> str:
        """Create a hash from the key."""
        return hashlib.md5(key.encode()).hexdigest()

    @classmethod
    def get_blueprint(cls, proposal_hash: str) -> Optional[dict]:
        """Get cached blueprint."""
        with cls._lock:
            if not cls._enabled or not cls._cache_dir:
                return None

            cache_file = os.path.join(
                cls._cache_dir, "blueprints", f"{proposal_hash}.json"
            )
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read blueprint cache: {e}")
            return None

    @classmethod
    def set_blueprint(cls, proposal_hash: str, blueprint: dict):
        """Cache blueprint."""
        with cls._lock:
            if not cls._enabled or not cls._cache_dir:
                return

            cache_file = os.path.join(
                cls._cache_dir, "blueprints", f"{proposal_hash}.json"
            )
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "blueprint": blueprint,
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                logger.warning(f"Failed to write blueprint cache: {e}")

    @classmethod
    def hash_proposal(cls, proposal) -> str:
        """Create hash from proposal for caching."""
        key = f"{proposal.idea.title}:{proposal.idea.description[:200]}"
        return cls._hash_key(key)

    @classmethod
    def clear(cls):
        """Clear all caches."""
        with cls._lock:
            if not cls._cache_dir:
                return

            import shutil

            for subdir in ["blueprints"]:
                path = os.path.join(cls._cache_dir, subdir)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path, exist_ok=True)
            logger.info("Cache cleared")

    @classmethod
    def get_run_state(cls) -> Dict[str, Any]:
        """
        Get global run state (single source of truth for orchestrator resume).

        Stored at: <cache_dir>/run_state.json
        """
        with cls._lock:
            if not cls._enabled or not cls._cache_dir:
                return {}

            path = os.path.join(cls._cache_dir, "run_state.json")
            if not os.path.exists(path):
                return {}

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.warning(f"Failed to read run_state: {e}")
                return {}

    @classmethod
    def set_run_state(cls, state: Dict[str, Any]) -> None:
        """Overwrite global run state."""
        with cls._lock:
            if not cls._enabled or not cls._cache_dir:
                return

            path = os.path.join(cls._cache_dir, "run_state.json")
            try:
                payload = dict(state or {})
                payload["timestamp"] = datetime.now().isoformat()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write run_state: {e}")

    @classmethod
    def update_run_state(cls, **kwargs) -> None:
        """Merge-update global run state."""
        with cls._lock:
            current = cls.get_run_state()
            current.update({k: v for k, v in kwargs.items() if v is not None})
            cls.set_run_state(current)

    @classmethod
    def update_layer_state(
        cls,
        namespace: str,
        active_layer: Optional[str] = None,
        active_stage: Optional[str] = None,
        step_index: Optional[int] = None,
        blueprint_id: Optional[str] = None,
        phase: Optional[str] = None,
        main_loop: Optional[int] = None,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Normalize layer-scoped run_state keys to reduce duplication at call sites.

        This preserves the existing run_state schema (e.g. code_step_index/science_step_index).
        """
        ns = (namespace or "").strip().lower()
        if not ns:
            cls.update_run_state(
                active_layer=active_layer,
                active_stage=active_stage,
                main_loop=main_loop,
                error=error,
                **extra,
            )
            return

        payload: Dict[str, Any] = {
            "namespace": ns,
            "active_layer": active_layer,
            "active_stage": active_stage,
            "main_loop": main_loop,
            "error": error,
        }
        if step_index is not None:
            payload[f"{ns}_step_index"] = int(step_index)
        if blueprint_id is not None:
            payload[f"{ns}_blueprint_id"] = blueprint_id
        if phase is not None:
            payload[f"{ns}_phase"] = phase
        payload.update({k: v for k, v in extra.items() if v is not None})
        cls.update_run_state(**payload)

    @classmethod
    def get_main_loop_num(cls) -> int:
        """
        Get the cached main loop number for resume.

        Returns:
            Loop number (>= 1). Defaults to 1 if not found or cache disabled.
        """
        with cls._lock:
            if not cls._enabled or not cls._cache_dir:
                return 1

            cache_file = os.path.join(cls._cache_dir, "main_loop.json")
            if not os.path.exists(cache_file):
                return 1

            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loop_num = int(data.get("loop_num", 1))
                return loop_num if loop_num >= 1 else 1
            except Exception as e:
                logger.warning(f"Failed to read main loop cache: {e}")
                return 1

    @classmethod
    def set_main_loop_num(cls, loop_num: int) -> None:
        """
        Cache the current main loop number for resume.

        Args:
            loop_num: Loop number (>= 1)
        """
        with cls._lock:
            if not cls._enabled or not cls._cache_dir:
                return

            try:
                loop_num_int = int(loop_num)
                if loop_num_int < 1:
                    loop_num_int = 1
            except Exception:
                loop_num_int = 1

            cache_file = os.path.join(cls._cache_dir, "main_loop.json")
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "loop_num": loop_num_int,
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                logger.warning(f"Failed to write main loop cache: {e}")
