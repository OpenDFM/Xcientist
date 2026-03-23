"""
Checkpoint Manager - Handles pause/resume functionality for agents.

This module provides checkpoint saving and loading for OpenHands-based agents,
enabling interruption and resume functionality.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for agent execution state."""

    def __init__(self, workspace_root: str):
        """
        Initialize the checkpoint manager.

        Args:
            workspace_root: Root directory for the workspace
        """
        self.workspace_root = workspace_root
        # Only create .checkpoints in workspace subdirectories, not in root
        if "/workspace/" in workspace_root or workspace_root.endswith("/workspace"):
            self.checkpoint_dir = os.path.join(workspace_root, ".checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def _get_checkpoint_path(self, agent_type: str) -> str:
        """Get the checkpoint file path for an agent type."""
        return os.path.join(self.checkpoint_dir, f"{agent_type}_checkpoint.json")

    def save_checkpoint(
        self,
        agent_type: str,
        iteration: int,
        phase: str,
        conversation_persistence_dir: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save a checkpoint for the agent.

        Args:
            agent_type: Type of agent (e.g., "code", "science", "master")
            iteration: Current iteration number
            phase: Current phase (e.g., "planning", "execution")
            conversation_persistence_dir: Path to conversation persistence directory
            additional_data: Optional additional data to save

        Returns:
            True if checkpoint was saved successfully
        """
        checkpoint_data = {
            "agent_type": agent_type,
            "iteration": iteration,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "conversation_persistence_dir": conversation_persistence_dir,
            "additional_data": additional_data or {},
        }

        # Skip checkpoint if checkpoint_dir is None (not in workspace subdirectory)
        if self.checkpoint_dir is None:
            logger.info(f"Checkpoint skipped: not in workspace subdirectory")
            return True
        
        checkpoint_path = self._get_checkpoint_path(agent_type)
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for the agent.

        Args:
            agent_type: Type of agent

        Returns:
            Checkpoint data dict, or None if no checkpoint exists
        """
        # Skip checkpoint loading if checkpoint_dir is None (not in workspace subdirectory)
        if self.checkpoint_dir is None:
            return None
        
        checkpoint_path = self._get_checkpoint_path(agent_type)
        if not os.path.exists(checkpoint_path):
            logger.debug(f"No checkpoint found for {agent_type}")
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def has_checkpoint(self, agent_type: str) -> bool:
        """Check if a checkpoint exists for the agent."""
        checkpoint_path = self._get_checkpoint_path(agent_type)
        return os.path.exists(checkpoint_path)

    def clear_checkpoint(self, agent_type: str) -> bool:
        """
        Clear the checkpoint for the agent.

        Args:
            agent_type: Type of agent

        Returns:
            True if checkpoint was cleared successfully
        """
        checkpoint_path = self._get_checkpoint_path(agent_type)
        if not os.path.exists(checkpoint_path):
            return True

        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint cleared: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}")
            return False

    def get_latest_conversation_dir(self, agent_type: str) -> Optional[str]:
        """
        Get the latest conversation persistence directory for an agent type.

        This is useful when resuming - we need to find the existing conversation
        state to continue from.

        Args:
            agent_type: Type of agent

        Returns:
            Path to conversation persistence directory, or None
        """
        checkpoint = self.load_checkpoint(agent_type)
        if checkpoint:
            return checkpoint.get("conversation_persistence_dir")

        # Check if there's a conversation directory directly
        conversation_dir = os.path.join(self.workspace_root, ".conversations")
        if os.path.exists(conversation_dir):
            # Look for agent-specific subdirectory
            agent_conv_dir = os.path.join(conversation_dir, agent_type)
            if os.path.exists(agent_conv_dir):
                return agent_conv_dir

        return None


# Global checkpoint manager instance (will be initialized per workspace)
_checkpoint_managers: Dict[str, CheckpointManager] = {}


def get_checkpoint_manager(workspace_root: str) -> CheckpointManager:
    """Get or create a checkpoint manager for the workspace."""
    if workspace_root not in _checkpoint_managers:
        _checkpoint_managers[workspace_root] = CheckpointManager(workspace_root)
    return _checkpoint_managers[workspace_root]
