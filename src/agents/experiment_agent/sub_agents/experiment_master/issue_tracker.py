"""
Issue Tracker for Experiment Workflow.

This module provides issue tracking functionality to:
1. Track issues across judge iterations using fingerprint matching
2. Automatically detect resolved issues (no longer reported by judge)
3. Auto-escalate severity for recurring issues
4. Provide formatted issue history for implement/judge agents

Compatible with the cache system - issue history is stored in manifest.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class TrackedIssue:
    """Represents a tracked issue across iterations."""
    
    issue_id: str  # Unique ID like "ISS-001"
    fingerprint: str  # Hash for matching similar issues
    
    # Issue details (from first occurrence)
    file_path: str
    issue_type: str
    original_severity: str  # Severity when first seen
    current_severity: str  # May be escalated
    description: str
    suggestion: str
    
    # Tracking metadata
    first_seen_step: int  # execution_step_counter when first seen
    last_seen_step: int  # execution_step_counter when last seen
    occurrence_count: int = 1  # How many times this issue appeared
    
    # Status tracking
    status: str = "open"  # "open" or "resolved"
    resolution_step: Optional[int] = None  # Step when resolved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedIssue":
        """Create from dictionary (for deserialization)."""
        return cls(**data)


class IssueTracker:
    """
    Tracks issues across judge iterations with fingerprint matching.
    
    Key features:
    - Generates fingerprints for issue matching
    - Tracks occurrence count and auto-escalates severity
    - Automatically marks issues as resolved when not reported
    - Provides formatted output for agent inputs
    """
    
    # Severity escalation rules
    SEVERITY_ORDER = ["minor", "major", "critical"]
    ESCALATION_THRESHOLD = 2  # Escalate after N occurrences
    
    def __init__(self):
        """Initialize the issue tracker."""
        self.issues: Dict[str, TrackedIssue] = {}  # fingerprint -> TrackedIssue
        self.issue_counter = 0  # For generating issue IDs
        self._last_step_fingerprints: Set[str] = set()  # Track which issues were seen last step
    
    def _generate_fingerprint(self, issue: Dict[str, Any]) -> str:
        """
        Generate a fingerprint for matching similar issues.
        
        Fingerprint is based on:
        - file_path
        - issue_type
        - Key terms from description (normalized)
        """
        file_path = issue.get("file_path", "unknown")
        issue_type = issue.get("issue_type", "unknown")
        description = issue.get("description", "")
        
        # Extract key terms from description
        # Remove common words and normalize
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "this", "that", "these", "those", "it", "its", "and", "or",
            "but", "if", "when", "which", "what", "where", "who", "how",
            "not", "no", "as", "but", "so", "than", "too", "very",
        }
        
        # Tokenize and normalize
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', description.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Take top 5 most significant terms (by length as proxy for specificity)
        key_terms = sorted(set(key_terms), key=lambda x: (-len(x), x))[:5]
        
        # Create fingerprint
        fingerprint_str = f"{file_path}::{issue_type}::{'-'.join(sorted(key_terms))}"
        
        # Hash for consistent length
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
    
    def _escalate_severity(self, current: str, occurrence_count: int) -> str:
        """
        Escalate severity based on occurrence count.
        
        Rules:
        - occurrence_count >= 2: minor -> major
        - occurrence_count >= 3: major -> critical
        """
        if current not in self.SEVERITY_ORDER:
            return current
        
        current_idx = self.SEVERITY_ORDER.index(current)
        
        # Calculate escalation
        escalations = (occurrence_count - 1) // self.ESCALATION_THRESHOLD
        new_idx = min(current_idx + escalations, len(self.SEVERITY_ORDER) - 1)
        
        return self.SEVERITY_ORDER[new_idx]
    
    def process_judge_output(
        self, 
        judge_output: Dict[str, Any], 
        current_step: int
    ) -> Dict[str, Any]:
        """
        Process judge output to update issue tracking.
        
        This method:
        1. Extracts issues from judge output
        2. Matches against existing issues using fingerprints
        3. Updates occurrence counts and escalates severity
        4. Marks issues not present in this output as resolved
        
        Args:
            judge_output: Output from code_judge agent
            current_step: Current execution_step_counter
            
        Returns:
            Summary of changes (new, updated, resolved counts)
        """
        # Extract issues from judge output
        issues = judge_output.get("issues", [])
        
        # Track which fingerprints we see in this step
        current_step_fingerprints: Set[str] = set()
        
        new_count = 0
        updated_count = 0
        
        for issue in issues:
            fingerprint = self._generate_fingerprint(issue)
            current_step_fingerprints.add(fingerprint)
            
            if fingerprint in self.issues:
                # Existing issue - update
                tracked = self.issues[fingerprint]
                tracked.occurrence_count += 1
                tracked.last_seen_step = current_step
                tracked.status = "open"  # Re-open if was resolved
                tracked.resolution_step = None
                
                # Update description/suggestion if more detailed
                if len(issue.get("description", "")) > len(tracked.description):
                    tracked.description = issue.get("description", tracked.description)
                if len(issue.get("suggestion", "")) > len(tracked.suggestion):
                    tracked.suggestion = issue.get("suggestion", tracked.suggestion)
                
                # Escalate severity if needed
                new_severity = self._escalate_severity(
                    tracked.original_severity, 
                    tracked.occurrence_count
                )
                if new_severity != tracked.current_severity:
                    print(f"[ISSUE_TRACKER] Escalating {tracked.issue_id}: "
                          f"{tracked.current_severity} -> {new_severity} "
                          f"(occurred {tracked.occurrence_count} times)")
                    tracked.current_severity = new_severity
                
                updated_count += 1
            else:
                # New issue
                self.issue_counter += 1
                issue_id = f"ISS-{self.issue_counter:03d}"
                
                tracked = TrackedIssue(
                    issue_id=issue_id,
                    fingerprint=fingerprint,
                    file_path=issue.get("file_path", "unknown"),
                    issue_type=issue.get("issue_type", "unknown"),
                    original_severity=issue.get("severity", "minor"),
                    current_severity=issue.get("severity", "minor"),
                    description=issue.get("description", ""),
                    suggestion=issue.get("suggestion", ""),
                    first_seen_step=current_step,
                    last_seen_step=current_step,
                    occurrence_count=1,
                    status="open",
                )
                
                self.issues[fingerprint] = tracked
                new_count += 1
                print(f"[ISSUE_TRACKER] New issue {issue_id}: {tracked.file_path} - {tracked.issue_type}")
        
        # Mark issues not seen in this step as resolved
        resolved_count = 0
        for fingerprint, tracked in self.issues.items():
            if (fingerprint not in current_step_fingerprints 
                and tracked.status == "open"
                and fingerprint in self._last_step_fingerprints):
                # Issue was present last step but not now -> resolved
                tracked.status = "resolved"
                tracked.resolution_step = current_step
                resolved_count += 1
                print(f"[ISSUE_TRACKER] Resolved {tracked.issue_id}: {tracked.file_path}")
        
        # Update last step fingerprints
        self._last_step_fingerprints = current_step_fingerprints
        
        return {
            "new_issues": new_count,
            "updated_issues": updated_count,
            "resolved_issues": resolved_count,
            "total_open": sum(1 for t in self.issues.values() if t.status == "open"),
            "total_resolved": sum(1 for t in self.issues.values() if t.status == "resolved"),
        }
    
    def get_open_issues(self) -> List[TrackedIssue]:
        """Get all open issues, sorted by severity and occurrence count."""
        open_issues = [t for t in self.issues.values() if t.status == "open"]
        
        # Sort by: severity (critical > major > minor), then occurrence count
        severity_weight = {"critical": 3, "major": 2, "minor": 1}
        
        return sorted(
            open_issues,
            key=lambda x: (
                -severity_weight.get(x.current_severity, 0),
                -x.occurrence_count,
            )
        )
    
    def get_resolved_issues(self) -> List[TrackedIssue]:
        """Get all resolved issues."""
        return [t for t in self.issues.values() if t.status == "resolved"]
    
    def get_recurring_issues(self, min_occurrences: int = 2) -> List[TrackedIssue]:
        """Get issues that have occurred multiple times."""
        return [
            t for t in self.issues.values() 
            if t.occurrence_count >= min_occurrences
        ]
    
    def format_for_implement_agent(self) -> str:
        """
        Format issue history for code_implement agent input.
        
        Returns a formatted string that includes:
        - Open issues (prioritized)
        - Recently resolved issues (for context)
        """
        lines = []
        
        open_issues = self.get_open_issues()
        resolved_issues = self.get_resolved_issues()
        
        if not open_issues and not resolved_issues:
            return ""
        
        lines.append("=== ISSUE HISTORY (from previous judge reviews) ===\n")
        
        # Open issues
        if open_issues:
            lines.append("### OPEN ISSUES (must be addressed)")
            lines.append("")
            
            for issue in open_issues:
                warning = ""
                if issue.occurrence_count >= 3:
                    warning = " ⚠️ RECURRING (appeared 3+ times)"
                elif issue.occurrence_count >= 2:
                    warning = " ⚠️ (appeared twice)"
                
                escalation_note = ""
                if issue.current_severity != issue.original_severity:
                    escalation_note = f" [ESCALATED from {issue.original_severity}]"
                
                lines.append(f"[{issue.issue_id}] [{issue.current_severity.upper()}]{escalation_note}{warning}")
                lines.append(f"  File: {issue.file_path}")
                lines.append(f"  Type: {issue.issue_type}")
                lines.append(f"  Problem: {issue.description}")
                lines.append(f"  Fix: {issue.suggestion}")
                lines.append(f"  First seen: step {issue.first_seen_step}, Occurrences: {issue.occurrence_count}")
                lines.append("")
        
        # Resolved issues (brief, for context)
        if resolved_issues:
            lines.append("### RECENTLY RESOLVED ISSUES (for reference)")
            lines.append("")
            
            # Only show recently resolved (last 3)
            recent_resolved = sorted(
                resolved_issues, 
                key=lambda x: x.resolution_step or 0, 
                reverse=True
            )[:3]
            
            for issue in recent_resolved:
                lines.append(f"[{issue.issue_id}] ✓ RESOLVED at step {issue.resolution_step}")
                lines.append(f"  File: {issue.file_path} - {issue.issue_type}")
                lines.append("")
        
        lines.append("=" * 60)
        lines.append("")
        
        return "\n".join(lines)
    
    def format_for_judge_agent(self) -> str:
        """
        Format issue history for code_judge agent input.
        
        Helps judge identify:
        - Which issues were previously identified
        - Which issues are recurring
        """
        lines = []
        
        open_issues = self.get_open_issues()
        
        if not open_issues:
            return ""
        
        lines.append("=== PREVIOUSLY IDENTIFIED ISSUES ===")
        lines.append("(Check if these issues have been resolved in the current implementation)")
        lines.append("")
        
        for issue in open_issues:
            status_note = ""
            if issue.occurrence_count >= 3:
                status_note = " [RECURRING - 3+ occurrences]"
            elif issue.occurrence_count >= 2:
                status_note = " [APPEARED TWICE]"
            
            lines.append(f"[{issue.issue_id}] {issue.file_path}::{issue.issue_type}{status_note}")
            lines.append(f"  Description: {issue.description[:200]}...")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for cache storage."""
        return {
            "issue_counter": self.issue_counter,
            "issues": {fp: issue.to_dict() for fp, issue in self.issues.items()},
            "last_step_fingerprints": list(self._last_step_fingerprints),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IssueTracker":
        """Deserialize tracker state from cache."""
        tracker = cls()
        tracker.issue_counter = data.get("issue_counter", 0)
        tracker.issues = {
            fp: TrackedIssue.from_dict(issue_data)
            for fp, issue_data in data.get("issues", {}).items()
        }
        tracker._last_step_fingerprints = set(data.get("last_step_fingerprints", []))
        return tracker
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        open_issues = self.get_open_issues()
        resolved_issues = self.get_resolved_issues()
        recurring = self.get_recurring_issues()
        
        return {
            "total_tracked": len(self.issues),
            "open_count": len(open_issues),
            "resolved_count": len(resolved_issues),
            "recurring_count": len(recurring),
            "severity_breakdown": {
                "critical": sum(1 for i in open_issues if i.current_severity == "critical"),
                "major": sum(1 for i in open_issues if i.current_severity == "major"),
                "minor": sum(1 for i in open_issues if i.current_severity == "minor"),
            },
        }

