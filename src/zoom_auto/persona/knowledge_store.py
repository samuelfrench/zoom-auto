"""Persistent storage for indexed project knowledge.

Stores ProjectIndex data as JSON files so the bot can reference
project context in meetings without re-indexing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from zoom_auto.persona.sources.project import ProjectIndex

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeStore:
    """Manages persistent project knowledge.

    Stores indexed project data in data/knowledge/ as JSON files.
    Provides a combined context string for the response generator.

    Args:
        data_dir: Directory to store knowledge JSON files.
            Defaults to data/knowledge/.
    """

    data_dir: Path

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path("data/knowledge")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self, project_name: str) -> Path:
        """Get the JSON file path for a project index."""
        # Sanitize name for filesystem safety
        safe_name = project_name.replace("/", "_").replace("\\", "_")
        return self.data_dir / f"{safe_name}.json"

    def save_index(self, index: ProjectIndex) -> Path:
        """Save a project index to disk.

        Args:
            index: The project index to persist.

        Returns:
            Path to the saved JSON file.
        """
        path = self._index_path(index.name)
        data = index.to_dict()
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved project index '%s' to %s", index.name, path)
        return path

    def load_index(self, project_name: str) -> ProjectIndex | None:
        """Load a project index from disk.

        Args:
            project_name: Name of the project to load.

        Returns:
            ProjectIndex if found, None otherwise.
        """
        path = self._index_path(project_name)
        if not path.is_file():
            logger.debug("No index found for project '%s'", project_name)
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ProjectIndex.from_dict(data)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "Failed to load index for '%s': %s", project_name, exc
            )
            return None

    def list_projects(self) -> list[str]:
        """List all indexed projects.

        Returns:
            List of project names that have been indexed.
        """
        if not self.data_dir.is_dir():
            return []

        projects: list[str] = []
        for path in sorted(self.data_dir.iterdir()):
            if path.is_file() and path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    name = data.get("name", path.stem)
                    projects.append(name)
                except (json.JSONDecodeError, OSError):
                    # Fall back to filename
                    projects.append(path.stem)

        return projects

    def delete_index(self, project_name: str) -> bool:
        """Delete a project index.

        Args:
            project_name: Name of the project to delete.

        Returns:
            True if the index was deleted, False if not found.
        """
        path = self._index_path(project_name)
        if path.is_file():
            path.unlink()
            logger.info("Deleted project index '%s'", project_name)
            return True

        logger.debug("No index to delete for project '%s'", project_name)
        return False

    def get_context_string(self, max_tokens: int = 1000) -> str:
        """Build a context string from all indexed projects.

        Returns a formatted string summarizing all indexed projects
        for inclusion in the LLM system prompt.

        Args:
            max_tokens: Approximate maximum token count for the
                context string. Uses a rough 4-chars-per-token estimate.

        Returns:
            Formatted context string, or empty string if no projects.
        """
        projects = self.list_projects()
        if not projects:
            return ""

        max_chars = max_tokens * 4  # Rough token estimate
        sections: list[str] = ["## Project Knowledge\n"]
        current_chars = len(sections[0])

        for project_name in projects:
            index = self.load_index(project_name)
            if index is None:
                continue

            section = self._format_project_section(index)
            section_len = len(section)

            if current_chars + section_len > max_chars:
                # Truncate this section to fit
                remaining = max_chars - current_chars
                if remaining > 100:
                    section = section[:remaining] + "\n... (truncated)"
                    sections.append(section)
                break

            sections.append(section)
            current_chars += section_len

        if len(sections) <= 1:
            return ""

        return "\n".join(sections)

    def _format_project_section(self, index: ProjectIndex) -> str:
        """Format a single project index as a context section."""
        lines: list[str] = [f"### {index.name}"]

        if index.tech_stack:
            lines.append(f"- Tech stack: {', '.join(index.tech_stack)}")

        if index.patterns:
            lines.append(f"- Key patterns: {', '.join(index.patterns)}")

        if index.dependencies:
            # Show top 15 dependencies to avoid bloat
            deps = index.dependencies[:15]
            deps_str = ", ".join(deps)
            if len(index.dependencies) > 15:
                deps_str += f" (+{len(index.dependencies) - 15} more)"
            lines.append(f"- Dependencies: {deps_str}")

        if index.readme_content:
            # Extract first paragraph as summary
            summary = self._extract_summary(index.readme_content)
            if summary:
                lines.append(f"- Summary: {summary}")

        lines.append("")  # Blank line between projects
        return "\n".join(lines)

    def _extract_summary(self, readme: str) -> str:
        """Extract a short summary from README content.

        Takes the first non-heading, non-empty paragraph.
        """
        lines = readme.split("\n")
        paragraph: list[str] = []

        for line in lines:
            stripped = line.strip()
            # Skip headings and badges
            if stripped.startswith("#"):
                if paragraph:
                    break
                continue
            if stripped.startswith("![") or stripped.startswith("[!["):
                continue
            if not stripped:
                if paragraph:
                    break
                continue
            paragraph.append(stripped)

        if not paragraph:
            return ""

        summary = " ".join(paragraph)
        # Truncate to ~200 chars
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary
