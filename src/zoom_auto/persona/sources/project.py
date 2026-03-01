"""Project directory indexing for building technical context.

Scans codebases to extract tech stack, architecture patterns,
key files, and project knowledge so the bot can contribute
meaningfully in technical discussions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum characters to keep from README files
_MAX_README_CHARS = 5000

# Maximum lines to read from config files
_MAX_CONFIG_LINES = 200


@dataclass
class ProjectIndex:
    """Indexed knowledge from a project directory.

    Attributes:
        name: Project name (from directory name or package config).
        root_path: Absolute path to the project root.
        tech_stack: Detected technologies and frameworks.
        key_files: Important files with summaries.
        readme_content: Content of README if present.
        structure_summary: High-level directory structure.
        dependencies: List of project dependencies.
        patterns: Detected code patterns and conventions.
        total_files: Total files scanned.
    """

    name: str
    root_path: str
    tech_stack: list[str] = field(default_factory=list)
    key_files: dict[str, str] = field(default_factory=dict)  # path -> summary
    readme_content: str = ""
    structure_summary: str = ""
    dependencies: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    total_files: int = 0

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProjectIndex:
        """Create a ProjectIndex from a dictionary."""
        return cls(**data)


class ProjectIndexer:
    """Indexes project directories for technical context.

    Scans a project directory to extract:
    - Tech stack (from file extensions, config files, package managers)
    - Key files (README, config files, main entry points)
    - Dependencies (from package.json, pyproject.toml, requirements.txt, etc.)
    - Directory structure summary
    - Code patterns and conventions
    """

    # File patterns that indicate tech stack
    TECH_INDICATORS: dict[str, str] = {
        "pyproject.toml": "Python",
        "setup.py": "Python",
        "requirements.txt": "Python",
        "package.json": "Node.js/JavaScript",
        "tsconfig.json": "TypeScript",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java/Maven",
        "build.gradle": "Java/Gradle",
        "Gemfile": "Ruby",
        "composer.json": "PHP",
        "Dockerfile": "Docker",
        "docker-compose.yml": "Docker Compose",
        "docker-compose.yaml": "Docker Compose",
        "Makefile": "Make",
        "CMakeLists.txt": "C/C++ CMake",
    }

    # Directory patterns that indicate tech stack
    TECH_DIR_INDICATORS: dict[str, str] = {
        ".github/workflows": "GitHub Actions CI",
    }

    # Key files to always read
    KEY_FILES: list[str] = [
        "README.md",
        "README.rst",
        "README.txt",
        "README",
        "CLAUDE.md",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env.example",
        "TODO.md",
    ]

    # Extensions to count for language detection
    LANG_EXTENSIONS: dict[str, str] = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript/React",
        ".jsx": "JavaScript/React",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".rb": "Ruby",
        ".php": "PHP",
        ".c": "C",
        ".cpp": "C++",
        ".cs": "C#",
        ".swift": "Swift",
        ".kt": "Kotlin",
    }

    # Directories to skip
    SKIP_DIRS: set[str] = {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",
        ".cargo",
        "vendor",
        ".tox",
        "eggs",
        ".eggs",
    }

    def index(self, path: Path) -> ProjectIndex:
        """Index a project directory.

        Args:
            path: Path to the project root directory.

        Returns:
            ProjectIndex with extracted knowledge.
        """
        path = path.resolve()

        if not path.is_dir():
            logger.warning("Not a directory: %s", path)
            return ProjectIndex(
                name=path.name,
                root_path=str(path),
            )

        # Detect project name from package config or directory name
        name = self._detect_name(path)

        # Scan for tech stack indicators (config files)
        tech_stack = self._detect_tech_stack(path)

        # Count file extensions for language detection
        ext_counts: dict[str, int] = {}
        total_files = self._count_files(path, ext_counts)

        # Add primary languages from file extension counts
        lang_tech = self._langs_from_extensions(ext_counts)
        for lang in lang_tech:
            if lang not in tech_stack:
                tech_stack.append(lang)

        # Read key files
        key_files = self._read_key_files(path)

        # Extract README content
        readme_content = self._extract_readme(path)

        # Build directory structure summary
        structure_summary = self._build_structure(path)

        # Extract dependencies
        dependencies = self._extract_dependencies(path)

        # Detect patterns
        patterns = self._detect_patterns(path, dependencies, ext_counts)

        idx = ProjectIndex(
            name=name,
            root_path=str(path),
            tech_stack=tech_stack,
            key_files=key_files,
            readme_content=readme_content,
            structure_summary=structure_summary,
            dependencies=dependencies,
            patterns=patterns,
            total_files=total_files,
        )

        logger.info(
            "Indexed project '%s': %d files, tech=%s",
            name,
            total_files,
            tech_stack,
        )
        return idx

    def index_multiple(self, paths: list[Path]) -> list[ProjectIndex]:
        """Index multiple project directories.

        Args:
            paths: List of project root directory paths.

        Returns:
            List of ProjectIndex for each valid directory.
        """
        return [self.index(p) for p in paths]

    def _detect_name(self, path: Path) -> str:
        """Detect project name from package config or directory name.

        Tries pyproject.toml, package.json, Cargo.toml first,
        then falls back to the directory name.
        """
        # Try pyproject.toml
        pyproject = path / "pyproject.toml"
        if pyproject.is_file():
            try:
                import tomllib

                data = tomllib.loads(
                    pyproject.read_text(encoding="utf-8", errors="replace")
                )
                name = data.get("project", {}).get("name")
                if name:
                    return str(name)
            except Exception:
                pass

        # Try package.json
        pkg_json = path / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(
                    pkg_json.read_text(encoding="utf-8", errors="replace")
                )
                name = data.get("name")
                if name:
                    return str(name)
            except Exception:
                pass

        # Try Cargo.toml
        cargo = path / "Cargo.toml"
        if cargo.is_file():
            try:
                import tomllib

                data = tomllib.loads(
                    cargo.read_text(encoding="utf-8", errors="replace")
                )
                name = data.get("package", {}).get("name")
                if name:
                    return str(name)
            except Exception:
                pass

        return path.name

    def _detect_tech_stack(self, path: Path) -> list[str]:
        """Detect tech stack from config files and directory patterns."""
        tech: list[str] = []
        seen: set[str] = set()

        # Check file-based indicators
        for filename, tech_name in self.TECH_INDICATORS.items():
            if (path / filename).exists() and tech_name not in seen:
                tech.append(tech_name)
                seen.add(tech_name)

        # Check directory-based indicators
        for dir_pattern, tech_name in self.TECH_DIR_INDICATORS.items():
            if (path / dir_pattern).exists() and tech_name not in seen:
                tech.append(tech_name)
                seen.add(tech_name)

        return tech

    def _should_skip(self, name: str) -> bool:
        """Check if a directory should be skipped."""
        # Check direct name match
        if name in self.SKIP_DIRS:
            return True
        # Check egg-info pattern
        if name.endswith(".egg-info"):
            return True
        return False

    def _count_files(
        self, path: Path, ext_counts: dict[str, int], depth: int = 0
    ) -> int:
        """Count files and track extensions, respecting SKIP_DIRS.

        Args:
            path: Directory to scan.
            ext_counts: Mutable dict to accumulate extension counts.
            depth: Current recursion depth (max 10 to avoid deep trees).

        Returns:
            Total number of files found.
        """
        if depth > 10:
            return 0

        total = 0
        try:
            for entry in path.iterdir():
                if entry.is_file():
                    total += 1
                    ext = entry.suffix.lower()
                    if ext in self.LANG_EXTENSIONS:
                        ext_counts[ext] = ext_counts.get(ext, 0) + 1
                elif entry.is_dir() and not self._should_skip(entry.name):
                    total += self._count_files(entry, ext_counts, depth + 1)
        except PermissionError:
            pass

        return total

    def _langs_from_extensions(self, ext_counts: dict[str, int]) -> list[str]:
        """Determine primary languages from file extension counts.

        Returns languages that have at least 5% of counted source files,
        sorted by count descending.
        """
        if not ext_counts:
            return []

        total = sum(ext_counts.values())
        threshold = max(1, int(total * 0.05))

        langs: list[tuple[str, int]] = []
        seen: set[str] = set()
        for ext, count in ext_counts.items():
            if count >= threshold:
                lang = self.LANG_EXTENSIONS.get(ext, "")
                if lang and lang not in seen:
                    langs.append((lang, count))
                    seen.add(lang)

        langs.sort(key=lambda x: x[1], reverse=True)
        return [lang for lang, _ in langs]

    def _read_key_files(self, path: Path) -> dict[str, str]:
        """Read key files and return path -> summary mapping."""
        key_files: dict[str, str] = {}

        for filename in self.KEY_FILES:
            filepath = path / filename
            if filepath.is_file():
                try:
                    lines = filepath.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[:_MAX_CONFIG_LINES]
                    content = "\n".join(lines).strip()
                    if content:
                        # Store a short summary (first 200 chars)
                        summary = content[:200]
                        if len(content) > 200:
                            summary += "..."
                        key_files[filename] = summary
                except OSError as exc:
                    logger.debug("Failed to read %s: %s", filepath, exc)

        return key_files

    def _extract_readme(self, path: Path) -> str:
        """Extract README content, truncated at max chars."""
        for name in ("README.md", "README.rst", "README.txt", "README"):
            readme = path / name
            if readme.is_file():
                try:
                    content = readme.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                    if len(content) > _MAX_README_CHARS:
                        content = content[:_MAX_README_CHARS] + "\n... (truncated)"
                    return content
                except OSError:
                    pass
        return ""

    def _build_structure(self, path: Path, depth: int = 0, max_depth: int = 2) -> str:
        """Build a directory tree summary (top N levels).

        Args:
            path: Directory to summarize.
            depth: Current depth.
            max_depth: Maximum depth to recurse.

        Returns:
            Formatted directory tree string.
        """
        if depth > max_depth:
            return ""

        lines: list[str] = []
        indent = "  " * depth

        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return ""

        for entry in entries:
            if entry.is_dir():
                if self._should_skip(entry.name):
                    continue
                lines.append(f"{indent}{entry.name}/")
                sub = self._build_structure(entry, depth + 1, max_depth)
                if sub:
                    lines.append(sub)
            elif depth < max_depth:
                # Only list files in the top levels
                lines.append(f"{indent}{entry.name}")

        return "\n".join(lines)

    def _extract_dependencies(self, path: Path) -> list[str]:
        """Extract dependency names from package config files."""
        deps: list[str] = []

        # pyproject.toml
        pyproject = path / "pyproject.toml"
        if pyproject.is_file():
            deps.extend(self._deps_from_pyproject(pyproject))

        # requirements.txt
        reqs = path / "requirements.txt"
        if reqs.is_file():
            deps.extend(self._deps_from_requirements(reqs))

        # package.json
        pkg = path / "package.json"
        if pkg.is_file():
            deps.extend(self._deps_from_package_json(pkg))

        # Cargo.toml
        cargo = path / "Cargo.toml"
        if cargo.is_file():
            deps.extend(self._deps_from_cargo(cargo))

        # go.mod
        gomod = path / "go.mod"
        if gomod.is_file():
            deps.extend(self._deps_from_gomod(gomod))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for dep in deps:
            if dep not in seen:
                seen.add(dep)
                unique.append(dep)

        return unique

    def _deps_from_pyproject(self, path: Path) -> list[str]:
        """Extract dependency names from pyproject.toml."""
        try:
            import tomllib

            data = tomllib.loads(
                path.read_text(encoding="utf-8", errors="replace")
            )
            raw_deps = data.get("project", {}).get("dependencies", [])
            return [self._parse_dep_name(d) for d in raw_deps if d.strip()]
        except Exception:
            return []

    def _deps_from_requirements(self, path: Path) -> list[str]:
        """Extract dependency names from requirements.txt."""
        deps: list[str] = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    deps.append(self._parse_dep_name(line))
        except OSError:
            pass
        return deps

    def _deps_from_package_json(self, path: Path) -> list[str]:
        """Extract dependency names from package.json."""
        try:
            data = json.loads(
                path.read_text(encoding="utf-8", errors="replace")
            )
            deps: list[str] = []
            for section in ("dependencies", "devDependencies"):
                section_deps = data.get(section, {})
                if isinstance(section_deps, dict):
                    deps.extend(section_deps.keys())
            return deps
        except Exception:
            return []

    def _deps_from_cargo(self, path: Path) -> list[str]:
        """Extract dependency names from Cargo.toml."""
        try:
            import tomllib

            data = tomllib.loads(
                path.read_text(encoding="utf-8", errors="replace")
            )
            return list(data.get("dependencies", {}).keys())
        except Exception:
            return []

    def _deps_from_gomod(self, path: Path) -> list[str]:
        """Extract dependency module paths from go.mod."""
        deps: list[str] = []
        try:
            in_require = False
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("require ("):
                    in_require = True
                    continue
                if in_require:
                    if line == ")":
                        in_require = False
                        continue
                    parts = line.split()
                    if parts:
                        deps.append(parts[0])
                elif line.startswith("require "):
                    parts = line.split()
                    if len(parts) >= 2:
                        deps.append(parts[1])
        except OSError:
            pass
        return deps

    def _parse_dep_name(self, dep_string: str) -> str:
        """Parse a dependency name from a version-pinned string.

        Handles formats like:
        - 'anthropic>=0.40.0'
        - 'fastapi[standard]>=0.115.0'
        - 'some-package==1.0.0'
        """
        # Strip extras like [standard]
        name = dep_string.split("[")[0]
        # Strip version specifiers
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", ";"):
            name = name.split(sep)[0]
        return name.strip()

    def _detect_patterns(
        self,
        path: Path,
        dependencies: list[str],
        ext_counts: dict[str, int],
    ) -> list[str]:
        """Detect code patterns and conventions."""
        patterns: list[str] = []

        # Testing frameworks
        if (path / "conftest.py").exists() or (path / "tests" / "conftest.py").exists():
            patterns.append("Uses pytest")
        if (path / "jest.config.js").exists() or (path / "jest.config.ts").exists():
            patterns.append("Uses Jest")
        if (path / "vitest.config.ts").exists() or (path / "vitest.config.js").exists():
            patterns.append("Uses Vitest")

        # Linting/formatting
        if (path / ".eslintrc.js").exists() or (path / ".eslintrc.json").exists():
            patterns.append("Uses ESLint")
        if (path / "ruff.toml").exists():
            patterns.append("Uses Ruff linter")
        elif any("ruff" in str(d).lower() for d in dependencies):
            patterns.append("Uses Ruff linter")

        # Check pyproject.toml for ruff config
        pyproject = path / "pyproject.toml"
        if pyproject.is_file():
            try:
                content = pyproject.read_text(encoding="utf-8", errors="replace")
                if "[tool.ruff" in content:
                    if "Uses Ruff linter" not in patterns:
                        patterns.append("Uses Ruff linter")
                if "[tool.mypy" in content:
                    patterns.append("Uses mypy type checking")
            except OSError:
                pass

        # Framework detection from dependencies
        dep_lower = {d.lower() for d in dependencies}
        framework_map = {
            "fastapi": "Uses FastAPI",
            "django": "Uses Django",
            "flask": "Uses Flask",
            "react": "Uses React",
            "react-dom": "Uses React",
            "vue": "Uses Vue.js",
            "next": "Uses Next.js",
            "express": "Uses Express.js",
            "anthropic": "Uses Anthropic API",
            "openai": "Uses OpenAI API",
            "torch": "Uses PyTorch",
            "tensorflow": "Uses TensorFlow",
            "sqlalchemy": "Uses SQLAlchemy",
            "prisma": "Uses Prisma ORM",
        }
        for dep, pattern in framework_map.items():
            if dep in dep_lower and pattern not in patterns:
                patterns.append(pattern)

        # Docker
        if (path / "Dockerfile").exists():
            patterns.append("Docker deployment")
        if (path / "docker-compose.yml").exists() or (path / "docker-compose.yaml").exists():
            patterns.append("Docker Compose setup")

        # CI/CD
        if (path / ".github" / "workflows").is_dir():
            patterns.append("GitHub Actions CI/CD")
        if (path / ".gitlab-ci.yml").exists():
            patterns.append("GitLab CI/CD")

        # Monorepo indicators
        if (path / "lerna.json").exists() or (path / "pnpm-workspace.yaml").exists():
            patterns.append("Monorepo structure")

        return patterns
