"""Tests for the project indexer and knowledge store.

Tests cover:
- ProjectIndexer: tech stack detection, dependency extraction,
  README parsing, directory structure, pattern detection
- KnowledgeStore: save/load/list/delete, context string generation
- Edge cases: missing directories, empty projects, multiple projects
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zoom_auto.persona.knowledge_store import KnowledgeStore
from zoom_auto.persona.sources.project import ProjectIndex, ProjectIndexer

# --- Fixtures ---


@pytest.fixture
def indexer() -> ProjectIndexer:
    """Create a ProjectIndexer instance."""
    return ProjectIndexer()


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a minimal Python project structure."""
    proj = tmp_path / "my-python-app"
    proj.mkdir()

    # pyproject.toml
    (proj / "pyproject.toml").write_text(
        '[project]\n'
        'name = "my-python-app"\n'
        'version = "0.1.0"\n'
        'dependencies = [\n'
        '    "fastapi>=0.115.0",\n'
        '    "anthropic>=0.40.0",\n'
        '    "pydantic>=2.10.0",\n'
        ']\n'
        "\n"
        "[tool.ruff]\n"
        'target-version = "py311"\n'
        "\n"
        "[tool.mypy]\n"
        "python_version = 3.11\n"
    )

    # README.md
    (proj / "README.md").write_text(
        "# My Python App\n\n"
        "A FastAPI application for data processing.\n\n"
        "## Features\n\n"
        "- REST API endpoints\n"
        "- Data validation with Pydantic\n"
    )

    # Source files
    src = proj / "src" / "myapp"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text("# main entry point\n")
    (src / "models.py").write_text("# pydantic models\n")

    # Tests
    tests = proj / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text("# pytest config\n")
    (tests / "test_main.py").write_text("# tests\n")

    # Dockerfile
    (proj / "Dockerfile").write_text("FROM python:3.11-slim\n")

    # GitHub Actions
    workflows = proj / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: CI\n")

    return proj


@pytest.fixture
def node_project(tmp_path: Path) -> Path:
    """Create a minimal Node.js project structure."""
    proj = tmp_path / "my-node-app"
    proj.mkdir()

    # package.json
    (proj / "package.json").write_text(
        json.dumps(
            {
                "name": "my-node-app",
                "version": "1.0.0",
                "dependencies": {
                    "express": "^4.18.0",
                    "react": "^18.0.0",
                    "react-dom": "^18.0.0",
                },
                "devDependencies": {
                    "typescript": "^5.0.0",
                    "vitest": "^1.0.0",
                },
            },
            indent=2,
        )
    )

    # tsconfig.json
    (proj / "tsconfig.json").write_text('{"compilerOptions": {}}\n')

    # README
    (proj / "README.md").write_text(
        "# My Node App\n\nA React frontend with Express backend.\n"
    )

    # Source files
    src = proj / "src"
    src.mkdir()
    (src / "index.ts").write_text("// entry\n")
    (src / "App.tsx").write_text("// react app\n")
    (src / "server.js").write_text("// express server\n")

    # Vitest config
    (proj / "vitest.config.ts").write_text("export default {}\n")

    return proj


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """Create an empty project directory."""
    proj = tmp_path / "empty-project"
    proj.mkdir()
    return proj


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    """Create a KnowledgeStore with a temp directory."""
    return KnowledgeStore(data_dir=tmp_path / "knowledge")


# --- ProjectIndexer: Tech Stack Detection ---


class TestTechStackDetection:
    """Tests for technology stack detection."""

    def test_detect_python(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect Python from pyproject.toml."""
        idx = indexer.index(python_project)
        assert "Python" in idx.tech_stack

    def test_detect_docker(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect Docker from Dockerfile."""
        idx = indexer.index(python_project)
        assert "Docker" in idx.tech_stack

    def test_detect_github_actions(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect GitHub Actions from .github/workflows."""
        idx = indexer.index(python_project)
        assert "GitHub Actions CI" in idx.tech_stack

    def test_detect_nodejs(
        self, indexer: ProjectIndexer, node_project: Path
    ) -> None:
        """Should detect Node.js from package.json."""
        idx = indexer.index(node_project)
        assert "Node.js/JavaScript" in idx.tech_stack

    def test_detect_typescript(
        self, indexer: ProjectIndexer, node_project: Path
    ) -> None:
        """Should detect TypeScript from tsconfig.json."""
        idx = indexer.index(node_project)
        assert "TypeScript" in idx.tech_stack

    def test_detect_from_extensions(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect languages from file extensions."""
        idx = indexer.index(python_project)
        # Python files should be detected from .py extensions
        assert "Python" in idx.tech_stack

    def test_empty_project_no_tech(
        self, indexer: ProjectIndexer, empty_project: Path
    ) -> None:
        """Empty project should have no tech stack."""
        idx = indexer.index(empty_project)
        assert idx.tech_stack == []


# --- ProjectIndexer: Name Detection ---


class TestNameDetection:
    """Tests for project name detection."""

    def test_name_from_pyproject(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should extract name from pyproject.toml."""
        idx = indexer.index(python_project)
        assert idx.name == "my-python-app"

    def test_name_from_package_json(
        self, indexer: ProjectIndexer, node_project: Path
    ) -> None:
        """Should extract name from package.json."""
        idx = indexer.index(node_project)
        assert idx.name == "my-node-app"

    def test_name_from_directory(
        self, indexer: ProjectIndexer, empty_project: Path
    ) -> None:
        """Should fall back to directory name."""
        idx = indexer.index(empty_project)
        assert idx.name == "empty-project"


# --- ProjectIndexer: README ---


class TestReadmeExtraction:
    """Tests for README content extraction."""

    def test_read_readme(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should read README content."""
        idx = indexer.index(python_project)
        assert "My Python App" in idx.readme_content
        assert "FastAPI application" in idx.readme_content

    def test_no_readme(
        self, indexer: ProjectIndexer, empty_project: Path
    ) -> None:
        """Missing README should return empty string."""
        idx = indexer.index(empty_project)
        assert idx.readme_content == ""

    def test_truncate_long_readme(
        self, indexer: ProjectIndexer, tmp_path: Path
    ) -> None:
        """Long READMEs should be truncated."""
        proj = tmp_path / "long-readme"
        proj.mkdir()
        (proj / "README.md").write_text("# Big\n\n" + "x" * 10000)
        idx = indexer.index(proj)
        assert len(idx.readme_content) < 6000
        assert "truncated" in idx.readme_content


# --- ProjectIndexer: Dependencies ---


class TestDependencyExtraction:
    """Tests for dependency extraction."""

    def test_python_deps(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should extract deps from pyproject.toml."""
        idx = indexer.index(python_project)
        assert "fastapi" in idx.dependencies
        assert "anthropic" in idx.dependencies
        assert "pydantic" in idx.dependencies

    def test_node_deps(
        self, indexer: ProjectIndexer, node_project: Path
    ) -> None:
        """Should extract deps from package.json."""
        idx = indexer.index(node_project)
        assert "express" in idx.dependencies
        assert "react" in idx.dependencies
        assert "typescript" in idx.dependencies

    def test_requirements_txt(
        self, indexer: ProjectIndexer, tmp_path: Path
    ) -> None:
        """Should extract deps from requirements.txt."""
        proj = tmp_path / "req-project"
        proj.mkdir()
        (proj / "requirements.txt").write_text(
            "flask>=2.0\n"
            "sqlalchemy\n"
            "# comment\n"
            "gunicorn==21.0\n"
        )
        idx = indexer.index(proj)
        assert "flask" in idx.dependencies
        assert "sqlalchemy" in idx.dependencies
        assert "gunicorn" in idx.dependencies

    def test_no_deps(
        self, indexer: ProjectIndexer, empty_project: Path
    ) -> None:
        """Empty project should have no dependencies."""
        idx = indexer.index(empty_project)
        assert idx.dependencies == []

    def test_deduplication(
        self, indexer: ProjectIndexer, tmp_path: Path
    ) -> None:
        """Dependencies should be deduplicated."""
        proj = tmp_path / "dup-deps"
        proj.mkdir()
        (proj / "pyproject.toml").write_text(
            '[project]\nname = "test"\n'
            'dependencies = ["requests>=2.0"]\n'
        )
        (proj / "requirements.txt").write_text("requests\n")
        idx = indexer.index(proj)
        assert idx.dependencies.count("requests") == 1


# --- ProjectIndexer: Structure ---


class TestStructureSummary:
    """Tests for directory structure summary."""

    def test_structure_includes_dirs(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Structure should include top-level directories."""
        idx = indexer.index(python_project)
        assert "src/" in idx.structure_summary
        assert "tests/" in idx.structure_summary

    def test_structure_skips_excluded(
        self, indexer: ProjectIndexer, tmp_path: Path
    ) -> None:
        """Structure should skip excluded directories."""
        proj = tmp_path / "skip-test"
        proj.mkdir()
        (proj / "src").mkdir()
        (proj / "node_modules").mkdir()
        (proj / "__pycache__").mkdir()
        (proj / "file.py").write_text("")

        idx = indexer.index(proj)
        assert "node_modules" not in idx.structure_summary
        assert "__pycache__" not in idx.structure_summary
        assert "src/" in idx.structure_summary

    def test_empty_structure(
        self, indexer: ProjectIndexer, empty_project: Path
    ) -> None:
        """Empty project should have empty structure."""
        idx = indexer.index(empty_project)
        assert idx.structure_summary == ""


# --- ProjectIndexer: Patterns ---


class TestPatternDetection:
    """Tests for code pattern detection."""

    def test_detect_pytest(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect pytest from conftest.py."""
        idx = indexer.index(python_project)
        assert "Uses pytest" in idx.patterns

    def test_detect_fastapi(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect FastAPI from dependencies."""
        idx = indexer.index(python_project)
        assert "Uses FastAPI" in idx.patterns

    def test_detect_anthropic(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect Anthropic API from dependencies."""
        idx = indexer.index(python_project)
        assert "Uses Anthropic API" in idx.patterns

    def test_detect_docker_pattern(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect Docker deployment pattern."""
        idx = indexer.index(python_project)
        assert "Docker deployment" in idx.patterns

    def test_detect_ci_cd(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect GitHub Actions CI/CD."""
        idx = indexer.index(python_project)
        assert "GitHub Actions CI/CD" in idx.patterns

    def test_detect_ruff(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect Ruff linter from pyproject.toml config."""
        idx = indexer.index(python_project)
        assert "Uses Ruff linter" in idx.patterns

    def test_detect_mypy(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should detect mypy from pyproject.toml config."""
        idx = indexer.index(python_project)
        assert "Uses mypy type checking" in idx.patterns

    def test_detect_vitest(
        self, indexer: ProjectIndexer, node_project: Path
    ) -> None:
        """Should detect Vitest from config file."""
        idx = indexer.index(node_project)
        assert "Uses Vitest" in idx.patterns

    def test_detect_react(
        self, indexer: ProjectIndexer, node_project: Path
    ) -> None:
        """Should detect React from dependencies."""
        idx = indexer.index(node_project)
        assert "Uses React" in idx.patterns

    def test_no_patterns_empty(
        self, indexer: ProjectIndexer, empty_project: Path
    ) -> None:
        """Empty project should have no patterns."""
        idx = indexer.index(empty_project)
        assert idx.patterns == []


# --- ProjectIndexer: File Counting ---


class TestFileCounting:
    """Tests for file counting."""

    def test_count_files(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should count all project files."""
        idx = indexer.index(python_project)
        assert idx.total_files > 0

    def test_skip_excluded_in_count(
        self, indexer: ProjectIndexer, tmp_path: Path
    ) -> None:
        """Should not count files in excluded directories."""
        proj = tmp_path / "count-test"
        proj.mkdir()
        (proj / "real.py").write_text("")
        nm = proj / "node_modules"
        nm.mkdir()
        (nm / "pkg.js").write_text("")

        idx = indexer.index(proj)
        assert idx.total_files == 1  # Only real.py


# --- ProjectIndexer: Multiple Projects ---


class TestIndexMultiple:
    """Tests for indexing multiple projects."""

    def test_index_multiple(
        self,
        indexer: ProjectIndexer,
        python_project: Path,
        node_project: Path,
    ) -> None:
        """Should index multiple projects."""
        indices = indexer.index_multiple([python_project, node_project])
        assert len(indices) == 2
        names = {idx.name for idx in indices}
        assert "my-python-app" in names
        assert "my-node-app" in names


# --- ProjectIndexer: Edge Cases ---


class TestIndexerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_directory(
        self, indexer: ProjectIndexer, tmp_path: Path
    ) -> None:
        """Missing directory should return minimal index."""
        idx = indexer.index(tmp_path / "nonexistent")
        assert idx.name == "nonexistent"
        assert idx.tech_stack == []
        assert idx.total_files == 0

    def test_key_files_read(
        self, indexer: ProjectIndexer, python_project: Path
    ) -> None:
        """Should read key files like pyproject.toml."""
        idx = indexer.index(python_project)
        assert "pyproject.toml" in idx.key_files
        assert "Dockerfile" in idx.key_files
        assert "README.md" in idx.key_files


# --- KnowledgeStore: Save/Load ---


class TestKnowledgeStorePersistence:
    """Tests for KnowledgeStore save and load."""

    def test_save_and_load(self, store: KnowledgeStore) -> None:
        """Should round-trip a ProjectIndex through JSON."""
        index = ProjectIndex(
            name="test-project",
            root_path="/tmp/test",
            tech_stack=["Python", "Docker"],
            key_files={"README.md": "# Test..."},
            readme_content="# Test Project\n\nA test.",
            structure_summary="src/\ntests/",
            dependencies=["fastapi", "pydantic"],
            patterns=["Uses pytest", "Docker deployment"],
            total_files=42,
        )

        path = store.save_index(index)
        assert path.exists()

        loaded = store.load_index("test-project")
        assert loaded is not None
        assert loaded.name == "test-project"
        assert loaded.tech_stack == ["Python", "Docker"]
        assert loaded.dependencies == ["fastapi", "pydantic"]
        assert loaded.patterns == ["Uses pytest", "Docker deployment"]
        assert loaded.total_files == 42

    def test_load_missing(self, store: KnowledgeStore) -> None:
        """Loading nonexistent project should return None."""
        loaded = store.load_index("nonexistent")
        assert loaded is None

    def test_save_overwrites(self, store: KnowledgeStore) -> None:
        """Saving with same name should overwrite."""
        idx1 = ProjectIndex(
            name="test", root_path="/tmp/1", tech_stack=["Python"]
        )
        idx2 = ProjectIndex(
            name="test", root_path="/tmp/2", tech_stack=["Rust"]
        )

        store.save_index(idx1)
        store.save_index(idx2)

        loaded = store.load_index("test")
        assert loaded is not None
        assert loaded.root_path == "/tmp/2"
        assert loaded.tech_stack == ["Rust"]


# --- KnowledgeStore: List/Delete ---


class TestKnowledgeStoreManagement:
    """Tests for listing and deleting project knowledge."""

    def test_list_empty(self, store: KnowledgeStore) -> None:
        """Empty store should return empty list."""
        assert store.list_projects() == []

    def test_list_projects(self, store: KnowledgeStore) -> None:
        """Should list all saved projects."""
        for name in ("alpha", "beta", "gamma"):
            store.save_index(
                ProjectIndex(name=name, root_path=f"/tmp/{name}")
            )

        projects = store.list_projects()
        assert len(projects) == 3
        assert "alpha" in projects
        assert "beta" in projects
        assert "gamma" in projects

    def test_delete_existing(self, store: KnowledgeStore) -> None:
        """Should delete an existing index."""
        store.save_index(
            ProjectIndex(name="doomed", root_path="/tmp/doomed")
        )
        assert store.delete_index("doomed") is True
        assert store.load_index("doomed") is None

    def test_delete_nonexistent(self, store: KnowledgeStore) -> None:
        """Deleting nonexistent project should return False."""
        assert store.delete_index("ghost") is False


# --- KnowledgeStore: Context String ---


class TestContextString:
    """Tests for context string generation."""

    def test_empty_context(self, store: KnowledgeStore) -> None:
        """Empty store should return empty string."""
        assert store.get_context_string() == ""

    def test_context_format(self, store: KnowledgeStore) -> None:
        """Context string should be properly formatted."""
        store.save_index(
            ProjectIndex(
                name="my-api",
                root_path="/tmp/api",
                tech_stack=["Python", "FastAPI"],
                dependencies=["fastapi", "pydantic", "anthropic"],
                patterns=["Uses pytest", "Docker deployment"],
                readme_content="# My API\n\nA REST API for data processing.",
            )
        )

        ctx = store.get_context_string()
        assert "## Project Knowledge" in ctx
        assert "### my-api" in ctx
        assert "Python" in ctx
        assert "FastAPI" in ctx
        assert "Uses pytest" in ctx
        assert "fastapi" in ctx
        assert "REST API" in ctx or "data processing" in ctx

    def test_context_multiple_projects(self, store: KnowledgeStore) -> None:
        """Context should include all projects."""
        store.save_index(
            ProjectIndex(
                name="frontend",
                root_path="/tmp/fe",
                tech_stack=["TypeScript", "React"],
            )
        )
        store.save_index(
            ProjectIndex(
                name="backend",
                root_path="/tmp/be",
                tech_stack=["Python", "FastAPI"],
            )
        )

        ctx = store.get_context_string()
        assert "### frontend" in ctx
        assert "### backend" in ctx
        assert "TypeScript" in ctx
        assert "Python" in ctx

    def test_context_truncation(self, store: KnowledgeStore) -> None:
        """Context should respect max_tokens limit."""
        # Create a project with a lot of data
        store.save_index(
            ProjectIndex(
                name="big-project",
                root_path="/tmp/big",
                tech_stack=["Python"] * 20,
                dependencies=["dep" + str(i) for i in range(100)],
                patterns=["pattern" + str(i) for i in range(50)],
                readme_content="# Big\n\n" + "word " * 1000,
            )
        )

        # Small token limit should truncate
        ctx = store.get_context_string(max_tokens=50)
        assert len(ctx) < 500  # ~50 tokens * 4 chars + some overhead


# --- Integration: Indexer + Store ---


class TestIntegration:
    """Integration tests for indexer and store together."""

    def test_index_and_store(
        self,
        indexer: ProjectIndexer,
        python_project: Path,
        store: KnowledgeStore,
    ) -> None:
        """Full pipeline: index -> store -> load -> context string."""
        # Index
        idx = indexer.index(python_project)
        assert idx.name == "my-python-app"

        # Store
        store.save_index(idx)

        # Load
        loaded = store.load_index("my-python-app")
        assert loaded is not None
        assert loaded.tech_stack == idx.tech_stack
        assert loaded.dependencies == idx.dependencies

        # Context string
        ctx = store.get_context_string()
        assert "my-python-app" in ctx
        assert "Python" in ctx

    def test_index_multiple_and_store(
        self,
        indexer: ProjectIndexer,
        python_project: Path,
        node_project: Path,
        store: KnowledgeStore,
    ) -> None:
        """Index multiple projects and build combined context."""
        indices = indexer.index_multiple([python_project, node_project])

        for idx in indices:
            store.save_index(idx)

        projects = store.list_projects()
        assert len(projects) == 2

        ctx = store.get_context_string()
        assert "my-python-app" in ctx
        assert "my-node-app" in ctx


# --- ProjectIndex serialization ---


class TestProjectIndexSerialization:
    """Tests for ProjectIndex to_dict / from_dict."""

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict should preserve all fields."""
        original = ProjectIndex(
            name="test",
            root_path="/tmp/test",
            tech_stack=["Python"],
            key_files={"README.md": "# Test"},
            readme_content="Hello",
            structure_summary="src/",
            dependencies=["dep1"],
            patterns=["pat1"],
            total_files=10,
        )

        data = original.to_dict()
        restored = ProjectIndex.from_dict(data)

        assert restored.name == original.name
        assert restored.root_path == original.root_path
        assert restored.tech_stack == original.tech_stack
        assert restored.key_files == original.key_files
        assert restored.readme_content == original.readme_content
        assert restored.structure_summary == original.structure_summary
        assert restored.dependencies == original.dependencies
        assert restored.patterns == original.patterns
        assert restored.total_files == original.total_files
