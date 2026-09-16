from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPO_TOP_LEVELS = {
    "artifacts",
    "config",
    "data",
    "docs",
    "outputs",
    "scripts",
    "src",
    "tests",
}
KNOWN_REPO_DIR_NAMES = {"streetlight", "streetlight-paper1-reproducibility"}

PATH_KEYS = {
    "manifest",
    "paper_output_dir",
    "quality",
    "quality_summary",
    "region_map",
    "source_config",
    "source_figure_data_dir",
    "source_results_dir",
    "staging_root",
    "streetlight_config",
}

PATH_KEY_SUFFIXES = ("_dir", "_path", "_root")


def _is_path_key(key: str | None) -> bool:
    if key is None:
        return False
    return key in PATH_KEYS or key.endswith(PATH_KEY_SUFFIXES)


def _relative_to(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _find_containing_git_root(path: Path) -> Path | None:
    probe = path if path.is_dir() else path.parent
    for candidate in (probe, *probe.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _clone_relative_path(path: Path) -> str | None:
    git_root = _find_containing_git_root(path)
    if git_root is None:
        return None
    relative = _relative_to(path, git_root)
    if relative is None:
        return None
    first = Path(relative).parts[0] if Path(relative).parts else ""
    if first not in REPO_TOP_LEVELS:
        return None
    return relative


def _known_repo_name_relative_path(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part not in KNOWN_REPO_DIR_NAMES:
            continue
        relative_parts = parts[index + 1 :]
        if not relative_parts or relative_parts[0] not in REPO_TOP_LEVELS:
            continue
        return Path(*relative_parts).as_posix()
    return None


def repo_display_path(value: str | Path, *, repo_root: Path = ROOT) -> str:
    """Serialize repo-local paths relative to their repo; keep external paths absolute."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()

    resolved_root = repo_root.resolve()
    resolved_path = path.resolve(strict=False)
    relative = _relative_to(resolved_path, resolved_root)
    if relative is not None:
        return relative

    clone_relative = _clone_relative_path(resolved_path)
    if clone_relative is not None:
        return clone_relative

    known_repo_relative = _known_repo_name_relative_path(path)
    if known_repo_relative is not None:
        return known_repo_relative

    return path.as_posix()


def resolve_repo_display_path(value: str | Path, *, repo_root: Path = ROOT) -> Path:
    """Resolve a display path back to an absolute path for readers."""
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_repo_paths(value: Any, *, repo_root: Path = ROOT, key: str | None = None) -> Any:
    """Recursively normalize path-like fields inside JSON-serializable structures."""
    if isinstance(value, dict):
        return {
            item_key: normalize_repo_paths(item_value, repo_root=repo_root, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            normalize_repo_paths(item, repo_root=repo_root, key=key)
            for item in value
        ]
    if isinstance(value, (str, Path)) and _is_path_key(key):
        return repo_display_path(value, repo_root=repo_root)
    return value
