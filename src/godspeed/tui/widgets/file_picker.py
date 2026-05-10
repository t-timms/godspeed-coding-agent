"""File picker widget — fuzzy @-mention file search dropdown."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar

from textual.widgets import Label, ListItem, ListView

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".godspeed",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}

EXCLUDE_FILES = {
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    "*.dylib",
    "*.bin",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp4",
    "*.mp3",
    "*.wav",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.xz",
    "*.bz2",
    "*.7z",
    "*.pdf",
    "*.lock",
    "*.min.js",
    "*.min.css",
}


class FilePicker(ListView):
    """Fuzzy file search dropdown shown during @-mention input."""

    BINDINGS: ClassVar[list] = [
        ("escape", "dismiss", "Dismiss"),
        ("tab", "select", "Select"),
    ]

    class Selected(ListView.Selected):
        """Posted when the user selects a file."""
        def __init__(self, item: str) -> None:
            super().__init__(ListView())
            self.item = item

    def __init__(self, project_dir: Path, max_items: int = 20) -> None:
        super().__init__(id="file-picker")
        self._project_dir = project_dir.resolve()
        self._max_items = max_items
        self._all_files: list[str] = []
        self._query: str = ""
        self.display = False
        self.styles.height = "auto"
        self.styles.max_height = 12

    def on_mount(self) -> None:
        self._scan_files()
        self.display = False

    def _scan_files(self) -> None:
        files: list[str] = []
        try:
            for entry in self._project_dir.rglob("*"):
                if entry.is_dir():
                    if entry.name in EXCLUDE_DIRS or any(
                        fnmatch(entry.name, p) for p in EXCLUDE_DIRS if "*" in p
                    ):
                        continue
                    continue
                rel = str(entry.relative_to(self._project_dir))
                parts_set = set(rel.replace("\\", "/").lower().split("/"))
                if parts_set & EXCLUDE_DIRS:
                    continue
                rel_lower = rel.lower()
                if any(fnmatch(rel_lower, p) for p in EXCLUDE_FILES):
                    continue
                files.append(rel)
        except OSError:
            return
        self._all_files = sorted(files, key=str.lower)

    def _find_matches(self, query: str) -> list[str]:
        query = query.strip().lower()
        if not self._all_files:
            return []
        if not query:
            return self._all_files[: self._max_items]
        matches: list[tuple[int, str]] = []
        for f in self._all_files:
            f_lower = f.lower()
            if query in f_lower:
                score = f_lower.find(query)
                matches.append((score, f))
            elif all(part in f_lower for part in query.split()):
                matches.append((len(f), f))
        matches.sort(key=lambda x: (x[0], len(x[1])))
        return [m[1] for m in matches[: self._max_items]]

    def filter_for(self, query: str) -> None:
        query = query.strip().lower()
        self._query = query
        if self.children:
            self.clear()
        if not query and not self._all_files:
            self.display = False
            return
        if not self._all_files:
            self._scan_files()
            if not self._all_files:
                self.display = False
                return
        top = self._find_matches(query)
        self.display = bool(top)
        for path in top:
            self.append(ListItem(Label(path)))

    def action_select(self) -> None:
        if self.index is not None and 0 <= self.index < len(self):
            item = self.children[self.index]
            label = item.query_one(Label)
            self.post_message(self.Selected(str(label.renderable)))

    def action_dismiss(self) -> None:
        self.display = False
