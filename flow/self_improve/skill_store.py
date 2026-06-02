"""Version-controlled skill/prompt store backed by git."""
from __future__ import annotations

import subprocess
from pathlib import Path
from datetime import datetime


class SkillStore:
    """Git-backed store for versioned skill files and system prompts."""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir).resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_git()

    def _ensure_git(self) -> None:
        """Initialize git repo if not already present."""
        git_dir = self.skills_dir / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=self.skills_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m", "flow: init skill store"],
                    cwd=self.skills_dir,
                    check=False,
                    capture_output=True,
                )
            except Exception:
                pass

    def save(self, name: str, content: str, message: str | None = None) -> bool:
        """Save a skill file and commit it."""
        path = self.skills_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        commit_msg = message or f"flow: update {name} [{datetime.utcnow().strftime('%Y-%m-%dT%H:%M')}]"
        try:
            subprocess.run(["git", "add", str(path)], cwd=self.skills_dir, check=False, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.skills_dir,
                check=False,
                capture_output=True,
            )
            return True
        except Exception:
            return False

    def load(self, name: str) -> str | None:
        """Load a skill file by name."""
        path = self.skills_dir / name
        if path.exists():
            return path.read_text()
        return None

    def history(self, name: str, n: int = 10) -> list[dict]:
        """Return the git log for a skill file."""
        try:
            result = subprocess.run(
                ["git", "log", f"-{n}", "--pretty=format:%H|%ai|%s", "--", name],
                cwd=self.skills_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            entries = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    entries.append({"hash": parts[0], "date": parts[1], "message": parts[2]})
            return entries
        except Exception:
            return []

    def revert(self, name: str, commit_hash: str) -> bool:
        """Revert a skill file to a specific git commit."""
        try:
            result = subprocess.run(
                ["git", "show", f"{commit_hash}:{name}"],
                cwd=self.skills_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            content = result.stdout
            return self.save(name, content, message=f"flow: revert {name} to {commit_hash[:8]}")
        except Exception:
            return False

    def list_skills(self) -> list[str]:
        """List all skill files."""
        return [
            str(p.relative_to(self.skills_dir))
            for p in self.skills_dir.rglob("*")
            if p.is_file() and ".git" not in str(p)
        ]
