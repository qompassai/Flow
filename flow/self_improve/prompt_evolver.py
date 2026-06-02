"""Prompt refinement engine — uses LLM to improve prompts based on feedback."""
from __future__ import annotations

import json
from pathlib import Path

from flow.self_improve.feedback import FeedbackStore


class PromptEvolver:
    """Uses feedback data to iteratively improve system prompts."""

    def __init__(self, llm, feedback: FeedbackStore, skills_dir: Path):
        self.llm = llm
        self.feedback = feedback
        self.skills_dir = skills_dir
        self.prompt_path = skills_dir / "system_prompt.md"

    def should_evolve(self, min_samples: int = 5, min_avg_below: float = 3.5) -> bool:
        """Check if we have enough negative feedback to trigger evolution."""
        avg = self.feedback.average_rating()
        low = self.feedback.get_low_rated(limit=1)
        return len(low) >= min_samples or (avg > 0 and avg < min_avg_below)

    def evolve(self) -> str | None:
        """Generate an improved system prompt based on feedback."""
        low_rated = self.feedback.get_low_rated(limit=5)
        if not low_rated:
            return None

        current_prompt = ""
        if self.prompt_path.exists():
            current_prompt = self.prompt_path.read_text()

        issues = "\n".join(
            f"- Rating {f['rating']}/5: {f['comment']} (Outcome: {f['outcome']})"
            for f in low_rated
        )

        evolution_prompt = f"""You are improving an AI system prompt based on user feedback.

Current System Prompt:
{current_prompt[:2000]}

User Feedback Issues (low ratings):
{issues}

Write an improved system prompt that addresses these issues.
Output ONLY the improved prompt text, no explanation.
"""
        resp = self.llm.chat(messages=[
            {"role": "system", "content": "You are a prompt engineering expert."},
            {"role": "user", "content": evolution_prompt},
        ])

        improved = resp["choices"][0]["message"]["content"]

        # Save with git versioning
        self._save_evolved_prompt(improved)
        return improved

    def _save_evolved_prompt(self, new_prompt: str) -> None:
        """Save the evolved prompt and commit to git."""
        import subprocess
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_path.write_text(new_prompt)

        # Git commit the improvement
        try:
            subprocess.run(["git", "add", str(self.prompt_path)], cwd=self.skills_dir.parent, check=False)
            subprocess.run(
                ["git", "commit", "-m", "flow: auto-evolved system prompt based on feedback"],
                cwd=self.skills_dir.parent, check=False
            )
        except Exception:
            pass  # Git not required
