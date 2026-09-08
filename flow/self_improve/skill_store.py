"""Read-only compatibility store. No implicit repo creation, hooks or Git commits."""

from flow.workspace import Workspace


class SkillStore:
    def __init__(self, skills_dir):
        self.workspace = Workspace(skills_dir)

    def load(self, name):
        try:
            return self.workspace.read(name)["content"]
        except FileNotFoundError:
            return None

    def list_skills(self):
        return self.workspace.list()["files"]

    def save(self, *args, **kwargs):
        raise RuntimeError("Automatic skill mutation is disabled; edit manually")

    def revert(self, *args, **kwargs):
        raise RuntimeError("Automatic Git mutation is disabled; revert manually")

    def history(self, *args, **kwargs):
        return []
