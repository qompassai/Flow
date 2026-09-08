"""Automatic executable prompt mutation is no longer part of the trusted-workspace product."""


class PromptEvolver:
    def __init__(self, *args, **kwargs):
        pass

    def evolve(self):
        raise RuntimeError(
            "Automatic prompt evolution and implicit Git commits are disabled. "
            "Review and edit operator-owned prompts manually."
        )
