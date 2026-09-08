"""Compatibility name for the shared safe runtime, not the former unrestricted ReAct loop."""

from flow.runtime import Runtime


class Orchestrator:
    def __init__(self, runtime: Runtime):
        if not isinstance(runtime, Runtime):
            raise TypeError("Orchestrator now requires an initialized safe Runtime")
        self.runtime = runtime

    def run(self, user_message: str) -> dict:
        return self.runtime.run(user_message)
