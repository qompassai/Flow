"""Scaffolding uses the same confined, verified runtime as every other task."""

from flow.runtime import Runtime


class AppGenerator:
    def __init__(self, runtime: Runtime):
        if not isinstance(runtime, Runtime):
            raise TypeError("AppGenerator now requires an initialized safe Runtime")
        self.runtime = runtime

    def generate(self, request, language="", framework="", project_name="", profile=None):
        return self.runtime.run(
            f"Build {project_name!r} using {language} {framework}. "
            f"Use workspace-relative file tools and required checks. Request: {request}"
        )
