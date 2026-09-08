"""Language stack profiles — LSPs, linters, formatters, and framework options."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

# A profile directory is operator-curated; one file per language is the expected shape, so a
# directory with more entries than this is misconfigured rather than merely large.
PROFILE_FILES_MAX = 256


@dataclass
class LanguageProfile:
    name: str
    extensions: list[str]
    lsp: str
    linter: str
    formatter: str
    test_runner: str
    build_tool: str
    package_manager: str
    frameworks: dict[str, str]  # name -> description
    install_hints: dict[str, str]  # tool -> pacman/cargo/etc command
    project_templates: dict[str, str]  # framework -> init command


BUILTIN_PROFILES: dict[str, LanguageProfile] = {
    "python": LanguageProfile(
        name="Python",
        extensions=[".py"],
        lsp="pylsp / pyright",
        linter="ruff",
        formatter="ruff format / black",
        test_runner="pytest",
        build_tool="hatch / setuptools",
        package_manager="pip / uv",
        frameworks={
            "fastapi": "Async REST API framework with auto-docs",
            "flask": "Lightweight WSGI web framework",
            "django": "Batteries-included web framework",
            "click": "CLI application framework",
            "typer": "FastAPI-inspired CLI framework",
            "langchain": "LLM application framework",
            "streamlit": "Data app / ML dashboard framework",
            "bare": "No framework — pure Python script/module",
        },
        install_hints={
            "pylsp": "python-lsp-server (pip install python-lsp-server)",
            "ruff": "ruff (pip install ruff)",
            "pytest": "pytest (pip install pytest)",
            "uv": "uv (pip install uv)",
        },
        project_templates={
            "fastapi": "uv init {name} && cd {name} && uv add fastapi uvicorn",
            "flask": "uv init {name} && cd {name} && uv add flask",
            "click": "uv init {name} && cd {name} && uv add click",
            "bare": "mkdir -p {name} && cd {name} && uv init",
        },
    ),
    "rust": LanguageProfile(
        name="Rust",
        extensions=[".rs"],
        lsp="rust-analyzer",
        linter="clippy",
        formatter="rustfmt",
        test_runner="cargo test",
        build_tool="cargo",
        package_manager="cargo",
        frameworks={
            "axum": "Ergonomic async web framework (tokio)",
            "actix-web": "High-performance actor-based web framework",
            "clap": "CLI argument parser and application framework",
            "tonic": "gRPC framework",
            "tokio": "Async runtime (bare application)",
            "egui": "Immediate mode GUI",
            "bevy": "ECS game engine",
            "bare": "No framework — pure Rust binary/library",
        },
        install_hints={
            "rust-analyzer": "rust-analyzer (pacman -S rust-analyzer)",
            "cargo": "rust (pacman -S rust)",
            "clippy": "included with rustup",
        },
        project_templates={
            "axum": "cargo new {name} && cd {name} && cargo add axum tokio --features tokio/full",
            "clap": "cargo new {name} && cd {name} && cargo add clap --features derive",
            "bare": "cargo new {name}",
        },
    ),
    "typescript": LanguageProfile(
        name="TypeScript",
        extensions=[".ts", ".tsx"],
        lsp="typescript-language-server",
        linter="eslint",
        formatter="prettier",
        test_runner="vitest / jest",
        build_tool="tsc / vite / esbuild",
        package_manager="npm / pnpm / bun",
        frameworks={
            "nextjs": "React full-stack framework with SSR/SSG",
            "react": "UI component library (Vite)",
            "express": "Node.js REST API framework",
            "fastify": "High-performance Node.js framework",
            "hono": "Ultrafast edge/node web framework",
            "nestjs": "Angular-inspired backend framework",
            "bare": "Pure TypeScript — node script or library",
        },
        install_hints={
            "tsserver": "npm i -g typescript typescript-language-server",
            "eslint": "npm i -g eslint",
        },
        project_templates={
            "nextjs": "npx create-next-app@latest {name} --typescript",
            "react": "npm create vite@latest {name} -- --template react-ts",
            "express": (
                "mkdir {name} && cd {name} && npm init -y && "
                "npm i express typescript @types/express ts-node"
            ),
            "bare": (
                "mkdir {name} && cd {name} && npm init -y && npm i typescript && npx tsc --init"
            ),
        },
    ),
    "go": LanguageProfile(
        name="Go",
        extensions=[".go"],
        lsp="gopls",
        linter="golangci-lint",
        formatter="gofmt / goimports",
        test_runner="go test",
        build_tool="go build",
        package_manager="go mod",
        frameworks={
            "gin": "Fast HTTP web framework",
            "echo": "High performance web framework",
            "fiber": "Express-inspired web framework",
            "cobra": "CLI framework (used by kubectl, git)",
            "bare": "Pure Go — no framework",
        },
        install_hints={
            "gopls": "gopls (go install golang.org/x/tools/gopls@latest)",
            "go": "go (pacman -S go)",
        },
        project_templates={
            "gin": (
                "mkdir {name} && cd {name} && go mod init {name} && go get github.com/gin-gonic/gin"
            ),
            "cobra": (
                "mkdir {name} && cd {name} && go mod init {name} && "
                "go install github.com/spf13/cobra-cli@latest && cobra-cli init"
            ),
            "bare": "mkdir {name} && cd {name} && go mod init {name}",
        },
    ),
    "lua": LanguageProfile(
        name="Lua",
        extensions=[".lua"],
        lsp="lua-language-server",
        linter="luacheck",
        formatter="stylua",
        test_runner="busted",
        build_tool="luarocks",
        package_manager="luarocks",
        frameworks={
            "neovim-plugin": "Neovim plugin (Lua API)",
            "love2d": "2D game framework",
            "openresty": "Nginx/Lua web server",
            "bare": "Pure Lua script",
        },
        install_hints={
            "lua-language-server": "lua-language-server (pacman -S lua-language-server)",
            "stylua": "stylua (pacman -S stylua or cargo install stylua)",
            "luacheck": "luacheck (luarocks install luacheck)",
        },
        project_templates={
            "neovim-plugin": "mkdir -p {name}/lua/{name} && touch {name}/lua/{name}/init.lua",
            "bare": "mkdir {name} && touch {name}/main.lua",
        },
    ),
    "bash": LanguageProfile(
        name="Bash",
        extensions=[".sh", ".bash"],
        lsp="bash-language-server",
        linter="shellcheck",
        formatter="shfmt",
        test_runner="bats",
        build_tool="make",
        package_manager="N/A",
        frameworks={
            "systemd-service": "systemd service wrapper script",
            "cli-tool": "Command-line tool with argument parsing",
            "deployment": "Deployment/provisioning script",
            "bare": "General purpose shell script",
        },
        install_hints={
            "bash-language-server": "bash-language-server (npm i -g bash-language-server)",
            "shellcheck": "shellcheck (pacman -S shellcheck)",
            "shfmt": "shfmt (pacman -S shfmt)",
        },
        project_templates={
            "bare": "mkdir {name} && touch {name}/main.sh && chmod +x {name}/main.sh",
        },
    ),
    "c": LanguageProfile(
        name="C",
        extensions=[".c", ".h"],
        lsp="clangd",
        linter="clang-tidy",
        formatter="clang-format",
        test_runner="cmocka / unity",
        build_tool="cmake / meson",
        package_manager="pacman",
        frameworks={
            "cmake-lib": "CMake library project",
            "cmake-bin": "CMake binary project",
            "meson": "Meson build system project",
            "bare": "Single-file C program",
        },
        install_hints={
            "clangd": "clang (pacman -S clang)",
            "cmake": "cmake (pacman -S cmake)",
        },
        project_templates={
            "cmake-bin": "mkdir {name} && cd {name} && cmake -DCMAKE_BUILD_TYPE=Debug ..",
            "bare": "touch {name}.c",
        },
    ),
    "nix": LanguageProfile(
        name="Nix",
        extensions=[".nix"],
        lsp="nil / nixd",
        linter="statix",
        formatter="alejandra / nixfmt",
        test_runner="nix flake check",
        build_tool="nix build",
        package_manager="nix",
        frameworks={
            "flake": "Nix flake with outputs",
            "home-manager": "Home Manager module",
            "nixos-module": "NixOS module",
            "bare": "Simple nix expression",
        },
        install_hints={
            "nil": "nil (nix profile install nixpkgs#nil)",
            "alejandra": "alejandra (nix run nixpkgs#alejandra)",
        },
        project_templates={
            "flake": "nix flake init",
            "bare": "echo '{}' > default.nix",
        },
    ),
}


def load_profiles(skills_dir: Path | None = None) -> dict[str, LanguageProfile]:
    """Load profiles from skills/language_profiles/*.toml and merge with builtins."""
    profiles = copy.deepcopy(BUILTIN_PROFILES)

    if skills_dir is None:
        skills_dir = files("flow").joinpath("skills/language_profiles")

    if not skills_dir.is_dir():
        return profiles

    for scanned, toml_file in enumerate(skills_dir.iterdir()):
        if scanned >= PROFILE_FILES_MAX:
            break
        if not toml_file.name.endswith(".toml"):
            continue
        lang = toml_file.name.removesuffix(".toml")
        try:
            with toml_file.open("rb") as f:
                data = tomllib.load(f)
            # Merge into existing or create new profile
            if lang in profiles:
                for k, v in data.items():
                    if hasattr(profiles[lang], k):
                        setattr(profiles[lang], k, v)
            else:
                profiles[lang] = LanguageProfile(
                    name=data.get("name", lang.title()),
                    extensions=data.get("extensions", [f".{lang}"]),
                    lsp=data.get("lsp", ""),
                    linter=data.get("linter", ""),
                    formatter=data.get("formatter", ""),
                    test_runner=data.get("test_runner", ""),
                    build_tool=data.get("build_tool", ""),
                    package_manager=data.get("package_manager", ""),
                    frameworks=data.get("frameworks", {}),
                    install_hints=data.get("install_hints", {}),
                    project_templates=data.get("project_templates", {}),
                )
        except Exception:
            pass

    return profiles


def profile_summary(profile: LanguageProfile) -> str:
    """Format a profile for inclusion in LLM context."""
    frameworks = "\n".join(f"  - {k}: {v}" for k, v in profile.frameworks.items())
    return f"""\
Language: {profile.name}
LSP: {profile.lsp}
Linter: {profile.linter}
Formatter: {profile.formatter}
Test Runner: {profile.test_runner}
Build Tool: {profile.build_tool}
Package Manager: {profile.package_manager}
Available Frameworks:
{frameworks}
"""
