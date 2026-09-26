//! Embedded language profiles with operator overrides.
//!
//! Ports `flow/codegen/language_profiles.py`. The eight bundled profiles
//! ship as TOML files embedded with `include_str!`, so they are available
//! outside the repository and never depend on the current working
//! directory. The TOML files are the effective builtins: verified against
//! the real Python on 2026-09-26, every shipped TOML fully specifies its
//! language, so parsing the TOML alone is field-for-field identical to
//! Python's `load_profiles()` — the six languages whose TOML ships in
//! Python (`flow/skills/language_profiles/*.toml`) carry the effective
//! values, i.e. the shipped file content plus the inherited
//! `project_templates` the Python dataclass layer would have supplied,
//! and `c`/`nix` carry the `BUILTIN_PROFILES` dataclass values (no TOML
//! ships for them). The dataclass layer is therefore not ported.
//!
//! Parsing preserves document order (`toml_edit`, the same order Python's
//! `tomllib` yields) so [`profile_summary`] renders frameworks exactly as
//! the Python implementation did.
//!
//! Operator overrides live in a skills directory: each `*.toml` file
//! either overrides fields of a bundled language or defines a new one.
//! At most [`PROFILE_FILES_MAX`] directory entries are scanned; more is an
//! error, not a silent truncation.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::CodegenError;

/// At most this many directory entries are scanned for operator profiles.
/// Mirrors Python's `PROFILE_FILES_MAX = 256`; exceeding it is an error.
pub const PROFILE_FILES_MAX: usize = 256;

/// One operator profile file is at most this many bytes; larger files are
/// skipped like malformed ones.
pub const PROFILE_TOML_BYTES_MAX: u64 = 1024 * 1024;

/// A language profile: toolchain names plus ordered framework, install
/// hint, and project template tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageProfile {
    /// Display name, e.g. `"Python"`.
    pub name: String,
    /// File extensions, e.g. `[".py"]`.
    pub extensions: Vec<String>,
    /// Language server, e.g. `"pylsp / pyright"`.
    pub lsp: String,
    /// Linter, e.g. `"ruff"`.
    pub linter: String,
    /// Formatter, e.g. `"ruff format / black"`.
    pub formatter: String,
    /// Test runner, e.g. `"pytest"`.
    pub test_runner: String,
    /// Build tool, e.g. `"hatch / setuptools"`.
    pub build_tool: String,
    /// Package manager, e.g. `"pip / uv"`.
    pub package_manager: String,
    /// Frameworks in document order: `(name, description)`.
    pub frameworks: Vec<(String, String)>,
    /// Install hints in document order: `(tool, command)`.
    pub install_hints: Vec<(String, String)>,
    /// Project templates in document order: `(name, description)`.
    pub project_templates: Vec<(String, String)>,
}

const PYTHON_TOML: &str = include_str!("../profiles/python.toml");
const RUST_TOML: &str = include_str!("../profiles/rust.toml");
const GO_TOML: &str = include_str!("../profiles/go.toml");
const TYPESCRIPT_TOML: &str = include_str!("../profiles/typescript.toml");
const LUA_TOML: &str = include_str!("../profiles/lua.toml");
const BASH_TOML: &str = include_str!("../profiles/bash.toml");
const C_TOML: &str = include_str!("../profiles/c.toml");
const NIX_TOML: &str = include_str!("../profiles/nix.toml");

/// The bundled profiles: `(language key, TOML source)`.
const BUILTIN_SOURCES: &[(&str, &str)] = &[
    ("python", PYTHON_TOML),
    ("rust", RUST_TOML),
    ("go", GO_TOML),
    ("typescript", TYPESCRIPT_TOML),
    ("lua", LUA_TOML),
    ("bash", BASH_TOML),
    ("c", C_TOML),
    ("nix", NIX_TOML),
];

/// Load the eight bundled profiles from the embedded TOML.
pub fn builtin_profiles() -> Result<BTreeMap<String, LanguageProfile>, CodegenError> {
    let mut profiles = BTreeMap::new();
    for (language, source) in BUILTIN_SOURCES {
        let document: toml_edit::DocumentMut =
            source.parse().map_err(|error: toml_edit::TomlError| {
                CodegenError::EmbeddedProfile {
                    language,
                    message: error.to_string(),
                }
            })?;
        profiles.insert(
            (*language).to_string(),
            profile_from_document(language, &document),
        );
    }
    Ok(profiles)
}

/// Load bundled profiles, then apply operator `*.toml` overrides from
/// `skills_dir`. A `None` directory, or one that is not a directory, yields
/// the bundled profiles unchanged.
///
/// Each file either overrides fields of an existing language (only keys it
/// defines) or introduces a new language keyed by file stem. Malformed or
/// oversized files are skipped, as the Python implementation skipped files
/// `tomllib` could not parse.
pub fn load_profiles(
    skills_dir: Option<&Path>,
) -> Result<BTreeMap<String, LanguageProfile>, CodegenError> {
    let mut profiles = builtin_profiles()?;
    let Some(dir) = skills_dir else {
        return Ok(profiles);
    };
    if !dir.is_dir() {
        return Ok(profiles);
    }
    // Collect lazily and stop at the cap: a hostile directory must not make
    // this allocate unboundedly.
    let mut entries = Vec::new();
    for entry in
        std::fs::read_dir(dir).map_err(|error| CodegenError::UnreadableDir(error.to_string()))?
    {
        let entry = entry.map_err(|error| CodegenError::UnreadableDir(error.to_string()))?;
        entries.push(entry);
        if entries.len() > PROFILE_FILES_MAX {
            return Err(CodegenError::TooManyProfileFiles {
                max: PROFILE_FILES_MAX,
            });
        }
    }
    // Deterministic merge order; the OS readdir order is arbitrary.
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.len() > PROFILE_TOML_BYTES_MAX {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(document) = text.parse::<toml_edit::DocumentMut>() else {
            continue;
        };
        match profiles.get_mut(stem) {
            Some(profile) => apply_overrides(profile, &document),
            None => {
                profiles.insert(stem.to_string(), new_profile(stem, &document));
            }
        }
    }
    Ok(profiles)
}

/// Format a profile for inclusion in LLM context, byte-identical to the
/// Python `profile_summary`.
pub fn profile_summary(profile: &LanguageProfile) -> String {
    // concat! keeps the golden format string byte-identical to Python while
    // staying within the line budget.
    let mut out = format!(
        concat!(
            "Language: {}\n",
            "LSP: {}\n",
            "Linter: {}\n",
            "Formatter: {}\n",
            "Test Runner: {}\n",
            "Build Tool: {}\n",
            "Package Manager: {}\n",
            "Available Frameworks:\n",
        ),
        profile.name,
        profile.lsp,
        profile.linter,
        profile.formatter,
        profile.test_runner,
        profile.build_tool,
        profile.package_manager,
    );
    // Python joins the framework lines with "\n" (no trailing newline) and
    // the template adds exactly one "\n" after; an empty framework list
    // still leaves the blank line after "Available Frameworks:".
    let lines: Vec<String> = profile
        .frameworks
        .iter()
        .map(|(name, description)| format!("  - {name}: {description}"))
        .collect();
    out.push_str(&lines.join("\n"));
    out.push('\n');
    out
}

/// Build a profile from a parsed TOML document. Missing scalar fields
/// default to `""`, matching the Python dataclass defaults.
fn profile_from_document(language: &str, document: &toml_edit::DocumentMut) -> LanguageProfile {
    LanguageProfile {
        name: string_field(document, "name").unwrap_or_else(|| titlecase(language)),
        extensions: string_array(document, "extensions")
            .unwrap_or_else(|| vec![format!(".{language}")]),
        lsp: string_field(document, "lsp").unwrap_or_default(),
        linter: string_field(document, "linter").unwrap_or_default(),
        formatter: string_field(document, "formatter").unwrap_or_default(),
        test_runner: string_field(document, "test_runner").unwrap_or_default(),
        build_tool: string_field(document, "build_tool").unwrap_or_default(),
        package_manager: string_field(document, "package_manager").unwrap_or_default(),
        frameworks: string_table(document, "frameworks"),
        install_hints: string_table(document, "install_hints"),
        project_templates: string_table(document, "project_templates"),
    }
}

/// A brand-new language from an operator file: same defaults as
/// `profile_from_document`.
fn new_profile(language: &str, document: &toml_edit::DocumentMut) -> LanguageProfile {
    profile_from_document(language, document)
}

/// Override only the fields the operator file defines, mirroring Python's
/// `setattr` loop over the TOML keys.
fn apply_overrides(profile: &mut LanguageProfile, document: &toml_edit::DocumentMut) {
    if let Some(name) = string_field(document, "name") {
        profile.name = name;
    }
    if let Some(extensions) = string_array(document, "extensions") {
        profile.extensions = extensions;
    }
    for (field, slot) in [
        ("lsp", &mut profile.lsp),
        ("linter", &mut profile.linter),
        ("formatter", &mut profile.formatter),
        ("test_runner", &mut profile.test_runner),
        ("build_tool", &mut profile.build_tool),
        ("package_manager", &mut profile.package_manager),
    ] {
        if let Some(value) = string_field(document, field) {
            *slot = value;
        }
    }
    for (field, slot) in [
        ("frameworks", &mut profile.frameworks),
        ("install_hints", &mut profile.install_hints),
        ("project_templates", &mut profile.project_templates),
    ] {
        if document
            .get(field)
            .and_then(|item| item.as_table())
            .is_some()
        {
            *slot = string_table(document, field);
        }
    }
}

/// A string field, or `None` when absent or not a string.
fn string_field(document: &toml_edit::DocumentMut, key: &str) -> Option<String> {
    document
        .get(key)
        .and_then(|item| item.as_str())
        .map(str::to_string)
}

/// An array of strings, or `None` when absent or not all strings.
fn string_array(document: &toml_edit::DocumentMut, key: &str) -> Option<Vec<String>> {
    let array = document.get(key)?.as_array()?;
    array
        .iter()
        .map(|item| item.as_str().map(str::to_string))
        .collect()
}

/// A string-to-string table in document order; non-string values are
/// skipped, absent tables yield an empty vec.
fn string_table(document: &toml_edit::DocumentMut, key: &str) -> Vec<(String, String)> {
    let Some(table) = document.get(key).and_then(|item| item.as_table()) else {
        return Vec::new();
    };
    table
        .iter()
        .filter_map(|(name, item)| {
            item.as_str()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect()
}

/// Python's `str.title()` for language keys: uppercase the first
/// alphanumeric after a non-alphanumeric boundary. ASCII-only; language
/// keys are ASCII in practice.
fn titlecase(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut boundary = true;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(if boundary {
                ch.to_ascii_uppercase()
            } else {
                ch
            });
            boundary = false;
        } else {
            out.push(ch);
            boundary = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// The Python `load_profiles()["python"]` summary, captured by driving
    /// the real Python (fixtures.json). Byte-exact, including the file-order
    /// framework list. Note: the shipped TOML files fully override the
    /// `BUILTIN_PROFILES` dataclasses, so this is the effective summary the
    /// Python runtime actually used.
    const PYTHON_SUMMARY: &str = "Language: Python\nLSP: pylsp / pyright\nLinter: ruff\nFormatter: ruff format\nTest Runner: pytest\nBuild Tool: hatch\nPackage Manager: uv\nAvailable Frameworks:\n  - fastapi: Async REST API with OpenAPI docs \u{2014} preferred for APIs\n  - flask: Lightweight WSGI web framework\n  - django: Full-stack web framework\n  - click: CLI framework \u{2014} simple\n  - typer: FastAPI-style CLI framework \u{2014} preferred for CLIs\n  - langchain: LLM pipeline/agent framework\n  - streamlit: Rapid data/ML dashboard\n  - bare: Pure Python script or module\n";

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-codegen-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir
    }

    #[test]
    fn eight_bundled_profiles_load() {
        let profiles = builtin_profiles().unwrap();
        let mut keys: Vec<&str> = profiles.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "bash",
                "c",
                "go",
                "lua",
                "nix",
                "python",
                "rust",
                "typescript"
            ]
        );
    }

    #[test]
    fn python_profile_linter_is_ruff() {
        // Mirrors tests/test_package_backend.py: the Python profile stays
        // available outside the repo and its linter is ruff.
        let profiles = builtin_profiles().unwrap();
        let python = &profiles["python"];
        assert_eq!(python.linter, "ruff");
        assert_eq!(python.extensions, vec![".py"]);
        assert!(!python.frameworks.is_empty());
    }

    #[test]
    fn profile_summary_matches_python_byte_for_byte() {
        let profiles = builtin_profiles().unwrap();
        assert_eq!(profile_summary(&profiles["python"]), PYTHON_SUMMARY);
    }

    #[test]
    fn all_profile_summaries_match_python_byte_for_byte() {
        // Differential against the real Python: profile_summary() over
        // load_profiles() for every bundled language, captured by driving
        // flow/codegen/language_profiles.py on 2026-09-26.
        let expected: &[(&str, &str)] = &[
            (
                "bash",
                concat!(
                    "Language: Bash\n",
                    "LSP: bash-language-server\n",
                    "Linter: shellcheck\n",
                    "Formatter: shfmt\n",
                    "Test Runner: bats\n",
                    "Build Tool: make\n",
                    "Package Manager: N/A\n",
                    "Available Frameworks:\n",
                    "  - systemd-service: systemd service wrapper script\n",
                    "  - cli-tool: Command-line tool with argument parsing\n",
                    "  - deployment: Deployment/provisioning script\n",
                    "  - bare: General purpose shell script\n",
                ),
            ),
            (
                "c",
                concat!(
                    "Language: C\n",
                    "LSP: clangd\n",
                    "Linter: clang-tidy\n",
                    "Formatter: clang-format\n",
                    "Test Runner: cmocka / unity\n",
                    "Build Tool: cmake / meson\n",
                    "Package Manager: pacman\n",
                    "Available Frameworks:\n",
                    "  - cmake-lib: CMake library project\n",
                    "  - cmake-bin: CMake binary project\n",
                    "  - meson: Meson build system project\n",
                    "  - bare: Single-file C program\n",
                ),
            ),
            (
                "go",
                concat!(
                    "Language: Go\n",
                    "LSP: gopls\n",
                    "Linter: golangci-lint\n",
                    "Formatter: gofmt / goimports\n",
                    "Test Runner: go test\n",
                    "Build Tool: go build\n",
                    "Package Manager: go mod\n",
                    "Available Frameworks:\n",
                    "  - gin: Fast HTTP web framework\n",
                    "  - echo: High performance web framework\n",
                    "  - fiber: Express-inspired web framework\n",
                    "  - cobra: CLI framework (used by kubectl, git) — preferred for CLIs\n",
                    "  - bare: Pure Go — no framework\n",
                ),
            ),
            (
                "lua",
                concat!(
                    "Language: Lua\n",
                    "LSP: lua-language-server\n",
                    "Linter: luacheck\n",
                    "Formatter: stylua\n",
                    "Test Runner: busted\n",
                    "Build Tool: luarocks\n",
                    "Package Manager: luarocks\n",
                    "Available Frameworks:\n",
                    "  - neovim-plugin: Neovim plugin (Lua API) — preferred for nvim work\n",
                    "  - love2d: 2D game framework\n",
                    "  - openresty: Nginx/Lua web server\n",
                    "  - bare: Pure Lua script\n",
                ),
            ),
            (
                "nix",
                concat!(
                    "Language: Nix\n",
                    "LSP: nil / nixd\n",
                    "Linter: statix\n",
                    "Formatter: alejandra / nixfmt\n",
                    "Test Runner: nix flake check\n",
                    "Build Tool: nix build\n",
                    "Package Manager: nix\n",
                    "Available Frameworks:\n",
                    "  - flake: Nix flake with outputs\n",
                    "  - home-manager: Home Manager module\n",
                    "  - nixos-module: NixOS module\n",
                    "  - bare: Simple nix expression\n",
                ),
            ),
            (
                "python",
                concat!(
                    "Language: Python\n",
                    "LSP: pylsp / pyright\n",
                    "Linter: ruff\n",
                    "Formatter: ruff format\n",
                    "Test Runner: pytest\n",
                    "Build Tool: hatch\n",
                    "Package Manager: uv\n",
                    "Available Frameworks:\n",
                    "  - fastapi: Async REST API with OpenAPI docs — preferred for APIs\n",
                    "  - flask: Lightweight WSGI web framework\n",
                    "  - django: Full-stack web framework\n",
                    "  - click: CLI framework — simple\n",
                    "  - typer: FastAPI-style CLI framework — preferred for CLIs\n",
                    "  - langchain: LLM pipeline/agent framework\n",
                    "  - streamlit: Rapid data/ML dashboard\n",
                    "  - bare: Pure Python script or module\n",
                ),
            ),
            (
                "rust",
                concat!(
                    "Language: Rust\n",
                    "LSP: rust-analyzer\n",
                    "Linter: clippy\n",
                    "Formatter: rustfmt\n",
                    "Test Runner: cargo test\n",
                    "Build Tool: cargo\n",
                    "Package Manager: cargo\n",
                    "Available Frameworks:\n",
                    "  - axum: Ergonomic async web (tokio-based) — preferred for APIs\n",
                    "  - actix-web: High-perf actor model web framework\n",
                    "  - clap: CLI argument parser — preferred for CLIs\n",
                    "  - tokio: Async runtime for custom apps\n",
                    "  - tonic: gRPC framework\n",
                    "  - egui: Immediate mode GUI\n",
                    "  - bevy: ECS game engine\n",
                    "  - bare: Pure Rust binary or library\n",
                ),
            ),
            (
                "typescript",
                concat!(
                    "Language: TypeScript\n",
                    "LSP: typescript-language-server\n",
                    "Linter: eslint\n",
                    "Formatter: prettier\n",
                    "Test Runner: vitest\n",
                    "Build Tool: vite / tsc / esbuild\n",
                    "Package Manager: pnpm / npm / bun\n",
                    "Available Frameworks:\n",
                    "  - nextjs: React full-stack framework with SSR/SSG\n",
                    "  - react: UI component library (Vite)\n",
                    "  - express: Node.js REST API framework\n",
                    "  - fastify: High-performance Node.js framework\n",
                    "  - hono: Ultrafast edge/node web framework — preferred for APIs\n",
                    "  - nestjs: Angular-inspired backend framework\n",
                    "  - bare: Pure TypeScript — node script or library\n",
                ),
            ),
        ];
        let profiles = builtin_profiles().unwrap();
        for (language, summary) in expected {
            assert_eq!(
                &profile_summary(&profiles[*language]),
                summary,
                "profile_summary diverged for {language}"
            );
        }
    }

    #[test]
    fn project_templates_match_python_effective_values() {
        // The shipped TOMLs previously dropped the `project_templates`
        // the Python dataclasses carry. Captured by driving
        // flow/codegen/language_profiles.py::load_profiles() on
        // 2026-09-26; order is the Python dict order.
        let expected: &[(&str, &[(&str, &str)])] = &[
            (
                "bash",
                &[(
                    "bare",
                    "mkdir {name} && touch {name}/main.sh && chmod +x {name}/main.sh",
                )],
            ),
            (
                "go",
                &[
                    (
                        "gin",
                        "mkdir {name} && cd {name} && go mod init {name} && go get github.com/gin-gonic/gin",
                    ),
                    (
                        "cobra",
                        "mkdir {name} && cd {name} && go mod init {name} && go install github.com/spf13/cobra-cli@latest && cobra-cli init",
                    ),
                    ("bare", "mkdir {name} && cd {name} && go mod init {name}"),
                ],
            ),
            (
                "lua",
                &[
                    (
                        "neovim-plugin",
                        "mkdir -p {name}/lua/{name} && touch {name}/lua/{name}/init.lua",
                    ),
                    ("bare", "mkdir {name} && touch {name}/main.lua"),
                ],
            ),
            (
                "python",
                &[
                    (
                        "fastapi",
                        "uv init {name} && cd {name} && uv add fastapi uvicorn",
                    ),
                    ("flask", "uv init {name} && cd {name} && uv add flask"),
                    ("click", "uv init {name} && cd {name} && uv add click"),
                    ("bare", "mkdir -p {name} && cd {name} && uv init"),
                ],
            ),
            (
                "rust",
                &[
                    (
                        "axum",
                        "cargo new {name} && cd {name} && cargo add axum tokio --features tokio/full",
                    ),
                    (
                        "clap",
                        "cargo new {name} && cd {name} && cargo add clap --features derive",
                    ),
                    ("bare", "cargo new {name}"),
                ],
            ),
            (
                "typescript",
                &[
                    ("nextjs", "npx create-next-app@latest {name} --typescript"),
                    (
                        "react",
                        "npm create vite@latest {name} -- --template react-ts",
                    ),
                    (
                        "express",
                        "mkdir {name} && cd {name} && npm init -y && npm i express typescript @types/express ts-node",
                    ),
                    (
                        "bare",
                        "mkdir {name} && cd {name} && npm init -y && npm i typescript && npx tsc --init",
                    ),
                ],
            ),
            (
                "c",
                &[
                    (
                        "cmake-bin",
                        "mkdir {name} && cd {name} && cmake -DCMAKE_BUILD_TYPE=Debug ..",
                    ),
                    ("bare", "touch {name}.c"),
                ],
            ),
            (
                "nix",
                &[
                    ("flake", "nix flake init"),
                    ("bare", "echo '{}' > default.nix"),
                ],
            ),
        ];
        let profiles = builtin_profiles().unwrap();
        for (language, templates) in expected {
            let actual: Vec<(&str, &str)> = profiles[*language]
                .project_templates
                .iter()
                .map(|(name, description)| (name.as_str(), description.as_str()))
                .collect();
            let expected: Vec<(&str, &str)> = templates.to_vec();
            assert_eq!(
                actual, expected,
                "project_templates diverged for {language}"
            );
        }
    }

    #[test]
    fn frameworks_keep_document_order() {
        let profiles = builtin_profiles().unwrap();
        let names: Vec<&str> = profiles["python"]
            .frameworks
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "fastapi",
                "flask",
                "django",
                "click",
                "typer",
                "langchain",
                "streamlit",
                "bare"
            ]
        );
    }

    #[test]
    fn load_profiles_without_dir_returns_builtins() {
        let profiles = load_profiles(None).unwrap();
        assert_eq!(profiles.len(), 8);
        let missing = temp_dir("missing-child");
        let _ = std::fs::remove_dir_all(&missing);
        let profiles = load_profiles(Some(&missing)).unwrap();
        assert_eq!(profiles.len(), 8);
    }

    #[test]
    fn operator_file_overrides_bundled_fields() {
        let dir = temp_dir("override");
        std::fs::write(dir.join("python.toml"), "linter = \"custom-lint\"\n").unwrap();
        let profiles = load_profiles(Some(&dir)).unwrap();
        assert_eq!(profiles["python"].linter, "custom-lint");
        // Untouched fields keep bundled values.
        assert_eq!(profiles["python"].formatter, "ruff format");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn operator_file_adds_new_language() {
        let dir = temp_dir("newlang");
        std::fs::write(dir.join("zig.toml"), "linter = \"ziglint\"\n[frameworks]\n").unwrap();
        let profiles = load_profiles(Some(&dir)).unwrap();
        let zig = &profiles["zig"];
        assert_eq!(zig.name, "Zig");
        assert_eq!(zig.extensions, vec![".zig"]);
        assert_eq!(zig.linter, "ziglint");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_operator_file_is_skipped() {
        let dir = temp_dir("malformed");
        std::fs::write(dir.join("python.toml"), "linter = \n").unwrap();
        let profiles = load_profiles(Some(&dir)).unwrap();
        assert_eq!(profiles["python"].linter, "ruff");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profile_cap_is_an_error_not_truncation() {
        let dir = temp_dir("cap");
        for i in 0..=PROFILE_FILES_MAX {
            std::fs::write(dir.join(format!("lang{i:03}.toml")), "linter = \"x\"\n").unwrap();
        }
        let error = load_profiles(Some(&dir)).unwrap_err();
        assert!(matches!(
            error,
            CodegenError::TooManyProfileFiles {
                max: PROFILE_FILES_MAX
            }
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profile_cap_boundary_loads() {
        let dir = temp_dir("cap-ok");
        for i in 0..PROFILE_FILES_MAX {
            std::fs::write(dir.join(format!("lang{i:03}.toml")), "linter = \"x\"\n").unwrap();
        }
        let profiles = load_profiles(Some(&dir)).unwrap();
        assert!(profiles.len() >= PROFILE_FILES_MAX);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn titlecase_matches_python_str_title() {
        assert_eq!(titlecase("rust"), "Rust");
        assert_eq!(titlecase("my-lang"), "My-Lang");
    }
}
