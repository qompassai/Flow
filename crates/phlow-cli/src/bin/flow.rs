//! `flow` binary: the legacy name, kept as a compatibility alias. Thin
//! entrypoint over the shared CLI implementation in [`phlow_cli`]; the
//! primary binary is `src/bin/phlow.rs`.

#![forbid(unsafe_code)]

fn main() {
    std::process::exit(phlow_cli::run());
}
