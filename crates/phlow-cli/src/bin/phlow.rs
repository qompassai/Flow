//! `phlow` binary: thin entrypoint over the shared CLI implementation in
//! [`phlow_cli`]. The `flow` compatibility alias is `src/bin/flow.rs`.

#![forbid(unsafe_code)]

fn main() {
    std::process::exit(phlow_cli::run());
}
