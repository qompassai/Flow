//! SIGTERM handling, mirroring `flow/main.py`.
//!
//! Python installs a `SIGTERM` handler that raises `KeyboardInterrupt`,
//! unwinding the command (including the `serve` loop) so `main` prints
//! `Flow interrupted; changes already written are not rolled back.` and
//! exits 130. This module reproduces the observable contract — the message
//! on stderr and exit code 130 — with a dedicated watcher thread: the
//! thread owns signal delivery, the main thread owns the runtime, and the
//! process exits promptly on delivery. Cleanup beyond process exit is moot
//! (the OS reclaims every handle), which is exactly what Python's message
//! documents: already-written changes are *not* rolled back.
//!
//! The watcher is unix-only, mirroring Python's
//! `if hasattr(signal, "SIGTERM")` guard.

/// Exit code for keyboard/signal interruption, matching Python's
/// `except KeyboardInterrupt: return 130`.
pub const EXIT_INTERRUPTED: i32 = 130;

/// Usage or configuration error, matching Python's `return 2`.
pub const EXIT_USAGE: i32 = 2;

/// The command ran but the report status was not `ok`, matching Python's
/// `return 0 if result.get("status") == "ok" else 1`.
pub const EXIT_COMMAND_FAILED: i32 = 1;

/// The stderr line Python prints on interruption, byte for byte.
pub const INTERRUPTED_MESSAGE: &str =
    "Flow interrupted; changes already written are not rolled back.";

/// Install the SIGTERM watcher. On unix this spawns the watcher thread;
/// elsewhere it is a no-op. Failure to install (a broken self-pipe) is a
/// startup error: without the watcher, SIGTERM would kill with the default
/// disposition (143), breaking the exit-code contract.
pub fn install_sigterm_watcher() -> Result<(), String> {
    #[cfg(unix)]
    {
        install_unix()
    }
    #[cfg(not(unix))]
    {
        Ok(())
    }
}

#[cfg(unix)]
fn install_unix() -> Result<(), String> {
    use std::io::Write as _;

    use signal_hook::consts::SIGTERM;
    use signal_hook::iterator::Signals;

    let mut signals =
        Signals::new([SIGTERM]).map_err(|error| format!("cannot watch for SIGTERM: {error}"))?;
    // `signals` is moved into the thread: it owns the self-pipe for the
    // life of the process, exactly one owner, no shared state.
    std::thread::Builder::new()
        .name("phlow-sigterm".to_owned())
        .spawn(move || {
            // `forever()` blocks until delivery; the first SIGTERM ends the
            // process with Python's message and exit code 130, mirroring
            // the SIGTERM->KeyboardInterrupt handler in flow/main.py.
            let _delivered = signals.forever().next();
            eprintln!("{INTERRUPTED_MESSAGE}");
            let _ = std::io::stderr().flush();
            std::process::exit(EXIT_INTERRUPTED);
        })
        .map(|_| ())
        .map_err(|error| format!("cannot spawn SIGTERM watcher thread: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_python() {
        assert_eq!(EXIT_INTERRUPTED, 130);
        assert_eq!(EXIT_USAGE, 2);
        assert_eq!(EXIT_COMMAND_FAILED, 1);
    }

    #[test]
    fn interrupted_message_matches_python_byte_for_byte() {
        assert_eq!(
            INTERRUPTED_MESSAGE,
            "Flow interrupted; changes already written are not rolled back."
        );
    }

    #[test]
    fn watcher_installs_without_error() {
        install_sigterm_watcher().expect("watcher must install");
    }
}
