# Packaging phlow

How to install and run the Rust port's binaries (`phlow`, plus the `flow`
compatibility alias) on each platform. Every section states what was
**validated in this environment** versus what is **target-only**
(instructions for the target machine, not run here).

The binaries are built with `cargo build --release` from the repo root on
the `rust` branch. No installer downloads anything at runtime; the only
runtime dependency is an Ollama server (default `http://127.0.0.1:11434`).

## Linux: systemd

**Validated here:** `systemd-analyze verify` passes on all four units; the
only notes are the expected "binary not installed at this path" for
`ExecStart` (the units reference install locations, and this environment
is not the install target).

**Target-only:** installing the binary, creating the system user, and
enabling the units.

### stdio versus socket — read this first

`phlow serve` speaks **newline-delimited JSON-RPC on stdin/stdout** and
exits on EOF. It is not a TCP daemon and must not be run as a plain
`Type=simple` service with pipes to nowhere. The units below use **socket
activation** instead: each accepted connection spawns one `phlow serve`
instance whose stdin/stdout *are* the connection. One connection, one
process, no shared state between sessions. A clean EOF shutdown (exit 0)
is the normal case and is not restarted; `Restart=on-failure` only fires
on abnormal exits.

### User instance

Files: `packaging/phlow.socket`, `packaging/phlow@.service`.

```sh
# target-only: install and enable
cargo build --release
install -m755 target/release/phlow ~/.local/bin/phlow
ln -s target/release/flow ~/.local/bin/flow   # or: install -m755 target/release/flow ~/.local/bin/flow
mkdir -p ~/.config/systemd/user
cp packaging/phlow.socket 'packaging/phlow@.service' ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now phlow.socket
# point --workspace at your projects dir by editing the unit's ExecStart
```

The socket is `$XDG_RUNTIME_DIR/phlow.sock`. A local client (e.g.
rose.nvim's MCP stdio client) dials it; systemd spawns `phlow serve` per
connection.

### System instance

Files: `packaging/phlow-system.socket`, `packaging/phlow-system@.service`.

```sh
# target-only: everything below needs root on the target machine
cargo build --release
sudo install -m755 target/release/phlow /usr/local/bin/phlow
sudo install -m755 target/release/flow /usr/local/bin/flow
sudo useradd --system --home-dir /var/lib/phlow --create-home \
     --shell /usr/sbin/nologin phlow
sudo cp packaging/phlow-system.socket 'packaging/phlow-system@.service' \
     /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now phlow-system.socket
```

The socket is `/run/phlow.sock`, owned `root:phlow` mode `0770`, so
members of the `phlow` group can connect without becoming root.

### Hardening

Both services set `NoNewPrivileges=yes`, `ProtectSystem=strict`,
`PrivateTmp=yes`, and `RestrictAddressFamilies=AF_UNIX AF_INET`
(AF_UNIX for the Neovim socket, AF_INET for loopback Ollama only).
`--trusted` is off: the system instance never enables workspace writes
or named checks by itself.

### Relation to `systemd/flow.service`

The old `systemd/flow.service` (a `flow status` oneshot against the
Python tree) is **superseded** by these units for the Rust port. It is
left byte-identical in place — the Python tree is not touched — but it
must not be enabled alongside the new units.

## Linux: desktop entry

**Validated here:** `phlow.desktop` is a static file in this directory;
the SVG icon is well-formed XML (parsed with Python's xml.dom.minidom —
`desktop-file-validate` is not installed here, so that check is
target-only).

**Target-only:** install and validate on the target desktop:

```sh
# target-only
sudo install -m644 packaging/phlow.desktop /usr/share/applications/
sudo install -m644 packaging/icons/phlow.svg /usr/share/icons/hicolor/scalable/apps/
sudo gtk-update-icon-cache /usr/share/icons/hicolor/   # if applicable
desktop-file-validate /usr/share/applications/phlow.desktop
```

`Terminal=true` is load-bearing: the TUI needs a real terminal. If
stdout is not a TTY, `phlow tui` runs the headless line loop instead of
the ratatui frontend.

## Windows

**Chosen route: winget.** Rationale: phlow is a portable CLI/TUI with no
system services on Windows; winget distributes exactly that (a versioned
binary on PATH) with the least moving parts. A WiX MSI would be the next
step only if an all-users machine-wide install with Start Menu shortcuts
is needed — it is documented below but not pursued first.

**Target-only** (every step needs a Windows host or CI runner):

1. `cargo build --release` on Windows (`x86_64-pc-windows-msvc`).
   The `flow` compatibility binary is the same `main.rs` built as the
   second `[[bin]]`; ship both, or ship `phlow.exe` and document `flow`
   as `phlow` invoked under that name.
2. Publish both `.exe` files to a GitHub release
   (`qompassai/phlow`, tag per release).
3. Submit a manifest to `microsoft/winget-pkgs`
   (`qompassai.phlow`, version, installer URL + SHA256, `winget install
   qompassai.phlow`).
4. Verify on a clean Windows VM: `phlow --version` prints `Flow 0.2.0`,
   `phlow status` prints one JSON line.

WiX route (if ever needed): `cargo wix` on the same release build,
producing a per-user MSI that installs both binaries and the
`packaging/phlow.desktop`-equivalent Start Menu shortcut. Not validated
here — no Windows host in this environment.

Note: the SIGTERM→130 contract is unix-only. On Windows, Ctrl-C /
console close terminates the process; the "changes already written are
not rolled back" semantics still hold (no rollback is ever attempted).

## Android: F-Droid and Google Play

**Status: documented route, not a built artifact.** phlow is a
terminal-first CLI/TUI; there is no Android app shell in this repo, and
this environment has no Android SDK/NDK, so no APK/AAB was built or
validated here.

Realistic route today: **Termux**. `cargo` runs on-device in Termux
(aarch64); `cargo build --release -p phlow-cli` produces working
`phlow`/`flow` binaries there, and the TUI's headless line loop
(`run_line_mode`) is the sane frontend until terminal-widget support on
Android is proven. An F-Droid listing would package this Termux build
path, not a native APK.

Google Play would additionally require an app wrapper (Activity hosting
a terminal emulator, e.g. termux-app derived) with the binaries as
bundled executables — a separate project, out of scope for the port.
This document records the route so the option stays visible; it does
not claim Play readiness.

## macOS

`cargo build --release` on the Mac produces both binaries; install to
`/usr/local/bin` or `~/.local/bin`. No launchd plist is shipped: the
socket-activation model is systemd-specific. For an always-on macOS
setup, run `phlow serve` under a supervisor that holds its stdio pipes
(the protocol is newline-delimited JSON-RPC; EOF ends the session).
Target-only — no macOS host in this environment.
