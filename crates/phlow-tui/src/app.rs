//! The interactive chat loop, ported from `FlowApp.run` in
//! `flow/tui/app.py`.
//!
//! The loop is deliberately line-based over stdin/stdout with inline
//! ratatui panels: the Python TUI used prompt_toolkit/rich inline printing,
//! not a full-screen alternate buffer, and this port keeps that behavior.
//! Ctrl-C terminates the process (there is no signal handler; Python's loop
//! caught it and continued); `/quit`, `/exit`, or Ctrl-D ends the session
//! cleanly.

use std::io::{self, BufRead, Read, Write};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph};
use serde_json::Value;

use crate::commands::FlowApp;
use crate::facade::RuntimeFacade;
use crate::panels::styled_panel;

/// What the loop should do with one input line.
#[derive(Debug, PartialEq, Eq)]
pub enum LineAction {
    /// End the session (`/quit` or `/exit`, exactly, after trimming).
    Quit,
    /// Ignore the line (empty after trimming).
    Skip,
    /// Execute it as a command or task; carries the trimmed text.
    Execute(String),
}

/// Classify one input line, mirroring the top of Python's `run()` loop:
/// trim, then check for quit/exit, then skip empties. Pure and
/// unit-testable; [`run`] acts on the outcome.
pub fn classify_line(line: &str) -> LineAction {
    let text = line.trim();
    if text.is_empty() {
        LineAction::Skip
    } else if text == "/quit" || text == "/exit" {
        LineAction::Quit
    } else {
        LineAction::Execute(text.to_string())
    }
}

/// Run the chat loop without terminal widgets, for when stdout is not a
/// TTY (pipes, CI, `phlow < /dev/null`).
///
/// The ratatui banner needs a real terminal; the line loop does not, so
/// this prints a plain-text banner, runs the same [`read_line`] /
/// [`classify_line`] loop as [`run`], and renders each result as compact
/// JSON on its own line. Ctrl-D (EOF) exits cleanly, mirroring Python's
/// `except EOFError: break`. (Deliberate deviation, shared with [`run`]:
/// Ctrl-C terminates the process instead of continuing the loop.)
pub fn run_line_mode<F: RuntimeFacade>(app: &mut FlowApp<F>) -> io::Result<()> {
    println!("Flow · local planner / coder / reviewer");
    println!("Workspace: {}", app.workspace_root());
    println!("Type /help for commands. /quit or Ctrl-D to exit.");
    if !app.trusted() {
        println!("Read-only mode: checks are disabled unless trusted (--trusted)");
    }

    while let Some(line) = read_line("flow > ")? {
        match classify_line(&line) {
            LineAction::Quit => break,
            LineAction::Skip => continue,
            LineAction::Execute(text) => {
                let result = app.execute(&text);
                println!("{result}");
            }
        }
    }
    Ok(())
}

/// Run the chat loop until `/quit`, `/exit`, Ctrl-D, or an I/O error.
pub fn run<F: RuntimeFacade>(app: &mut FlowApp<F>) -> io::Result<()> {
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    draw_banner(&mut terminal, app.workspace_root())?;
    if !app.trusted() {
        println!("Read-only mode: checks are disabled unless trusted (--trusted)");
    }

    while let Some(line) = read_line("flow > ")? {
        match classify_line(&line) {
            LineAction::Quit => break,
            LineAction::Skip => continue,
            LineAction::Execute(text) => {
                let result = app.execute(&text);
                draw_result(&mut terminal, &result)?;
            }
        }
    }
    Ok(())
}

/// The startup banner: `Flow · local planner / coder / reviewer` with the
/// workspace root, as in `flow/tui.py`.
fn draw_banner(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    workspace_root: &str,
) -> io::Result<()> {
    let green_bold = Style::default()
        .fg(Color::Green)
        .add_modifier(Modifier::BOLD);
    terminal.draw(|frame| {
        let mut banner_text = String::from("Flow · local planner / coder / reviewer\n");
        banner_text.push_str(&format!("Workspace: {workspace_root}\n"));
        banner_text.push_str("Type /help for commands. /quit or Ctrl-D to exit.");
        let banner = Paragraph::new(banner_text).block(
            Block::bordered()
                .title(Line::from("Flow").style(green_bold))
                .border_style(green_bold),
        );
        frame.render_widget(banner, frame.area());
    })?;
    Ok(())
}

/// Render one command result as a cyan `Flow result` panel.
fn draw_result(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    value: &Value,
) -> io::Result<()> {
    let pretty = serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string());
    terminal.draw(|frame| {
        let panel = styled_panel(&pretty, "Flow result", Color::Cyan);
        frame.render_widget(panel, frame.area());
    })?;
    Ok(())
}

/// An input line longer than this many bytes is rejected: stdin is
/// operator-typed, but the TUI must not buffer an unbounded line.
pub const INPUT_LINE_BYTES_MAX: usize = 1_000_000;

/// Read one input line after printing the `flow > ` prompt. Returns `None`
/// on EOF (Ctrl-D). The bound counts every byte read, including the
/// line's single trailing newline: a read longer than
/// [`INPUT_LINE_BYTES_MAX`] bytes fails instead of growing the buffer
/// without bound. The returned line has that one trailing newline
/// removed, like Python's `input()`.
pub fn read_line(prompt: &str) -> io::Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    // Read one byte past the cap so an overlong line is detectable.
    let mut limited = io::stdin().take((INPUT_LINE_BYTES_MAX + 1) as u64);
    let mut line = String::new();
    let bytes = io::BufReader::new(&mut limited).read_line(&mut line)?;
    if bytes == 0 {
        return Ok(None);
    }
    check_input_bound(bytes)?;
    Ok(Some(strip_trailing_newline(line)))
}

/// Remove the single trailing newline after the bound check: like
/// Python's `input()`, the caller sees the line without it, while
/// [`check_input_bound`] still counts every byte read (a 1,000,000-byte
/// line plus its newline is 1,000,001 bytes and is rejected before the
/// strip). Only one `\n` goes; a final line with no newline is
/// untouched.
fn strip_trailing_newline(mut line: String) -> String {
    if line.ends_with('\n') {
        line.pop();
    }
    line
}

/// Reject a `read_line` result whose total byte count exceeds
/// [`INPUT_LINE_BYTES_MAX`]. The trailing newline counts toward the
/// bound: a 1,000,000-byte line plus its newline is 1,000,001 bytes and
/// is rejected. Kept separate so the exact boundary is unit-testable
/// without stdin.
fn check_input_bound(bytes_read: usize) -> io::Result<()> {
    if bytes_read > INPUT_LINE_BYTES_MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input line exceeds INPUT_LINE_BYTES_MAX bytes",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_bound_counts_the_trailing_newline() {
        // Exactly at the bound: accepted. One byte more — e.g. a
        // 1,000,000-byte line plus its newline — is rejected.
        assert!(check_input_bound(INPUT_LINE_BYTES_MAX).is_ok());
        assert!(check_input_bound(INPUT_LINE_BYTES_MAX - 1).is_ok());
        assert!(check_input_bound(INPUT_LINE_BYTES_MAX + 1).is_err());
    }

    #[test]
    fn strip_trailing_newline_removes_exactly_one() {
        assert_eq!(strip_trailing_newline("abc\n".to_string()), "abc");
        assert_eq!(strip_trailing_newline("abc".to_string()), "abc");
        assert_eq!(strip_trailing_newline("\n".to_string()), "");
        assert_eq!(strip_trailing_newline("".to_string()), "");
        // Two newlines: only the last one goes.
        assert_eq!(strip_trailing_newline("abc\n\n".to_string()), "abc\n");
    }

    #[test]
    fn classify_line_handles_quit_variants() {
        assert_eq!(classify_line("/quit"), LineAction::Quit);
        assert_eq!(classify_line("/exit"), LineAction::Quit);
        assert_eq!(classify_line("  /quit  "), LineAction::Quit);
    }

    #[test]
    fn classify_line_skips_empty_lines() {
        assert_eq!(classify_line(""), LineAction::Skip);
        assert_eq!(classify_line("   "), LineAction::Skip);
    }

    #[test]
    fn classify_line_trims_executable_lines() {
        assert_eq!(
            classify_line("/status"),
            LineAction::Execute("/status".to_string())
        );
        assert_eq!(
            classify_line("  hello world  "),
            LineAction::Execute("hello world".to_string())
        );
    }

    #[test]
    fn classify_line_does_not_quit_on_embedded_quit() {
        assert_eq!(
            classify_line("please /quit now"),
            LineAction::Execute("please /quit now".to_string())
        );
        assert_eq!(
            classify_line("/quitter"),
            LineAction::Execute("/quitter".to_string())
        );
    }
}
