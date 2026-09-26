//! Ratatui panels, ported from the rich panels in `flow/tui.py`.
//!
//! Each panel is a pure widget constructor: no terminal access, no I/O.
//! The interactive loop in [`crate::app`] renders them; tests render them
//! into a [`TestBackend`](ratatui::backend::TestBackend) and assert on the
//! buffer text.
//!
//! # Truncation
//!
//! Panels that embed model-controlled text truncate it with named bounds:
//! [`TOOL_RESULT_CHARS_MAX_DEFAULT`], [`MEMORY_WHEN_CHARS_MAX`],
//! [`MEMORY_QUERY_CHARS_MAX`], [`MEMORY_RESPONSE_CHARS_MAX`],
//! [`TOOL_DESCRIPTION_CHARS_MAX`].
//!
//! [`tool_call_panel`] is the exception: it renders the whole validated
//! argument map, as the Python TUI did. Argument size is bounded upstream
//! at the validation layer (the runtime caps arguments at depth 16 and
//! 4,096 nodes), so the panel itself does not re-truncate.

use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Cell, Paragraph, Row, Table};

use serde_json::{Map, Value};

/// Default cap for a tool result's rendered characters.
pub const TOOL_RESULT_CHARS_MAX_DEFAULT: usize = 500;

/// `memory_table` truncates the timestamp to this many characters.
pub const MEMORY_WHEN_CHARS_MAX: usize = 16;

/// `memory_table` truncates the query to this many characters.
pub const MEMORY_QUERY_CHARS_MAX: usize = 50;

/// `memory_table` truncates the response to this many characters.
pub const MEMORY_RESPONSE_CHARS_MAX: usize = 60;

/// `tools_table` truncates each description to this many characters.
pub const TOOL_DESCRIPTION_CHARS_MAX: usize = 70;

/// The `Tool Call` panel: the tool name and its arguments, each rendered
/// Python-`repr` style (`flow/tui.py::tool_call_panel`).
pub fn tool_call_panel(tool_name: &str, args: &Map<String, Value>) -> Paragraph<'static> {
    let mut text = tool_name.to_string();
    for (key, value) in args {
        text.push('\n');
        text.push_str("  ");
        text.push_str(key);
        text.push_str(": ");
        text.push_str(&py_repr(value));
    }
    titled_panel(&text, "Tool Call", Color::Cyan)
}

/// The `Result: <tool>` panel: the tool result truncated to `max_chars`
/// characters with a `...` marker when truncated, exactly as in Python.
pub fn tool_result_panel(tool_name: &str, result: &str, max_chars: usize) -> Paragraph<'static> {
    let title = format!("Result: {tool_name}");
    titled_panel(&truncate_chars(result, max_chars), &title, Color::Cyan)
}

/// The red `Error` panel (`flow/tui.py::error_panel`).
pub fn error_panel(message: &str) -> Paragraph<'static> {
    styled_panel(message, "Error", Color::Red)
}

/// The green `Success` panel (`flow/tui.py::success_panel`).
pub fn success_panel(message: &str) -> Paragraph<'static> {
    styled_panel(message, "Success", Color::Green)
}

/// The yellow `Warning` panel (`flow/tui.py::warning_panel`).
pub fn warning_panel(message: &str) -> Paragraph<'static> {
    styled_panel(message, "Warning", Color::Yellow)
}

/// A panel with a colored title and matching border.
pub fn styled_panel(message: &str, title: &str, color: Color) -> Paragraph<'static> {
    titled_panel(message, title, color)
}

/// The code panel: title `<language> — <title>` when a title is given,
/// else `<language>` (`flow/tui.py::code_panel`).
pub fn code_panel(code: &str, language: &str, title: Option<&str>) -> Paragraph<'static> {
    let panel_title = match title {
        Some(title) if !title.is_empty() => format!("{language} — {title}"),
        _ => language.to_string(),
    };
    titled_panel(code, &panel_title, Color::Magenta)
}

/// The two-column status table: keys and `str(value)` cells, as in
/// `flow/tui.py::status_table`.
pub fn status_table(items: &Map<String, Value>) -> Table<'static> {
    let rows: Vec<Row<'static>> = items
        .iter()
        .map(|(key, value)| {
            Row::new(vec![
                Cell::from(key.clone()),
                Cell::from(json_scalar_text(value)),
            ])
        })
        .collect();
    Table::new(
        rows,
        [
            ratatui::layout::Constraint::Length(20),
            ratatui::layout::Constraint::Min(0),
        ],
    )
    .header(Row::new(vec![Cell::from("Key"), Cell::from("Value")]))
    .block(Block::bordered().title("Status"))
    .column_spacing(1)
}

/// One row of the memory table: when, query, response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRow {
    /// Timestamp; shown truncated to [`MEMORY_WHEN_CHARS_MAX`].
    pub when: String,
    /// The query; shown truncated to [`MEMORY_QUERY_CHARS_MAX`].
    pub query: String,
    /// The response; shown truncated to [`MEMORY_RESPONSE_CHARS_MAX`].
    pub response: String,
}

/// The memory table: `When | Query | Response` with bounded columns
/// (`flow/tui.py::memory_table`).
pub fn memory_table(memories: &[MemoryRow]) -> Table<'static> {
    let rows: Vec<Row<'static>> = memories
        .iter()
        .map(|memory| {
            Row::new(vec![
                Cell::from(truncate_chars(&memory.when, MEMORY_WHEN_CHARS_MAX)),
                Cell::from(truncate_chars(&memory.query, MEMORY_QUERY_CHARS_MAX)),
                Cell::from(truncate_chars(&memory.response, MEMORY_RESPONSE_CHARS_MAX)),
            ])
        })
        .collect();
    Table::new(
        rows,
        [
            ratatui::layout::Constraint::Length(18),
            ratatui::layout::Constraint::Length(52),
            ratatui::layout::Constraint::Min(0),
        ],
    )
    .header(Row::new(vec![
        Cell::from("When"),
        Cell::from("Query"),
        Cell::from("Response"),
    ]))
    .block(Block::bordered().title("Memory"))
    .column_spacing(1)
}

/// One row of the tools table: name and description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRow {
    /// The tool name.
    pub name: String,
    /// The tool description; shown truncated to
    /// [`TOOL_DESCRIPTION_CHARS_MAX`].
    pub description: String,
}

/// The tools table: `Tool | Description` (`flow/tui.py::tools_table`).
pub fn tools_table(tools: &[ToolRow]) -> Table<'static> {
    let rows: Vec<Row<'static>> = tools
        .iter()
        .map(|tool| {
            Row::new(vec![
                Cell::from(tool.name.clone()),
                Cell::from(truncate_chars(
                    &tool.description,
                    TOOL_DESCRIPTION_CHARS_MAX,
                )),
            ])
        })
        .collect();
    Table::new(
        rows,
        [
            ratatui::layout::Constraint::Length(20),
            ratatui::layout::Constraint::Min(0),
        ],
    )
    .header(Row::new(vec![
        Cell::from("Tool"),
        Cell::from("Description"),
    ]))
    .block(Block::bordered().title("Tools"))
    .column_spacing(1)
}

/// A bordered paragraph with a colored title and matching border style.
/// Long lines soft-wrap (`trim: false`) so truncated content keeps its
/// `...` marker, as rich's panels did in the Python TUI.
fn titled_panel(text: &str, title: &str, color: Color) -> Paragraph<'static> {
    let style = Style::default().fg(color);
    Paragraph::new(text.to_string())
        .wrap(ratatui::widgets::Wrap { trim: false })
        .block(
            Block::bordered()
                .title(Line::from(title.to_string()).style(style))
                .border_style(style),
        )
}

/// Python `repr()` for a JSON value, so tool arguments render as they did
/// in the Python TUI (`'strings'`, `True`/`False`, `None`).
///
/// The recursion over nesting depth is bounded upstream: values arrive
/// through `serde_json`, whose parser refuses nesting deeper than 128
/// levels, so this cannot recurse without bound.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => python_string_repr(text),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(key, value)| format!("{}: {}", python_string_repr(key), py_repr(value)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python single-quoted string `repr()`: backslashes and single quotes
/// escaped, non-printables as `\x..`/`\u....`.
fn python_string_repr(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('\'');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 || (ch as u32) == 0x7f => {
                out.push_str(&format!("\\x{:02x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('\'');
    out
}

/// Python `str(value)` for a status-table cell: scalars render plainly,
/// containers render compact JSON.
fn json_scalar_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(_) | Value::Bool(_) => value.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(_) | Value::Object(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| "?".to_string())
        }
    }
}

/// Truncate to `max_chars` Unicode characters, appending `...` when
/// truncated, exactly like Python's `text[:max_chars] + "..."`.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let kept: String = text.chars().take(max_chars).collect();
        format!("{kept}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::widgets::Widget;

    /// Render a widget into a test terminal and return its text rows.
    fn render_text<W: Widget>(widget: W, width: u16, height: u16) -> Vec<String> {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test setup: terminal");
        terminal
            .draw(|frame| frame.render_widget(widget, frame.area()))
            .expect("test setup: draw");
        let buffer = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect()
    }

    fn joined(rows: &[String]) -> String {
        rows.join("\n")
    }

    fn args(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    #[test]
    fn tool_call_panel_shows_name_and_python_repr_args() {
        let rows = render_text(
            tool_call_panel(
                "file_read",
                &args(
                    serde_json::json!({"path": "a.txt", "count": 3, "ok": true, "missing": null}),
                ),
            ),
            80,
            8,
        );
        let text = joined(&rows);
        assert!(text.contains("Tool Call"), "{text}");
        assert!(text.contains("file_read"), "{text}");
        assert!(text.contains("path: 'a.txt'"), "{text}");
        assert!(text.contains("count: 3"), "{text}");
        assert!(text.contains("ok: True"), "{text}");
        assert!(text.contains("missing: None"), "{text}");
    }

    #[test]
    fn tool_result_panel_truncates_long_results() {
        let long = "x".repeat(600);
        let rows = render_text(tool_result_panel("file_read", &long, 500), 80, 12);
        let text = joined(&rows);
        assert!(text.contains("Result: file_read"), "{text}");
        assert!(text.contains("..."), "long results get a truncation marker");
        let short = render_text(tool_result_panel("file_read", "ok", 500), 80, 6);
        let short_text = joined(&short);
        assert!(short_text.contains("ok"), "{short_text}");
        assert!(
            !short_text.contains("..."),
            "short results are not truncated: {short_text}"
        );
    }

    #[test]
    fn error_success_warning_panels_carry_titles() {
        for (panel, title) in [
            (error_panel("boom"), "Error"),
            (success_panel("done"), "Success"),
            (warning_panel("careful"), "Warning"),
        ] {
            let text = joined(&render_text(panel, 40, 5));
            assert!(text.contains(title), "{text}");
        }
        let text = joined(&render_text(error_panel("boom"), 40, 5));
        assert!(text.contains("boom"), "{text}");
    }

    #[test]
    fn code_panel_titles_combine_language_and_title() {
        let text = joined(&render_text(
            code_panel("print(1)", "python", Some("main.py")),
            60,
            5,
        ));
        assert!(text.contains("python — main.py"), "{text}");
        assert!(text.contains("print(1)"), "{text}");
        let bare = joined(&render_text(code_panel("print(1)", "python", None), 60, 5));
        assert!(bare.contains("python"), "{bare}");
    }

    #[test]
    fn status_table_shows_keys_and_plain_values() {
        let items = args(serde_json::json!({"model": "llama3.1", "tools": 7}));
        let text = joined(&render_text(status_table(&items), 80, 8));
        assert!(text.contains("Key"), "{text}");
        assert!(text.contains("Value"), "{text}");
        assert!(text.contains("model"), "{text}");
        assert!(text.contains("llama3.1"), "{text}");
        assert!(text.contains("7"), "{text}");
    }

    #[test]
    fn memory_table_truncates_columns() {
        let rows = vec![MemoryRow {
            when: "2026-09-26T09:10:00.123456".to_string(),
            query: "q".repeat(60),
            response: "r".repeat(70),
        }];
        let text = joined(&render_text(memory_table(&rows), 120, 8));
        assert!(text.contains("When"), "{text}");
        assert!(text.contains("2026-09-26T09:10"), "{text}");
        // The overlong strings are gone; their truncated forms (with the
        // "..." marker, possibly clipped by the column width) remain.
        assert!(!text.contains(&"q".repeat(60)), "query truncated: {text}");
        assert!(
            !text.contains(&"r".repeat(70)),
            "response truncated: {text}"
        );
        assert!(text.contains(&"q".repeat(50)), "{text}");
        // The response column (46 wide at this terminal width) clips the
        // 63-char truncated response; the overlong original is gone.
        assert!(text.contains(&"r".repeat(40)), "{text}");
        assert!(text.contains(".."), "truncation marker present: {text}");
    }

    #[test]
    fn tools_table_truncates_long_descriptions() {
        let rows = vec![ToolRow {
            name: "status".to_string(),
            description: "d".repeat(100),
        }];
        let text = joined(&render_text(tools_table(&rows), 100, 8));
        assert!(text.contains("Tool"), "{text}");
        assert!(text.contains("status"), "{text}");
        assert!(
            !text.contains(&"d".repeat(100)),
            "description truncated: {text}"
        );
        assert!(text.contains("..."), "{text}");
    }
}
