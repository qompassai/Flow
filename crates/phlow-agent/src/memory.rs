//! SQLite FTS5 memory store. Mirrors `flow/agent/memory.py`.
//!
//! Schema, trigger, and output shapes are preserved exactly: the `memories`
//! table, the `memories_fts` FTS5 index fed by the `memories_ai` insert
//! trigger, `Q: …\nA: …` search hits truncated to 100/200 chars, and
//! `recent()` entries shaped like `{"query", "response", "created_at"}`.
//! `search` never fails — like Python's `except Exception: return []`, any
//! storage or FTS syntax error yields an empty hit list.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};

/// Longest stored query, in chars.
pub const QUERY_CHARS_MAX: usize = 10_000;
/// Longest stored response, in chars.
pub const RESPONSE_CHARS_MAX: usize = 100_000;
/// Longest single tag, in chars.
pub const TAG_CHARS_MAX: usize = 64;
/// Most tags per memory.
pub const TAGS_MAX: usize = 32;
/// Longest FTS search term, in chars.
pub const SEARCH_TERM_CHARS_MAX: usize = 500;
/// Most hits a search returns.
pub const TOP_K_MAX: u32 = 100;
/// Most entries `recent` returns.
pub const RECENT_MAX: u32 = 1_000;
/// Chars of the query shown in a search hit (Python `r[0][:100]`).
pub const HIT_QUERY_CHARS: usize = 100;
/// Chars of the response shown in a search hit (Python `r[1][:200]`).
pub const HIT_RESPONSE_CHARS: usize = 200;

/// Failures from the memory store. Expected storage/validation failures
/// are typed; `search` swallows its errors into an empty list instead.
#[derive(Debug)]
pub enum MemoryError {
    /// SQLite failure (open, schema, insert, read).
    Database(rusqlite::Error),
    /// A field exceeded its char budget.
    TooLong {
        /// Which field (`"query"`, `"response"`, `"tag"`, `"search term"`).
        field: &'static str,
        /// The budget that was exceeded.
        max_chars: usize,
    },
    /// Too many tags on one memory.
    TooManyTags {
        /// The tag budget that was exceeded.
        max_tags: usize,
    },
    /// The database path has no usable parent or file name.
    BadPath(PathBuf),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Database(error) => write!(f, "memory database error: {error}"),
            Self::TooLong { field, max_chars } => {
                write!(f, "{field} exceeds {max_chars} chars")
            }
            Self::TooManyTags { max_tags } => {
                write!(f, "more than {max_tags} tags")
            }
            Self::BadPath(path) => {
                write!(f, "unusable memory database path: {}", path.display())
            }
        }
    }
}

impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Database(error) => Some(error),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for MemoryError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

/// One entry from [`MemoryStore::recent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEntry {
    /// The stored query.
    pub query: String,
    /// The stored response.
    pub response: String,
    /// UTC ISO-8601 creation timestamp, like Python's `utcnow().isoformat()`.
    pub created_at: String,
}

/// Persistent memory using SQLite full-text search.
///
/// Owns one SQLite [`Connection`]; wrap in a `Mutex` to share across
/// threads. Every mutating call commits before returning, like Python's
/// per-operation `sqlite3.connect`.
pub struct MemoryStore {
    db_path: PathBuf,
    connection: Connection,
}

impl MemoryStore {
    /// Open (creating parent directories) and initialize the schema.
    pub fn open(db_path: &Path) -> Result<Self, MemoryError> {
        let parent = db_path
            .parent()
            .ok_or_else(|| MemoryError::BadPath(db_path.to_path_buf()))?;
        std::fs::create_dir_all(parent).map_err(|_| MemoryError::BadPath(db_path.to_path_buf()))?;
        let connection = Connection::open(db_path).map_err(MemoryError::Database)?;
        let store = Self {
            db_path: db_path.to_path_buf(),
            connection,
        };
        store.init_schema()?;
        Ok(store)
    }

    /// The database file path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Store one memory. Tags serialize to JSON like Python's
    /// `json.dumps(tags or [])`.
    pub fn store(&self, query: &str, response: &str, tags: &[&str]) -> Result<(), MemoryError> {
        check_chars("query", query, QUERY_CHARS_MAX)?;
        check_chars("response", response, RESPONSE_CHARS_MAX)?;
        if tags.len() > TAGS_MAX {
            return Err(MemoryError::TooManyTags { max_tags: TAGS_MAX });
        }
        for tag in tags {
            check_chars("tag", tag, TAG_CHARS_MAX)?;
        }
        let tags_json = serde_json::to_string(tags).expect("tags are strings");
        self.connection.execute(
            "INSERT INTO memories (query, response, tags, created_at) VALUES (?, ?, ?, ?)",
            params![query, response, tags_json, utc_now_iso()],
        )?;
        Ok(())
    }

    /// Full-text search over queries and responses. Never fails: any
    /// storage error or FTS5 syntax error yields an empty list, mirroring
    /// Python's `except Exception: return []`.
    pub fn search(&self, query: &str, top_k: u32) -> Vec<String> {
        if query.chars().count() > SEARCH_TERM_CHARS_MAX {
            return Vec::new();
        }
        let top_k = top_k.min(TOP_K_MAX);
        let mut statement = match self.connection.prepare(
            "SELECT m.query, m.response FROM memories_fts \
             JOIN memories m ON memories_fts.rowid = m.id \
             WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
        ) {
            Ok(statement) => statement,
            Err(_) => return Vec::new(),
        };
        let rows = statement.query_map(params![query, top_k], |row| {
            let query: String = row.get(0)?;
            let response: String = row.get(1)?;
            Ok((query, response))
        });
        match rows {
            Ok(mapped) => mapped
                .filter_map(|row| row.ok())
                .map(|(query, response)| {
                    format!(
                        "Q: {}\nA: {}",
                        truncate_chars(&query, HIT_QUERY_CHARS),
                        truncate_chars(&response, HIT_RESPONSE_CHARS)
                    )
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// The `n` most recent memories, newest first.
    pub fn recent(&self, n: u32) -> Result<Vec<MemoryEntry>, MemoryError> {
        let n = n.min(RECENT_MAX);
        let mut statement = self
            .connection
            .prepare("SELECT query, response, created_at FROM memories ORDER BY id DESC LIMIT ?")?;
        let entries = statement.query_map(params![n], |row| {
            Ok(MemoryEntry {
                query: row.get(0)?,
                response: row.get(1)?,
                created_at: row.get(2)?,
            })
        })?;
        entries
            .collect::<Result<Vec<_>, _>>()
            .map_err(MemoryError::from)
    }

    /// Delete all memories, mirroring Python's `clear()`. Orphaned FTS5
    /// index rows are harmless: `search` joins against `memories`, so
    /// deleted rowids never surface.
    pub fn clear(&self) -> Result<(), MemoryError> {
        self.connection.execute("DELETE FROM memories", [])?;
        Ok(())
    }

    /// Create the table, FTS5 index, and insert trigger when absent.
    fn init_schema(&self) -> Result<(), MemoryError> {
        self.connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(query, response, content='memories', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, query, response)
                VALUES (new.id, new.query, new.response);
            END;",
        )?;
        Ok(())
    }
}

/// Reject a field longer than its char budget.
fn check_chars(field: &'static str, value: &str, max_chars: usize) -> Result<(), MemoryError> {
    if value.chars().count() > max_chars {
        return Err(MemoryError::TooLong { field, max_chars });
    }
    Ok(())
}

/// Truncate to a char budget, like Python's `value[:n]` slicing.
fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

/// Current UTC time as ISO-8601 (`2026-09-26T12:34:56.789012`), mirroring
/// `datetime.utcnow().isoformat()`. Dependency-free: civil date from
/// Howard Hinnant's days-to-civil algorithm.
fn utc_now_iso() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let days = (now.as_secs() / 86_400) as i64;
    let secs_of_day = now.as_secs() % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}",
        secs_of_day / 3_600,
        (secs_of_day % 3_600) / 60,
        secs_of_day % 60,
        now.subsec_micros(),
    )
}

/// Days since the Unix epoch to (year, month, day).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32;
    let full_year = if month <= 2 { year + 1 } else { year };
    (full_year, month, day)
}
