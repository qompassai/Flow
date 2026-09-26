//! The SQLite feedback store.
//!
//! Ports `flow/self_improve/feedback.py`. The schema is identical: the
//! `CREATE TABLE` statement below normalizes in `sqlite_master` to exactly
//! the text Python's `_init_db` stored (verified against the real Python
//! on 2026-09-26). Row keys (`id`, `session_id`, `rating`, `comment`,
//! `prompt_used`, `outcome`, `created_at`) and query semantics
//! (`rating <= threshold`, `ORDER BY created_at DESC`, `AVG` defaulting to
//! `0.0`) are preserved.
//!
//! Like the agent memory store, this owns one synchronous [`Connection`];
//! feedback recording is rare, so there is no background thread, no queue,
//! and no batching.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};

use crate::error::SelfImproveError;

/// The feedback table DDL. Written so SQLite's `sqlite_master` normalization
/// yields exactly the text Python stored: `CREATE TABLE feedback (...)`
/// with 20-space column indentation.
const FEEDBACK_DDL: &str = "CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    prompt_used TEXT,
                    outcome TEXT,
                    created_at TEXT
                )";

/// `sqlite_master.sql` for the feedback table, as Python's `_init_db`
/// stored it. The schema-parity test asserts this byte-for-byte.
pub const FEEDBACK_SCHEMA_SQL: &str = "CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    prompt_used TEXT,
                    outcome TEXT,
                    created_at TEXT
                )";

/// A session id longer than this is rejected; it is an operator-chosen
/// label, not a payload.
pub const SESSION_ID_CHARS_MAX: usize = 256;

/// A comment longer than this is rejected; feedback is a short note.
pub const COMMENT_CHARS_MAX: usize = 10_000;

/// A stored prompt longer than this is rejected; prompts are bounded
/// elsewhere and must not smuggle unbounded text into the database.
pub const PROMPT_USED_CHARS_MAX: usize = 100_000;

/// An outcome longer than this is rejected; outcomes are short labels.
pub const OUTCOME_CHARS_MAX: usize = 10_000;

/// `get_low_rated` never returns more than this many rows.
pub const LOW_RATED_LIMIT_MAX: u32 = 1000;

/// One feedback row, keyed exactly as Python's `dict(zip(columns, row))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackEntry {
    /// Row id (`INTEGER PRIMARY KEY AUTOINCREMENT`).
    pub id: i64,
    /// The session the feedback belongs to.
    pub session_id: String,
    /// Numeric rating.
    pub rating: i64,
    /// Free-text comment.
    pub comment: String,
    /// The prompt that was used.
    pub prompt_used: String,
    /// The session outcome label.
    pub outcome: String,
    /// UTC timestamp, `datetime.utcnow().isoformat()` shape.
    pub created_at: String,
}

impl FeedbackEntry {
    /// The row as a JSON object with Python's row keys.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "session_id": self.session_id,
            "rating": self.rating,
            "comment": self.comment,
            "prompt_used": self.prompt_used,
            "outcome": self.outcome,
            "created_at": self.created_at,
        })
    }
}

/// The feedback store: record rows, list low-rated sessions, average
/// ratings. Owns one SQLite [`Connection`]; wrap in a `Mutex` to share
/// across threads.
pub struct FeedbackStore {
    db_path: PathBuf,
    connection: Connection,
}

impl FeedbackStore {
    /// Open (creating parent directories) and initialize the schema.
    pub fn open(db_path: &Path) -> Result<FeedbackStore, SelfImproveError> {
        let parent = db_path
            .parent()
            .ok_or_else(|| SelfImproveError::BadPath(db_path.to_path_buf()))?;
        std::fs::create_dir_all(parent).map_err(SelfImproveError::Io)?;
        let connection = Connection::open(db_path).map_err(SelfImproveError::Database)?;
        connection.execute_batch(FEEDBACK_DDL)?;
        Ok(FeedbackStore {
            db_path: db_path.to_path_buf(),
            connection,
        })
    }

    /// The database file path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Record one feedback row, timestamped like Python's
    /// `datetime.utcnow().isoformat()`.
    pub fn record(
        &self,
        session_id: &str,
        rating: i64,
        comment: &str,
        prompt_used: &str,
        outcome: &str,
    ) -> Result<(), SelfImproveError> {
        check_chars("session_id", session_id, SESSION_ID_CHARS_MAX)?;
        check_chars("comment", comment, COMMENT_CHARS_MAX)?;
        check_chars("prompt_used", prompt_used, PROMPT_USED_CHARS_MAX)?;
        check_chars("outcome", outcome, OUTCOME_CHARS_MAX)?;
        self.connection.execute(
            "INSERT INTO feedback (session_id, rating, comment, prompt_used, outcome, created_at)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                session_id,
                rating,
                comment,
                prompt_used,
                outcome,
                utc_now_iso()
            ],
        )?;
        Ok(())
    }

    /// Rows with `rating <= threshold`, newest first, at most `limit` rows.
    /// Mirrors `get_low_rated(threshold=3, limit=10)`.
    pub fn get_low_rated(
        &self,
        threshold: i64,
        limit: u32,
    ) -> Result<Vec<FeedbackEntry>, SelfImproveError> {
        let limit = limit.min(LOW_RATED_LIMIT_MAX);
        let mut statement = self.connection.prepare(
            "SELECT id, session_id, rating, comment, prompt_used, outcome, created_at
             FROM feedback WHERE rating <= ? ORDER BY created_at DESC LIMIT ?",
        )?;
        let rows = statement.query_map(params![threshold, limit], |row| {
            Ok(FeedbackEntry {
                id: row.get(0)?,
                session_id: row.get(1)?,
                rating: row.get(2)?,
                comment: row.get(3)?,
                prompt_used: row.get(4)?,
                outcome: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
            if entries.len() >= LOW_RATED_LIMIT_MAX as usize {
                break;
            }
        }
        Ok(entries)
    }

    /// The average rating, or `0.0` when no feedback exists, as in Python.
    pub fn average_rating(&self) -> Result<f64, SelfImproveError> {
        let average: Option<f64> =
            self.connection
                .query_row("SELECT AVG(rating) FROM feedback", [], |row| row.get(0))?;
        Ok(average.unwrap_or(0.0))
    }
}

/// Reject over-long text fields before they reach the database.
fn check_chars(field: &'static str, value: &str, max_chars: usize) -> Result<(), SelfImproveError> {
    if value.chars().count() > max_chars {
        return Err(SelfImproveError::TooLong { field, max_chars });
    }
    Ok(())
}

/// UTC timestamp in `datetime.utcnow().isoformat()` shape
/// (`YYYY-MM-DDTHH:MM:SS.ffffff`), so `ORDER BY created_at DESC` sorts
/// newest first exactly as in Python.
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

/// Days since the Unix epoch to a civil date (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month + 2) / 5 + 1;
    let month = if month < 10 { month + 3 } else { month - 9 };
    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn db_path(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-self-improve-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir.join("feedback.db")
    }

    fn cleanup(path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir_all(parent);
        }
    }

    /// Insert rows with explicit timestamps through a second connection,
    /// so ordering tests are deterministic.
    fn insert_row(db: &Path, session: &str, rating: i64, created_at: &str) {
        let conn = Connection::open(db).expect("test setup: open db");
        conn.execute(
            "INSERT INTO feedback (session_id, rating, comment, prompt_used, outcome, created_at)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![session, rating, "", "", "", created_at],
        )
        .expect("test setup: insert row");
    }

    #[test]
    fn schema_matches_python_sqlite_master_byte_for_byte() {
        let db = db_path("schema");
        let store = FeedbackStore::open(&db).unwrap();
        let sql: String = store
            .connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name = 'feedback'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sql, FEEDBACK_SCHEMA_SQL);
        cleanup(&db);
    }

    #[test]
    fn record_round_trip_preserves_row_keys() {
        let db = db_path("roundtrip");
        let store = FeedbackStore::open(&db).unwrap();
        store
            .record("s1", 5, "great", "the prompt", "done")
            .unwrap();
        let rows = store.get_low_rated(10, 10).unwrap();
        assert_eq!(rows.len(), 1);
        let entry = &rows[0];
        assert_eq!(entry.session_id, "s1");
        assert_eq!(entry.rating, 5);
        assert_eq!(entry.comment, "great");
        assert_eq!(entry.prompt_used, "the prompt");
        assert_eq!(entry.outcome, "done");
        assert!(!entry.created_at.is_empty());
        // Row keys match Python's dict(zip(columns, row)).
        let json = entry.to_json();
        for key in [
            "id",
            "session_id",
            "rating",
            "comment",
            "prompt_used",
            "outcome",
            "created_at",
        ] {
            assert!(json.get(key).is_some(), "missing key {key}");
        }
        cleanup(&db);
    }

    #[test]
    fn low_rated_filters_by_threshold_and_orders_newest_first() {
        let db = db_path("lowrated");
        let store = FeedbackStore::open(&db).unwrap();
        insert_row(&db, "old", 1, "2026-01-01T00:00:00.000001");
        insert_row(&db, "new", 2, "2026-01-03T00:00:00.000001");
        insert_row(&db, "edge", 3, "2026-01-04T00:00:00.000001");
        insert_row(&db, "high", 5, "2026-01-02T00:00:00.000001");
        // threshold=3: ratings 1, 2 and 3 qualify (Python: rating <=
        // threshold); 5 does not.
        let rows = store.get_low_rated(3, 10).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].session_id, "edge");
        assert_eq!(rows[1].session_id, "new");
        assert_eq!(rows[2].session_id, "old");
        cleanup(&db);
    }

    #[test]
    fn low_rated_respects_limit() {
        let db = db_path("limit");
        let store = FeedbackStore::open(&db).unwrap();
        for i in 0..5 {
            insert_row(
                &db,
                &format!("s{i}"),
                1,
                &format!("2026-01-0{i}T00:00:00.000001"),
            );
        }
        let rows = store.get_low_rated(3, 2).unwrap();
        assert_eq!(rows.len(), 2);
        cleanup(&db);
    }

    #[test]
    fn average_rating_matches_python() {
        let db = db_path("avg");
        let store = FeedbackStore::open(&db).unwrap();
        // Empty database averages to 0.0, as in Python.
        assert_eq!(store.average_rating().unwrap(), 0.0);
        store.record("s1", 5, "", "", "").unwrap();
        store.record("s2", 2, "", "", "").unwrap();
        assert_eq!(store.average_rating().unwrap(), 3.5);
        cleanup(&db);
    }

    #[test]
    fn overlong_fields_are_rejected() {
        let db = db_path("bounds");
        let store = FeedbackStore::open(&db).unwrap();
        let long = "x".repeat(SESSION_ID_CHARS_MAX + 1);
        let error = store.record(&long, 5, "", "", "").unwrap_err();
        assert!(matches!(
            error,
            SelfImproveError::TooLong {
                field: "session_id",
                ..
            }
        ));
        let long_comment = "x".repeat(COMMENT_CHARS_MAX + 1);
        assert!(store.record("s", 5, &long_comment, "", "").is_err());
        cleanup(&db);
    }

    #[test]
    fn open_creates_parent_directories() {
        let dir = std::env::temp_dir().join(format!(
            "phlow-self-improve-test-parents-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let db = dir.join("nested").join("feedback.db");
        let store = FeedbackStore::open(&db).unwrap();
        assert_eq!(store.db_path(), db.as_path());
        assert!(db.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
