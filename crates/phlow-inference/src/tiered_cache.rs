//! Two-tier cache with asymmetric entry lifetimes.
//!
//! Plain words: not every cached byte deserves the same storage. Global
//! entries (long-horizon context: agent plans, distilled tool results) live
//! a long time and are persisted to disk, so they survive restarts. Local
//! entries (the recent working set: last turns, scratch state) live only in
//! memory with a short TTL. This mirrors the DeepSeek-V4.1-Flash serving
//! design, where global KV persists to SSD for 72h while sliding-window KV
//! is kept in a short-TTL host pool and never persisted.
//!
//! On a local miss the caller does *bounded recompute*: it hands back at
//! most [`REPLAY_ENTRY_MAX`] recent items and a rebuild closure, and the
//! cache stores the rebuilt value. The cache never performs a full rebuild
//! itself, and it rejects oversized replay inputs instead of truncating
//! them silently (fail closed, like the paper's SWA Bounded Replay, which
//! replays only the last `n_win` tokens rather than the full `L x n_win`).
//!
//! # Concurrency
//!
//! Each tier owns one [`RwLock`]. The two locks are never held at the same
//! time (lock ordering is trivially safe: there is no nesting).
//! [`TieredCache::persist`] serializes under the global *read* lock into a
//! stack-local buffer, then performs file I/O with no lock held, so a slow
//! disk never blocks readers. A poisoned lock surfaces as
//! [`CacheError::LockPoisoned`]; the cache does not silently reuse the
//! guarded state.
//!
//! # Persistence format
//!
//! A hand-rolled little-endian binary format (no serde dependency):
//!
//! ```text
//! magic:      12 bytes  "PHLOWCACHEv1"
//! entry_count: u32
//! per entry:  key_len u32, key bytes, value_len u32, value bytes,
//!             stored_at_ms u64, ttl_ms u64
//! ```
//!
//! [`TieredCache::load`] parses the whole file into a temporary map and
//! only swaps it in after every entry validates: corrupt input fails closed
//! and leaves existing state untouched. Writes are atomic (temp file +
//! rename), so a crash mid-persist never leaves a half-written cache.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

/// Maximum bytes per cached value (16 MiB). Checked before allocation.
pub const VALUE_BYTES_MAX: usize = 16 * 1024 * 1024;
/// Maximum bytes per cache key (1 KiB). Checked before hashing.
pub const KEY_BYTES_MAX: usize = 1024;
/// Maximum entries in the memory-only local tier.
pub const LOCAL_CAPACITY_ENTRIES: usize = 4096;
/// Maximum entries in the persisted global tier.
pub const GLOBAL_CAPACITY_ENTRIES: usize = 65_536;
/// Maximum items a single bounded recompute may replay.
pub const REPLAY_ENTRY_MAX: usize = 512;
/// Maximum bytes of a persisted cache file (1 GiB). Checked via metadata
/// before the file is read, so a hostile file cannot force a huge read.
pub const PERSISTED_FILE_BYTES_MAX: u64 = 1024 * 1024 * 1024;
/// On-disk magic, exactly 12 bytes.
const FILE_MAGIC: &[u8; 12] = b"PHLOWCACHEv1";

/// Source of wall-clock time, in milliseconds since the Unix epoch.
///
/// Injected (rather than read inline) so tests can drive TTL expiry with a
/// manual clock instead of sleeping.
pub trait Clock: Send + Sync {
    /// Current time in milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;
}

/// [`Clock`] backed by the operating system clock.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }
}

/// Everything that can go wrong in the tiered cache. External problems
/// (bad input, bad files, I/O) are typed errors; only lock poisoning, which
/// signals a panicked thread elsewhere, escapes as a distinct variant.
#[derive(Debug)]
pub enum CacheError {
    /// Key is empty or longer than [`KEY_BYTES_MAX`].
    InvalidKey {
        /// Actual key length in bytes.
        len: usize,
    },
    /// Value is larger than [`VALUE_BYTES_MAX`].
    ValueTooLarge {
        /// Actual value length in bytes.
        len: usize,
    },
    /// Rebuild was handed more items than [`REPLAY_ENTRY_MAX`].
    ReplayTooLarge {
        /// Actual item count.
        len: usize,
    },
    /// Persisted file failed structural validation. Existing cache state is
    /// left untouched.
    CorruptPersistedData {
        /// Short machine-oriented reason, no file bytes echoed.
        reason: &'static str,
    },
    /// TTL of zero or a zero capacity: a cache that can hold nothing, or an
    /// entry that is dead on arrival, is a configuration bug.
    InvalidConfig {
        /// Which field was invalid.
        field: &'static str,
    },
    /// Filesystem failure while persisting or loading.
    Io(io::Error),
    /// A lock was poisoned by a panicked thread; state may be inconsistent.
    LockPoisoned,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::InvalidKey { len } => {
                write!(f, "cache key must be 1..={KEY_BYTES_MAX} bytes, got {len}")
            }
            CacheError::ValueTooLarge { len } => {
                write!(f, "cache value exceeds {VALUE_BYTES_MAX} bytes, got {len}")
            }
            CacheError::ReplayTooLarge { len } => {
                write!(
                    f,
                    "replay of {len} items exceeds bound of {REPLAY_ENTRY_MAX}"
                )
            }
            CacheError::CorruptPersistedData { reason } => {
                write!(f, "persisted cache failed validation: {reason}")
            }
            CacheError::InvalidConfig { field } => {
                write!(f, "invalid cache configuration: {field}")
            }
            CacheError::Io(err) => write!(f, "cache I/O failed: {err}"),
            CacheError::LockPoisoned => write!(f, "cache lock poisoned"),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for CacheError {
    fn from(err: io::Error) -> Self {
        CacheError::Io(err)
    }
}

/// Configuration for [`TieredCache`]. Capacities are entry counts, not byte
/// counts; values are additionally capped by [`VALUE_BYTES_MAX`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Where the global tier is persisted.
    pub persist_path: PathBuf,
    /// How long local entries live, in milliseconds.
    pub local_ttl_ms: u64,
    /// Maximum entries in the local tier before LRU eviction.
    pub local_capacity_entries: usize,
    /// Maximum entries in the global tier before oldest-first eviction.
    pub global_capacity_entries: usize,
}

impl CacheConfig {
    /// Sensible defaults: 5-minute local TTL, [`LOCAL_CAPACITY_ENTRIES`] /
    /// [`GLOBAL_CAPACITY_ENTRIES`] capacities.
    pub fn new(persist_path: PathBuf) -> Self {
        CacheConfig {
            persist_path,
            local_ttl_ms: 5 * 60 * 1000,
            local_capacity_entries: LOCAL_CAPACITY_ENTRIES,
            global_capacity_entries: GLOBAL_CAPACITY_ENTRIES,
        }
    }

    fn validate(&self) -> Result<(), CacheError> {
        if self.local_ttl_ms == 0 {
            return Err(CacheError::InvalidConfig {
                field: "local_ttl_ms",
            });
        }
        if self.local_capacity_entries == 0 {
            return Err(CacheError::InvalidConfig {
                field: "local_capacity_entries",
            });
        }
        if self.global_capacity_entries == 0 {
            return Err(CacheError::InvalidConfig {
                field: "global_capacity_entries",
            });
        }
        Ok(())
    }
}

fn validate_key(key: &str) -> Result<(), CacheError> {
    let len = key.len();
    if len == 0 || len > KEY_BYTES_MAX {
        return Err(CacheError::InvalidKey { len });
    }
    Ok(())
}

fn validate_value(value: &[u8]) -> Result<(), CacheError> {
    if value.len() > VALUE_BYTES_MAX {
        return Err(CacheError::ValueTooLarge { len: value.len() });
    }
    Ok(())
}

/// Elapsed-time expiry check. Uses saturating subtraction so a backwards
/// clock jump yields "not expired" instead of underflowing.
fn is_expired(stored_at_ms: u64, ttl_ms: u64, now_ms: u64) -> bool {
    now_ms.saturating_sub(stored_at_ms) >= ttl_ms
}

struct GlobalEntry {
    value: Vec<u8>,
    stored_at_ms: u64,
    ttl_ms: u64,
}

struct LocalEntry {
    value: Vec<u8>,
    stored_at_ms: u64,
}

/// Memory-only tier with LRU eviction. `recency` holds keys oldest-first;
/// every successful read or write moves the key to the back.
struct LocalTier {
    entries: HashMap<String, LocalEntry>,
    recency: VecDeque<String>,
}

impl LocalTier {
    fn new() -> Self {
        LocalTier {
            entries: HashMap::new(),
            recency: VecDeque::new(),
        }
    }

    /// Move `key` to the most-recent position. O(capacity); capacity is
    /// bounded by configuration.
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.recency.iter().position(|k| k == key) {
            self.recency.remove(pos);
        }
        self.recency.push_back(key.to_owned());
    }

    /// Evict least-recently-used entries until under `capacity`.
    fn evict_lru(&mut self, capacity: usize) {
        while self.entries.len() >= capacity {
            let Some(oldest) = self.recency.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    fn remove(&mut self, key: &str) {
        self.entries.remove(key);
        if let Some(pos) = self.recency.iter().position(|k| k == key) {
            self.recency.remove(pos);
        }
    }
}

/// Two-tier cache: a persisted global tier with per-entry TTLs and a
/// memory-only local tier with a short TTL and LRU eviction.
///
/// Share the cache between threads with `Arc<TieredCache>` (every method
/// takes `&self`); cloning a *value* out of it copies bytes, which
/// [`get_global`](TieredCache::get_global) and
/// [`get_local`](TieredCache::get_local) document at their call sites.
pub struct TieredCache {
    config: CacheConfig,
    clock: Arc<dyn Clock>,
    global: RwLock<HashMap<String, GlobalEntry>>,
    local: RwLock<LocalTier>,
}

impl TieredCache {
    /// Build a cache from validated config and an explicit clock.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::InvalidConfig`] when a TTL or capacity is zero.
    pub fn new(config: CacheConfig, clock: Arc<dyn Clock>) -> Result<Self, CacheError> {
        config.validate()?;
        Ok(TieredCache {
            config,
            clock,
            global: RwLock::new(HashMap::new()),
            local: RwLock::new(LocalTier::new()),
        })
    }

    /// Store a global entry with its own TTL. Evicts the oldest entries when
    /// the tier is full. The entry is *not* written to disk until
    /// [`persist`](TieredCache::persist) runs.
    pub fn put_global(&self, key: &str, value: &[u8], ttl_ms: u64) -> Result<(), CacheError> {
        validate_key(key)?;
        validate_value(value)?;
        if ttl_ms == 0 {
            return Err(CacheError::InvalidConfig { field: "ttl_ms" });
        }
        let now_ms = self.clock.now_ms();
        let mut global = self.global.write().map_err(|_| CacheError::LockPoisoned)?;
        // Only a *new* key can grow the tier: refreshing an existing key
        // must not evict an innocent entry.
        if !global.contains_key(key) {
            while global.len() >= self.config.global_capacity_entries {
                let oldest = oldest_key(&global);
                let Some(oldest) = oldest else { break };
                global.remove(&oldest);
            }
        }
        global.insert(
            key.to_owned(),
            GlobalEntry {
                value: value.to_vec(),
                stored_at_ms: now_ms,
                ttl_ms,
            },
        );
        Ok(())
    }

    /// Fetch a global entry. Returns `Ok(None)` on miss or expiry; expired
    /// entries are removed lazily. The returned bytes are a copy.
    pub fn get_global(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        validate_key(key)?;
        let now_ms = self.clock.now_ms();
        {
            let global = self.global.read().map_err(|_| CacheError::LockPoisoned)?;
            match global.get(key) {
                Some(entry) if !is_expired(entry.stored_at_ms, entry.ttl_ms, now_ms) => {
                    return Ok(Some(entry.value.clone()));
                }
                Some(_) => {}
                None => return Ok(None),
            }
        }
        // Slow path: the entry exists but expired. Re-check under the write
        // lock in case another thread already removed it. Re-read the clock:
        // it may have moved while waiting for the lock, and the expiry
        // decision must use the freshest timestamp.
        let mut global = self.global.write().map_err(|_| CacheError::LockPoisoned)?;
        let now_ms = self.clock.now_ms();
        match global.get(key) {
            Some(entry) if !is_expired(entry.stored_at_ms, entry.ttl_ms, now_ms) => {
                Ok(Some(entry.value.clone()))
            }
            Some(_) => {
                global.remove(key);
                Ok(None)
            }
            None => Ok(None),
        }
    }

    /// Drop a global entry immediately. Missing keys are not an error.
    pub fn invalidate_global(&self, key: &str) -> Result<(), CacheError> {
        validate_key(key)?;
        let mut global = self.global.write().map_err(|_| CacheError::LockPoisoned)?;
        global.remove(key);
        Ok(())
    }

    /// Store a local (memory-only, short-TTL) entry with LRU eviction.
    pub fn put_local(&self, key: &str, value: &[u8]) -> Result<(), CacheError> {
        validate_key(key)?;
        validate_value(value)?;
        let now_ms = self.clock.now_ms();
        let mut local = self.local.write().map_err(|_| CacheError::LockPoisoned)?;
        // Only a *new* key can grow the tier: refreshing an existing key
        // must not evict an innocent entry.
        if !local.entries.contains_key(key) {
            local.evict_lru(self.config.local_capacity_entries);
        }
        local.entries.insert(
            key.to_owned(),
            LocalEntry {
                value: value.to_vec(),
                stored_at_ms: now_ms,
            },
        );
        local.touch(key);
        Ok(())
    }

    /// Fetch a local entry, refreshing its LRU position. Returns `Ok(None)`
    /// on miss or expiry; the returned bytes are a copy.
    ///
    /// A miss only takes the read lock. A hit upgrades to the write lock to
    /// refresh recency, so recently read entries are not evicted.
    pub fn get_local(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        validate_key(key)?;
        let ttl_ms = self.config.local_ttl_ms;
        {
            let local = self.local.read().map_err(|_| CacheError::LockPoisoned)?;
            if !local.entries.contains_key(key) {
                return Ok(None);
            }
            // Hit (live or expired): re-check under the write lock, which
            // also refreshes LRU recency on live hits.
        }
        let mut local = self.local.write().map_err(|_| CacheError::LockPoisoned)?;
        // Re-read the clock: it may have moved while waiting for the lock.
        let now_ms = self.clock.now_ms();
        match local.entries.get(key) {
            Some(entry) if !is_expired(entry.stored_at_ms, ttl_ms, now_ms) => {
                let value = entry.value.clone();
                local.touch(key);
                Ok(Some(value))
            }
            Some(_) => {
                local.remove(key);
                Ok(None)
            }
            None => Ok(None),
        }
    }

    /// Drop a local entry immediately. Missing keys are not an error.
    pub fn invalidate_local(&self, key: &str) -> Result<(), CacheError> {
        validate_key(key)?;
        let mut local = self.local.write().map_err(|_| CacheError::LockPoisoned)?;
        local.remove(key);
        Ok(())
    }

    /// Bounded recompute for a local miss: rebuild the value from at most
    /// [`REPLAY_ENTRY_MAX`] recent items and store it in the local tier.
    ///
    /// The `rebuild` closure runs *without* any cache lock held, so a slow
    /// rebuild never blocks other threads. The rebuilt value goes through
    /// the same size validation as [`put_local`](TieredCache::put_local).
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::ReplayTooLarge`] when `recent` exceeds the
    /// replay bound; the request is rejected, never silently truncated.
    pub fn rebuild_local<F>(
        &self,
        key: &str,
        recent: &[Vec<u8>],
        rebuild: F,
    ) -> Result<(), CacheError>
    where
        F: FnOnce(&[Vec<u8>]) -> Vec<u8>,
    {
        validate_key(key)?;
        if recent.len() > REPLAY_ENTRY_MAX {
            return Err(CacheError::ReplayTooLarge { len: recent.len() });
        }
        let value = rebuild(recent);
        self.put_local(key, &value)
    }

    /// Atomically persist the global tier to disk. Expired entries are not
    /// written. Returns the number of entries persisted.
    ///
    /// Serialization happens under the global *read* lock into a local
    /// buffer; the file write itself is lock-free (temp file + rename), so
    /// a slow disk never stalls readers.
    pub fn persist(&self) -> Result<usize, CacheError> {
        let now_ms = self.clock.now_ms();
        let buffer = {
            let global = self.global.read().map_err(|_| CacheError::LockPoisoned)?;
            serialize_global(&global, now_ms)?
        };
        let count = entry_count(&buffer)?;
        let tmp_path = self.config.persist_path.with_extension("tmp");
        write_tmp_exclusive(&tmp_path, &buffer)?;
        fs::rename(&tmp_path, &self.config.persist_path)?;
        Ok(count)
    }

    /// Load the global tier from disk, failing closed: the file is fully
    /// parsed and validated into a temporary map first; only then does it
    /// replace the live tier. Expired entries are dropped, not resurrected.
    /// Returns the number of live entries loaded.
    pub fn load(&self) -> Result<usize, CacheError> {
        let metadata = fs::metadata(&self.config.persist_path)?;
        if metadata.len() > PERSISTED_FILE_BYTES_MAX {
            return Err(CacheError::CorruptPersistedData {
                reason: "file exceeds size bound",
            });
        }
        let buffer = fs::read(&self.config.persist_path)?;
        if buffer.len() as u64 > PERSISTED_FILE_BYTES_MAX {
            return Err(CacheError::CorruptPersistedData {
                reason: "file exceeds size bound",
            });
        }
        let now_ms = self.clock.now_ms();
        let parsed = parse_global(&buffer, now_ms)?;
        let live = parsed.len();
        let mut global = self.global.write().map_err(|_| CacheError::LockPoisoned)?;
        *global = parsed;
        Ok(live)
    }

    /// Number of live global entries (expired entries are not counted and
    /// are removed).
    pub fn len_global(&self) -> Result<usize, CacheError> {
        let now_ms = self.clock.now_ms();
        let mut global = self.global.write().map_err(|_| CacheError::LockPoisoned)?;
        global.retain(|_, entry| !is_expired(entry.stored_at_ms, entry.ttl_ms, now_ms));
        Ok(global.len())
    }

    /// Number of live local entries.
    pub fn len_local(&self) -> Result<usize, CacheError> {
        let now_ms = self.clock.now_ms();
        let ttl_ms = self.config.local_ttl_ms;
        let mut local = self.local.write().map_err(|_| CacheError::LockPoisoned)?;
        let expired: Vec<String> = local
            .entries
            .iter()
            .filter(|(_, entry)| is_expired(entry.stored_at_ms, ttl_ms, now_ms))
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            local.remove(&key);
        }
        Ok(local.entries.len())
    }
}

/// Oldest entry by `stored_at_ms`. O(n); the tier size is bounded by
/// configuration, and eviction is the cold path.
fn oldest_key(global: &HashMap<String, GlobalEntry>) -> Option<String> {
    global
        .iter()
        .min_by_key(|(_, entry)| entry.stored_at_ms)
        .map(|(key, _)| key.clone())
}

fn serialize_global(
    global: &HashMap<String, GlobalEntry>,
    now_ms: u64,
) -> Result<Vec<u8>, CacheError> {
    // Sort keys for deterministic output: same map, same bytes.
    let mut keys: Vec<&String> = global.keys().collect();
    keys.sort();
    let mut buffer = Vec::new();
    buffer.extend_from_slice(FILE_MAGIC);
    let mut count: u32 = 0;
    // Reserve space for the count; patched after filtering expired entries.
    buffer.extend_from_slice(&0u32.to_le_bytes());
    for key in keys {
        let entry = &global[key];
        if is_expired(entry.stored_at_ms, entry.ttl_ms, now_ms) {
            continue;
        }
        let key_bytes = key.as_bytes();
        buffer.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(key_bytes);
        buffer.extend_from_slice(&(entry.value.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&entry.value);
        buffer.extend_from_slice(&entry.stored_at_ms.to_le_bytes());
        buffer.extend_from_slice(&entry.ttl_ms.to_le_bytes());
        count += 1;
    }
    let count_bytes = count.to_le_bytes();
    buffer[FILE_MAGIC.len()..FILE_MAGIC.len() + 4].copy_from_slice(&count_bytes);
    Ok(buffer)
}

/// Write the persist buffer to a *new* temp file, never following a symlink.
///
/// `create_new` is `O_CREAT|O_EXCL`: a planted symlink at the tmp path fails
/// with `AlreadyExists` instead of redirecting the write to an
/// attacker-chosen file. On a stale tmp left by a crashed persist, remove
/// and retry once; a re-planted link still fails closed with an I/O error.
fn write_tmp_exclusive(tmp_path: &Path, buffer: &[u8]) -> Result<(), CacheError> {
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tmp_path)
    {
        Ok(mut file) => {
            file.write_all(buffer)?;
            Ok(())
        }
        Err(io_error) if io_error.kind() == io::ErrorKind::AlreadyExists => {
            fs::remove_file(tmp_path)?;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(tmp_path)?;
            file.write_all(buffer)?;
            Ok(())
        }
        Err(io_error) => Err(CacheError::Io(io_error)),
    }
}

/// Entry count from an already-serialized buffer (used by `persist`).
fn entry_count(buffer: &[u8]) -> Result<usize, CacheError> {
    if buffer.len() < FILE_MAGIC.len() + 4 {
        return Err(CacheError::CorruptPersistedData {
            reason: "buffer too short for header",
        });
    }
    let count = u32::from_le_bytes(
        buffer[FILE_MAGIC.len()..FILE_MAGIC.len() + 4]
            .try_into()
            .map_err(|_| CacheError::CorruptPersistedData {
                reason: "header length mismatch",
            })?,
    ) as usize;
    Ok(count)
}

/// Fail-closed parser: any structural problem aborts the whole load and the
/// caller keeps its existing state.
fn parse_global(buffer: &[u8], now_ms: u64) -> Result<HashMap<String, GlobalEntry>, CacheError> {
    let corrupt = |reason: &'static str| CacheError::CorruptPersistedData { reason };
    if buffer.len() < FILE_MAGIC.len() + 4 || &buffer[..FILE_MAGIC.len()] != FILE_MAGIC {
        return Err(corrupt("bad magic or truncated header"));
    }
    let count = u32::from_le_bytes(
        buffer[FILE_MAGIC.len()..FILE_MAGIC.len() + 4]
            .try_into()
            .map_err(|_| corrupt("header length mismatch"))?,
    ) as usize;
    if count > GLOBAL_CAPACITY_ENTRIES {
        return Err(corrupt("entry count exceeds capacity"));
    }
    let mut parsed = HashMap::with_capacity(count.min(1024));
    let mut cursor = FILE_MAGIC.len() + 4;
    for _ in 0..count {
        let key_len =
            read_u32(buffer, &mut cursor).ok_or(corrupt("truncated key length"))? as usize;
        if key_len == 0 || key_len > KEY_BYTES_MAX {
            return Err(corrupt("key length out of bounds"));
        }
        let key_bytes = read_slice(buffer, &mut cursor, key_len).ok_or(corrupt("truncated key"))?;
        let key = std::str::from_utf8(key_bytes).map_err(|_| corrupt("key not UTF-8"))?;
        let value_len =
            read_u32(buffer, &mut cursor).ok_or(corrupt("truncated value length"))? as usize;
        if value_len > VALUE_BYTES_MAX {
            return Err(corrupt("value length out of bounds"));
        }
        let value = read_slice(buffer, &mut cursor, value_len).ok_or(corrupt("truncated value"))?;
        let stored_at_ms = read_u64(buffer, &mut cursor).ok_or(corrupt("truncated timestamp"))?;
        let ttl_ms = read_u64(buffer, &mut cursor).ok_or(corrupt("truncated TTL"))?;
        if ttl_ms == 0 {
            return Err(corrupt("zero TTL"));
        }
        if !is_expired(stored_at_ms, ttl_ms, now_ms) {
            parsed.insert(
                key.to_owned(),
                GlobalEntry {
                    value: value.to_vec(),
                    stored_at_ms,
                    ttl_ms,
                },
            );
        }
    }
    if cursor != buffer.len() {
        return Err(corrupt("trailing bytes after entries"));
    }
    Ok(parsed)
}

fn read_u32(buffer: &[u8], cursor: &mut usize) -> Option<u32> {
    let bytes = read_slice(buffer, cursor, 4)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

fn read_u64(buffer: &[u8], cursor: &mut usize) -> Option<u64> {
    let bytes = read_slice(buffer, cursor, 8)?;
    Some(u64::from_le_bytes(bytes.try_into().ok()?))
}

/// Bounds-checked slice read: validates the range *before* computing the
/// end, so a hostile length cannot overflow into a smaller accepted range.
fn read_slice<'a>(buffer: &'a [u8], cursor: &mut usize, len: usize) -> Option<&'a [u8]> {
    let remaining = buffer.len().checked_sub(*cursor)?;
    if len > remaining {
        return None;
    }
    let end = cursor.checked_add(len)?;
    let slice = &buffer[*cursor..end];
    *cursor = end;
    Some(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    /// Unique temp dir per test: unit tests run in parallel threads of one
    /// process, so the process id alone does not isolate them.
    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct ManualClock {
        now_ms: AtomicU64,
    }

    impl ManualClock {
        fn new(now_ms: u64) -> Arc<Self> {
            Arc::new(ManualClock {
                now_ms: AtomicU64::new(now_ms),
            })
        }

        fn advance(&self, delta_ms: u64) {
            self.now_ms.fetch_add(delta_ms, Ordering::SeqCst);
        }
    }

    impl Clock for ManualClock {
        fn now_ms(&self) -> u64 {
            self.now_ms.load(Ordering::SeqCst)
        }
    }

    fn test_cache(ttl_ms: u64) -> (TieredCache, Arc<ManualClock>) {
        let clock = ManualClock::new(1_000_000);
        let dir = std::env::temp_dir().join(format!(
            "phlow-tiered-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = fs::create_dir_all(&dir);
        let mut config = CacheConfig::new(dir.join("cache.bin"));
        config.local_ttl_ms = ttl_ms;
        let cache = TieredCache::new(config, clock.clone()).expect("valid config");
        (cache, clock)
    }

    #[test]
    fn global_roundtrip() {
        let (cache, _clock) = test_cache(60_000);
        cache
            .put_global("plan", b"do the thing", 3_600_000)
            .unwrap();
        assert_eq!(
            cache.get_global("plan").unwrap(),
            Some(b"do the thing".to_vec())
        );
        assert_eq!(cache.get_global("missing").unwrap(), None);
    }

    #[test]
    fn global_ttl_expiry_removes_entry() {
        let (cache, clock) = test_cache(60_000);
        cache.put_global("k", b"v", 1_000).unwrap();
        assert!(cache.get_global("k").unwrap().is_some());
        clock.advance(1_000);
        assert_eq!(cache.get_global("k").unwrap(), None);
        assert_eq!(cache.len_global().unwrap(), 0);
    }

    #[test]
    fn local_ttl_expiry() {
        let (cache, clock) = test_cache(500);
        cache.put_local("w", b"work").unwrap();
        assert!(cache.get_local("w").unwrap().is_some());
        clock.advance(499);
        assert!(cache.get_local("w").unwrap().is_some());
        clock.advance(1);
        assert_eq!(cache.get_local("w").unwrap(), None);
    }

    #[test]
    fn local_lru_evicts_oldest() {
        let clock = ManualClock::new(0);
        let dir = std::env::temp_dir().join(format!("phlow-lru-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let mut config = CacheConfig::new(dir.join("c.bin"));
        config.local_ttl_ms = 60_000;
        config.local_capacity_entries = 2;
        let cache = TieredCache::new(config, clock).unwrap();
        cache.put_local("a", b"a").unwrap();
        cache.put_local("b", b"b").unwrap();
        // Touch "a" so "b" becomes the eviction victim.
        assert!(cache.get_local("a").unwrap().is_some());
        cache.put_local("c", b"c").unwrap();
        assert!(cache.get_local("a").unwrap().is_some());
        assert_eq!(cache.get_local("b").unwrap(), None);
        assert!(cache.get_local("c").unwrap().is_some());
    }

    #[test]
    fn rebuild_local_bounded() {
        let (cache, _clock) = test_cache(60_000);
        let recent = vec![b"one".to_vec(), b"two".to_vec()];
        cache
            .rebuild_local("sum", &recent, |items| items.concat())
            .unwrap();
        assert_eq!(cache.get_local("sum").unwrap(), Some(b"onetwo".to_vec()));

        let too_many = vec![Vec::new(); REPLAY_ENTRY_MAX + 1];
        let err = cache
            .rebuild_local("x", &too_many, |items| items.concat())
            .unwrap_err();
        assert!(matches!(err, CacheError::ReplayTooLarge { len } if len == REPLAY_ENTRY_MAX + 1));
        // Rejected rebuild stores nothing.
        assert_eq!(cache.get_local("x").unwrap(), None);

        // Exactly at the bound is accepted.
        let at_bound = vec![b"z".to_vec(); REPLAY_ENTRY_MAX];
        cache
            .rebuild_local("edge", &at_bound, |_| b"ok".to_vec())
            .unwrap();
        assert_eq!(cache.get_local("edge").unwrap(), Some(b"ok".to_vec()));
    }

    #[test]
    fn persist_load_roundtrip() {
        let (cache, clock) = test_cache(60_000);
        cache.put_global("g1", b"v1", 3_600_000).unwrap();
        cache.put_global("g2", b"v2", 3_600_000).unwrap();
        // Local entries must NOT persist.
        cache.put_local("tmp", b"volatile").unwrap();
        let persisted = cache.persist().unwrap();
        assert_eq!(persisted, 2);

        // Point a fresh cache at the same file.
        let path = cache.config.persist_path.clone();
        let mut config = CacheConfig::new(path);
        config.local_ttl_ms = 60_000;
        let fresh = TieredCache::new(config, clock).expect("valid config");
        let loaded = fresh.load().unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(fresh.get_global("g1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(fresh.get_global("g2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(fresh.get_local("tmp").unwrap(), None);
        let _ = fs::remove_file(&cache.config.persist_path);
    }

    #[test]
    fn load_fails_closed_on_corrupt_file() {
        let (cache, _clock) = test_cache(60_000);
        cache.put_global("keep", b"me", 3_600_000).unwrap();
        fs::write(&cache.config.persist_path, b"definitely not a cache file").unwrap();
        let err = cache.load().unwrap_err();
        assert!(matches!(err, CacheError::CorruptPersistedData { .. }));
        // Existing state untouched.
        assert_eq!(cache.get_global("keep").unwrap(), Some(b"me".to_vec()));
        let _ = fs::remove_file(&cache.config.persist_path);
    }

    #[test]
    fn load_rejects_truncated_and_oversized_counts() {
        let (cache, _clock) = test_cache(60_000);
        // Truncated after magic.
        fs::write(&cache.config.persist_path, b"PHLOWCACHEv1").unwrap();
        assert!(matches!(
            cache.load().unwrap_err(),
            CacheError::CorruptPersistedData { .. }
        ));
        // Declared count beyond capacity.
        let mut bad = Vec::from(b"PHLOWCACHEv1".as_slice());
        bad.extend_from_slice(&(GLOBAL_CAPACITY_ENTRIES as u32 + 1).to_le_bytes());
        fs::write(&cache.config.persist_path, &bad).unwrap();
        assert!(matches!(
            cache.load().unwrap_err(),
            CacheError::CorruptPersistedData { .. }
        ));
        let _ = fs::remove_file(&cache.config.persist_path);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let (cache, _clock) = test_cache(60_000);
        assert!(matches!(
            cache.put_global("", b"v", 1000).unwrap_err(),
            CacheError::InvalidKey { len: 0 }
        ));
        let long_key = "k".repeat(KEY_BYTES_MAX + 1);
        assert!(matches!(
            cache.put_global(&long_key, b"v", 1000).unwrap_err(),
            CacheError::InvalidKey { .. }
        ));
        let big = vec![0u8; VALUE_BYTES_MAX + 1];
        assert!(matches!(
            cache.put_global("k", &big, 1000).unwrap_err(),
            CacheError::ValueTooLarge { .. }
        ));
        assert!(matches!(
            cache.put_global("k", b"v", 0).unwrap_err(),
            CacheError::InvalidConfig { .. }
        ));
        let mut bad_config = CacheConfig::new("x".into());
        bad_config.local_ttl_ms = 0;
        assert!(TieredCache::new(bad_config, Arc::new(SystemClock)).is_err());
    }

    #[test]
    fn global_capacity_evicts_oldest() {
        let clock = ManualClock::new(0);
        let dir = std::env::temp_dir().join(format!("phlow-gcap-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let mut config = CacheConfig::new(dir.join("c.bin"));
        config.local_ttl_ms = 60_000;
        config.global_capacity_entries = 2;
        let cache = TieredCache::new(config, clock.clone()).unwrap();
        cache.put_global("first", b"1", 60_000).unwrap();
        clock.advance(10);
        cache.put_global("second", b"2", 60_000).unwrap();
        clock.advance(10);
        cache.put_global("third", b"3", 60_000).unwrap();
        assert_eq!(cache.get_global("first").unwrap(), None);
        assert!(cache.get_global("second").unwrap().is_some());
        assert!(cache.get_global("third").unwrap().is_some());
    }

    #[test]
    fn persist_skips_expired_entries() {
        let (cache, clock) = test_cache(60_000);
        cache.put_global("live", b"1", 60_000).unwrap();
        cache.put_global("dead", b"2", 100).unwrap();
        clock.advance(200);
        let persisted = cache.persist().unwrap();
        assert_eq!(persisted, 1);
        let _ = fs::remove_file(&cache.config.persist_path);
    }
}
