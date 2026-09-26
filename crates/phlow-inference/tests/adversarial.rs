//! Adversarial tests for `phlow-inference`.
//!
//! Plain words: these tests try to break the four modules instead of
//! confirming they work. Each test names the attack it attempts. Tests
//! that exposed real bugs carry a `REGRESSION:` note naming the bug and
//! the fix. Builder C's 44 unit tests are untouched; this file only adds
//! integration coverage through the public API.

use phlow_inference::kv_policy::{
    KvComponent, Precision, QuantPolicy, bytes_per_token, recommend, validate,
};
use phlow_inference::speculative::{
    DRAFT_TOKEN_MAX, SpecError, ThroughputPoint, ThroughputTable, draft_verify, schedule_for_load,
};
use phlow_inference::tiered_cache::{CacheConfig, CacheError, Clock, TieredCache, VALUE_BYTES_MAX};
use phlow_inference::two_stage::{TwoStageConfig, rank};
use std::fs;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{
    Arc, Barrier, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

/// Unique temp dir per test: the whole suite runs in parallel threads.
static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

fn test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "phlow-adv-{}-{}-{tag}",
        std::process::id(),
        TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

/// Manually driven clock for TTL attacks.
struct ManualClock {
    now_ms: AtomicU64,
}

impl ManualClock {
    fn new(now_ms: u64) -> Arc<Self> {
        Arc::new(ManualClock {
            now_ms: AtomicU64::new(now_ms),
        })
    }

    fn set(&self, now_ms: u64) {
        self.now_ms.store(now_ms, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> u64 {
        self.now_ms.load(Ordering::SeqCst)
    }
}

/// Clock returning a scripted timestamp sequence, then repeating the last.
/// Models a clock that jumps between two reads of the same operation.
struct ScriptedClock {
    times_ms: Vec<u64>,
    calls: AtomicU64,
}

impl ScriptedClock {
    fn arc(times_ms: Vec<u64>) -> Arc<Self> {
        assert!(!times_ms.is_empty(), "script needs at least one timestamp");
        Arc::new(ScriptedClock {
            times_ms,
            calls: AtomicU64::new(0),
        })
    }
}

impl Clock for ScriptedClock {
    fn now_ms(&self) -> u64 {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        self.times_ms[call.min(self.times_ms.len() - 1)]
    }
}

fn test_cache(tag: &str, local_ttl_ms: u64, clock: Arc<dyn Clock>) -> TieredCache {
    let mut config = CacheConfig::new(test_dir(tag).join("cache.bin"));
    config.local_ttl_ms = local_ttl_ms;
    TieredCache::new(config, clock).expect("valid config")
}

// ---------------------------------------------------------------------------
// tiered_cache attacks
// ---------------------------------------------------------------------------

#[test]
fn ttl_boundary_global_live_one_ms_before_dead_at() {
    let clock = ManualClock::new(0);
    let dyn_clock: Arc<dyn Clock> = clock.clone();
    let cache = test_cache("ttl-boundary", 60_000, dyn_clock);
    cache.put_global("k", b"v", 1_000).unwrap();
    clock.set(999);
    assert_eq!(cache.get_global("k").unwrap(), Some(b"v".to_vec()));
    clock.set(1_000);
    assert_eq!(cache.get_global("k").unwrap(), None);
}

#[test]
fn clock_going_backwards_keeps_entries_live() {
    let clock = ManualClock::new(1_000);
    let dyn_clock: Arc<dyn Clock> = clock.clone();
    let cache = test_cache("clock-back", 60_000, dyn_clock);
    cache.put_global("g", b"v", 60_000).unwrap();
    cache.put_local("l", b"v").unwrap();
    // Backwards jump: saturating arithmetic must not underflow or expire.
    clock.set(500);
    assert_eq!(cache.get_global("g").unwrap(), Some(b"v".to_vec()));
    assert_eq!(cache.get_local("l").unwrap(), Some(b"v".to_vec()));
}

#[test]
fn refresh_existing_local_key_evicts_nobody() {
    // REGRESSION: put_local evicted the LRU entry even when the key already
    // existed, so refreshing a hot key killed an innocent entry. Eviction
    // now only runs for genuinely new keys.
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let mut config = CacheConfig::new(test_dir("refresh-local").join("c.bin"));
    config.local_ttl_ms = 60_000;
    config.local_capacity_entries = 2;
    let cache = TieredCache::new(config, clock).unwrap();
    cache.put_local("a", b"1").unwrap();
    cache.put_local("b", b"2").unwrap();
    // Touch "a" so "b" is the LRU victim if eviction misfires.
    assert!(cache.get_local("a").unwrap().is_some());
    cache.put_local("a", b"1-new").unwrap();
    assert_eq!(
        cache.get_local("b").unwrap(),
        Some(b"2".to_vec()),
        "refreshing 'a' evicted 'b'"
    );
    assert_eq!(cache.get_local("a").unwrap(), Some(b"1-new".to_vec()));
}

#[test]
fn refresh_existing_global_key_evicts_nobody() {
    // REGRESSION: put_global evicted the oldest entry even when refreshing
    // an existing key. Same fix as the local tier.
    let clock = ManualClock::new(0);
    let dyn_clock: Arc<dyn Clock> = clock.clone();
    let mut config = CacheConfig::new(test_dir("refresh-global").join("c.bin"));
    config.local_ttl_ms = 60_000;
    config.global_capacity_entries = 2;
    let cache = TieredCache::new(config, dyn_clock).unwrap();
    cache.put_global("first", b"1", 60_000).unwrap();
    clock.set(10);
    cache.put_global("second", b"2", 60_000).unwrap();
    clock.set(20);
    // Refresh the *newer* key: a misfiring eviction kills "first".
    cache.put_global("second", b"2-new", 60_000).unwrap();
    assert_eq!(
        cache.get_global("first").unwrap(),
        Some(b"1".to_vec()),
        "refreshing 'second' evicted 'first'"
    );
    assert_eq!(cache.get_global("second").unwrap(), Some(b"2-new".to_vec()));
}

#[test]
fn slow_path_rereads_clock_global() {
    // REGRESSION: get_global reused the timestamp taken before the read
    // lock in its write-lock slow path. A clock jump between the two locks
    // made it report a live entry as expired. The slow path re-reads the
    // clock now; the scripted clock jumps backwards mid-call.
    let clock = ScriptedClock::arc(vec![0, 1_000, 500]);
    let dyn_clock: Arc<dyn Clock> = clock;
    let cache = test_cache("slow-clock-g", 60_000, dyn_clock);
    cache.put_global("k", b"v", 1_000).unwrap(); // stored_at = 0
    // Read lock sees t=1000 (expired); write lock must see t=500 (live).
    assert_eq!(cache.get_global("k").unwrap(), Some(b"v".to_vec()));
}

#[test]
fn slow_path_rereads_clock_local() {
    // Same hardening as the global tier: the slow path consults the clock
    // after acquiring the write lock, not a timestamp from before the read
    // lock. Pins the post-fix behavior (single-threaded old/new code made
    // the same number of clock calls, so the global test above is the
    // pre/post distinguisher; this one locks in the local tier's shape).
    let clock = ScriptedClock::arc(vec![0, 500]);
    let dyn_clock: Arc<dyn Clock> = clock;
    let mut config = CacheConfig::new(test_dir("slow-clock-l").join("c.bin"));
    config.local_ttl_ms = 1_000;
    let cache = TieredCache::new(config, dyn_clock).unwrap();
    cache.put_local("k", b"v").unwrap(); // stored_at = 0
    assert_eq!(cache.get_local("k").unwrap(), Some(b"v".to_vec()));
}

#[test]
fn wrong_magic_version_fails_closed() {
    let dir = test_dir("bad-version");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    cache.put_global("keep", b"me", 3_600_000).unwrap();
    let mut bad = b"PHLOWCACHEv2".to_vec();
    bad.extend_from_slice(&0u32.to_le_bytes());
    fs::write(dir.join("cache.bin"), &bad).unwrap();
    let err = cache.load().unwrap_err();
    assert!(matches!(err, CacheError::CorruptPersistedData { .. }));
    assert_eq!(cache.get_global("keep").unwrap(), Some(b"me".to_vec()));
}

#[test]
fn trailing_bytes_after_entries_rejected() {
    let dir = test_dir("trailing");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    cache.put_global("g", b"v", 60_000).unwrap();
    cache.persist().unwrap();
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(dir.join("cache.bin"))
        .unwrap();
    file.write_all(b"junk-bytes").unwrap();
    drop(file);
    assert!(matches!(
        cache.load().unwrap_err(),
        CacheError::CorruptPersistedData { .. }
    ));
}

#[test]
fn non_utf8_key_rejected() {
    let dir = test_dir("non-utf8");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    let mut buf = b"PHLOWCACHEv1".to_vec();
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&[0xff, 0xfe]);
    fs::write(dir.join("cache.bin"), &buf).unwrap();
    assert!(matches!(
        cache.load().unwrap_err(),
        CacheError::CorruptPersistedData { .. }
    ));
}

#[test]
fn duplicate_keys_in_file_last_wins_deterministically() {
    // Two entries, same key: the parser keeps the last one. Deterministic
    // and documented; a corrupt-file fuzzer must not see key duplication
    // as an error, but the behavior must be pinned.
    let dir = test_dir("dup-keys");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    fn entry(buf: &mut Vec<u8>, key: &[u8], value: &[u8]) {
        buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        buf.extend_from_slice(value);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&60_000u64.to_le_bytes());
    }
    let mut buf = b"PHLOWCACHEv1".to_vec();
    buf.extend_from_slice(&2u32.to_le_bytes());
    entry(&mut buf, b"dup", b"first");
    entry(&mut buf, b"dup", b"second");
    fs::write(dir.join("cache.bin"), &buf).unwrap();
    assert_eq!(cache.load().unwrap(), 1);
    assert_eq!(cache.get_global("dup").unwrap(), Some(b"second".to_vec()));
}

#[test]
fn zero_entry_file_loads_empty() {
    let dir = test_dir("zero-entries");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    let mut buf = b"PHLOWCACHEv1".to_vec();
    buf.extend_from_slice(&0u32.to_le_bytes());
    fs::write(dir.join("cache.bin"), &buf).unwrap();
    assert_eq!(cache.load().unwrap(), 0);
}

#[test]
fn load_replaces_the_live_tier() {
    let dir = test_dir("load-replace");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    cache.put_global("old", b"1", 60_000).unwrap();
    cache.persist().unwrap();
    cache.put_global("other", b"2", 60_000).unwrap();
    assert_eq!(cache.load().unwrap(), 1);
    assert_eq!(cache.get_global("old").unwrap(), Some(b"1".to_vec()));
    assert_eq!(
        cache.get_global("other").unwrap(),
        None,
        "load must swap the tier, not merge into it"
    );
}

#[test]
fn load_drops_expired_entries_without_resurrecting_them() {
    let dir = test_dir("load-expired");
    let clock = ManualClock::new(1_000);
    let dyn_clock: Arc<dyn Clock> = clock.clone();
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), dyn_clock).unwrap();
    let mut buf = b"PHLOWCACHEv1".to_vec();
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(b"k");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(b"v");
    buf.extend_from_slice(&0u64.to_le_bytes()); // stored_at
    buf.extend_from_slice(&100u64.to_le_bytes()); // ttl: dead long before t=1000
    fs::write(dir.join("cache.bin"), &buf).unwrap();
    assert_eq!(cache.load().unwrap(), 0);
    assert_eq!(cache.get_global("k").unwrap(), None);
}

#[test]
fn rebuild_panic_propagates_and_leaves_no_poisoned_lock() {
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = test_cache("rebuild-panic", 60_000, clock);
    let result = catch_unwind(AssertUnwindSafe(|| {
        cache.rebuild_local("k", &[b"a".to_vec()], |_| -> Vec<u8> {
            panic!("rebuild boom")
        })
    }));
    assert!(
        result.is_err(),
        "rebuild panic must propagate, not be swallowed"
    );
    // No lock was held during the rebuild, so nothing is poisoned.
    cache.put_local("k2", b"v").unwrap();
    assert_eq!(cache.get_local("k2").unwrap(), Some(b"v".to_vec()));
    assert_eq!(cache.get_local("k").unwrap(), None);
}

#[test]
fn rebuild_oversized_value_fails_closed() {
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = test_cache("rebuild-big", 60_000, clock);
    let err = cache
        .rebuild_local("big", &[], |_| vec![0u8; VALUE_BYTES_MAX + 1])
        .unwrap_err();
    assert!(matches!(err, CacheError::ValueTooLarge { .. }));
    assert_eq!(cache.get_local("big").unwrap(), None);
}

#[cfg(unix)]
#[test]
fn persist_never_writes_through_planted_tmp_symlink() {
    // REGRESSION: persist used fs::write on "<path>.tmp", which follows a
    // planted symlink and writes cache bytes to an attacker-chosen file.
    // The tmp file is now created with O_CREAT|O_EXCL (never follows a
    // link); a planted link is removed and retried instead of followed.
    use std::os::unix::fs::symlink;
    let dir = test_dir("symlink");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    cache.put_global("g", b"v", 60_000).unwrap();
    let target = dir.join("victim.txt");
    fs::write(&target, b"SENTINEL").unwrap();
    symlink(&target, dir.join("cache.tmp")).unwrap();
    assert_eq!(cache.persist().expect("persist must succeed"), 1);
    assert_eq!(
        fs::read(&target).unwrap(),
        b"SENTINEL",
        "persist wrote cache bytes through the planted symlink"
    );
    // The tmp file is renamed away on success, so the planted link is gone
    // with it; the victim was never touched.
    assert!(!dir.join("cache.tmp").exists());
}

#[cfg(unix)]
#[test]
fn persist_replaces_not_follows_path_symlink() {
    // rename(2) onto a symlink replaces the link itself: if persist_path is
    // a symlink, persist must swap in a regular file, never write cache
    // bytes through the link to its old target.
    use std::os::unix::fs::symlink;
    let dir = test_dir("path-symlink");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let link = dir.join("cache.bin");
    let target = dir.join("real.txt");
    fs::write(&target, b"SENTINEL").unwrap();
    symlink(&target, &link).unwrap();
    let cache = TieredCache::new(CacheConfig::new(link.clone()), clock).unwrap();
    cache.put_global("g", b"v", 60_000).unwrap();
    assert_eq!(cache.persist().unwrap(), 1);
    assert!(
        !fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink(),
        "persist must replace the symlink, not follow it"
    );
    assert_eq!(fs::read(&target).unwrap(), b"SENTINEL");
    assert_eq!(cache.load().unwrap(), 1);
}

#[test]
fn stale_tmp_file_does_not_brick_persist() {
    // A tmp left behind by a crashed persist is recovered (remove +
    // exclusive retry), not fatal, and never leaves a half-written cache.
    let dir = test_dir("stale-tmp");
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = TieredCache::new(CacheConfig::new(dir.join("cache.bin")), clock).unwrap();
    cache.put_global("g", b"v", 60_000).unwrap();
    fs::write(dir.join("cache.tmp"), b"half-written junk").unwrap();
    assert_eq!(cache.persist().expect("stale tmp must be recovered"), 1);
    assert!(
        !dir.join("cache.tmp").exists(),
        "tmp must be renamed away after persist"
    );
    assert_eq!(cache.load().unwrap(), 1);
}

#[test]
fn concurrent_persist_never_corrupts_the_file() {
    // Four threads persist the same path at once. Every outcome must be a
    // clean Ok/Err (never a panic, never a torn file): after one orderly
    // persist the file parses.
    let clock: Arc<dyn Clock> = ManualClock::new(0);
    let cache = Arc::new(
        TieredCache::new(
            CacheConfig::new(test_dir("race-persist").join("cache.bin")),
            clock,
        )
        .unwrap(),
    );
    for i in 0..50 {
        cache.put_global(&format!("k{i}"), b"v", 60_000).unwrap();
    }
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let cache = Arc::clone(&cache);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..10 {
                let _ = cache.persist();
            }
        }));
    }
    for handle in handles {
        handle.join().expect("persist worker panicked");
    }
    cache.persist().expect("final orderly persist");
    assert_eq!(
        cache.load().expect("file must parse after persist races"),
        50
    );
}

#[test]
fn concurrent_hammer_no_deadlock_no_panic() {
    // Eight threads hammer both tiers (the get_local read->write lock
    // upgrade path included). A watchdog turns a deadlock into a test
    // failure instead of a hung suite.
    let clock: Arc<dyn Clock> = ManualClock::new(1_000_000);
    let cache: Arc<TieredCache> = Arc::new(test_cache("hammer", 60_000, clock));
    let barrier = Arc::new(Barrier::new(8));
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    thread::spawn(move || {
        let mut handles = Vec::new();
        for worker in 0..8 {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                for round in 0..300 {
                    let local_key = format!("w{worker}-k{}", round % 16);
                    let global_key = format!("g{worker}-k{}", round % 16);
                    cache.put_local(&local_key, b"v").unwrap();
                    let _ = cache.get_local(&local_key).unwrap();
                    cache.put_global(&global_key, b"v", 60_000).unwrap();
                    let _ = cache.get_global(&global_key).unwrap();
                    // Shared keys maximize contention on the same entries.
                    cache.put_local("shared-local", b"s").unwrap();
                    let _ = cache.get_local("shared-local").unwrap();
                    let recent = vec![b"x".to_vec()];
                    cache
                        .rebuild_local("shared-rebuilt", &recent, |items| items.concat())
                        .unwrap();
                    let _ = cache.len_local().unwrap();
                    let _ = cache.len_global().unwrap();
                    if round % 50 == 0 {
                        cache.invalidate_local(&local_key).unwrap();
                        cache.invalidate_global(&global_key).unwrap();
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("hammer worker panicked");
        }
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(120))
        .expect("hammer hung: suspected deadlock");
}

// ---------------------------------------------------------------------------
// two_stage attacks
// ---------------------------------------------------------------------------

fn two_stage_config() -> TwoStageConfig {
    TwoStageConfig {
        block_size: 8,
        candidate_pool_size: 16,
        top_k: 4,
    }
}

#[test]
fn pool_membership_holds_in_release_builds_too() {
    // The fine-scorer-never-sees-out-of-pool guarantee is structural (the
    // scoring loop iterates the pool itself), so it must hold identically
    // in release builds where debug_assert is compiled out. Record every
    // index the scorer sees and check pool membership directly.
    let candidates: Vec<i32> = (0..64).collect();
    let seen = Mutex::new(Vec::new());
    let out = rank(
        &candidates,
        &two_stage_config(),
        |c| *c as f32,
        |c| {
            seen.lock().unwrap().push(*c);
            *c as f32
        },
    )
    .unwrap();
    assert_eq!(out.len(), 4);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 16);
    // Best two blocks by coarse max are 48..64.
    assert!(
        seen.iter().all(|c| (48..64).contains(c)),
        "fine scorer saw out-of-pool indices: {seen:?}"
    );
}

#[test]
fn one_million_candidates_rank_correctly() {
    // Stage 1 is linear; this is a perf sanity check (must finish promptly)
    // plus a correctness check at scale.
    let candidates: Vec<u32> = (0..1_000_000).collect();
    let cfg = TwoStageConfig {
        block_size: 64,
        candidate_pool_size: 1_024,
        top_k: 10,
    };
    let out = rank(&candidates, &cfg, |c| *c as f32, |c| *c as f32).unwrap();
    let expected: Vec<usize> = (999_990..1_000_000).rev().collect();
    assert_eq!(out, expected);
}

#[test]
fn ranking_is_deterministic_across_fifty_runs() {
    // Deterministic hash scores with occasional NaN: all 50 runs must agree
    // exactly, proving sort stability + index tie-breaking, not luck.
    fn hash_score(c: &u32) -> f32 {
        let mut x = (*c as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 32;
        if x.is_multiple_of(17) {
            return f32::NAN;
        }
        ((x >> 11) as f32) / ((1u64 << 53) as f32)
    }
    let candidates: Vec<u32> = (0..2_000).collect();
    let cfg = TwoStageConfig {
        block_size: 16,
        candidate_pool_size: 128,
        top_k: 20,
    };
    let first = rank(&candidates, &cfg, hash_score, hash_score).unwrap();
    for _ in 1..50 {
        assert_eq!(
            rank(&candidates, &cfg, hash_score, hash_score).unwrap(),
            first
        );
    }
}

#[test]
fn all_nan_coarse_scores_are_deterministic() {
    // Every block ties at -inf; stable sort must keep block order, so the
    // pool is the first blocks and the output is pinned.
    let candidates: Vec<i32> = (0..32).collect();
    let cfg = TwoStageConfig {
        block_size: 8,
        candidate_pool_size: 8,
        top_k: 8,
    };
    let first = rank(&candidates, &cfg, |_| f32::NAN, |c| *c as f32).unwrap();
    let second = rank(&candidates, &cfg, |_| f32::NAN, |c| *c as f32).unwrap();
    assert_eq!(first, second);
    // Pool is the first block (0..8); fine scores rank best-first.
    assert_eq!(first, (0..8).rev().collect::<Vec<_>>());
}

#[test]
fn fine_scorer_panic_propagates() {
    // rank is pure (no locks, no global state): a panicking scorer unwinds
    // to the caller; nothing is swallowed or left half-mutated.
    let candidates: Vec<i32> = (0..16).collect();
    let result = catch_unwind(AssertUnwindSafe(|| {
        rank(
            &candidates,
            &two_stage_config(),
            |c| *c as f32,
            |c| {
                if *c == 5 {
                    panic!("scorer boom")
                }
                *c as f32
            },
        )
    }));
    assert!(result.is_err(), "scorer panic must propagate");
}

#[test]
fn huge_block_size_does_not_overflow() {
    let candidates: Vec<i32> = (0..10).collect();
    let cfg = TwoStageConfig {
        block_size: usize::MAX,
        candidate_pool_size: 4,
        top_k: 3,
    };
    let out = rank(&candidates, &cfg, |c| *c as f32, |c| *c as f32).unwrap();
    assert_eq!(out, vec![9, 8, 7]);
}

#[test]
fn top_k_equal_to_pool_size_is_allowed() {
    let candidates: Vec<i32> = (0..8).collect();
    let cfg = TwoStageConfig {
        block_size: 4,
        candidate_pool_size: 8,
        top_k: 8,
    };
    let out = rank(&candidates, &cfg, |c| *c as f32, |c| *c as f32).unwrap();
    assert_eq!(out, vec![7, 6, 5, 4, 3, 2, 1, 0]);
}

#[test]
fn infinite_scores_sort_at_extremes() {
    let candidates: Vec<i32> = (0..8).collect();
    let cfg = TwoStageConfig {
        block_size: 4,
        candidate_pool_size: 8,
        top_k: 8,
    };
    let out = rank(
        &candidates,
        &cfg,
        |c| if *c == 0 { f32::INFINITY } else { *c as f32 },
        |c| {
            if *c == 1 {
                f32::NEG_INFINITY
            } else if *c == 2 {
                f32::INFINITY
            } else {
                *c as f32
            }
        },
    )
    .unwrap();
    assert_eq!(out[0], 2, "+inf fine score must rank first");
    assert_eq!(out[7], 1, "-inf fine score must rank last");
}

// ---------------------------------------------------------------------------
// speculative attacks
// ---------------------------------------------------------------------------

fn single_point_table(cost: f32) -> ThroughputTable {
    ThroughputTable::new(vec![ThroughputPoint {
        load: 0.0,
        tokens_per_sec: 100.0,
        verify_cost_per_token: cost,
    }])
    .unwrap()
}

#[test]
fn nan_confidence_fails_before_verify_runs() {
    // A NaN confidence must fail closed during the confidence pass; the
    // verifier must never see a submission derived from bad confidences.
    let verified = Mutex::new(false);
    let err = draft_verify(
        || vec![1, 2, 3],
        |_| {
            *verified.lock().unwrap() = true;
            vec![true; 3]
        },
        |i| if i == 0 { f32::NAN } else { 0.5 },
        0.0,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        SpecError::InvalidConfidence { index: 0, value } if value.is_nan()
    ));
    assert!(
        !*verified.lock().unwrap(),
        "verifier ran despite invalid confidence"
    );
}

#[test]
fn verifier_panic_propagates() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        draft_verify(
            || vec![1, 2, 3],
            |_| -> Vec<bool> { panic!("verifier down") },
            |_| 1.0,
            0.0,
        )
    }));
    assert!(result.is_err(), "verifier panic must propagate");
}

#[test]
fn drafter_panic_propagates_before_any_work() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        draft_verify(
            || -> Vec<u8> { panic!("drafter down") },
            |_| vec![],
            |_| 1.0,
            0.0,
        )
    }));
    assert!(result.is_err(), "drafter panic must propagate");
}

#[test]
fn duplicate_load_points_resolve_deterministically() {
    // Duplicate loads: stable sort keeps insertion order, stepwise scan
    // takes the last point at or below the load. Pinned, not accidental.
    let table = ThroughputTable::new(vec![
        ThroughputPoint {
            load: 5.0,
            tokens_per_sec: 100.0,
            verify_cost_per_token: 0.5,
        },
        ThroughputPoint {
            load: 5.0,
            tokens_per_sec: 50.0,
            verify_cost_per_token: 0.9,
        },
    ])
    .unwrap();
    assert_eq!(table.throughput_at(5.0).unwrap().tokens_per_sec, 50.0);
}

#[test]
fn non_finite_table_rows_rejected() {
    let bad_points = [
        ThroughputPoint {
            load: f32::NAN,
            tokens_per_sec: 100.0,
            verify_cost_per_token: 0.5,
        },
        ThroughputPoint {
            load: 0.0,
            tokens_per_sec: f32::NAN,
            verify_cost_per_token: 0.5,
        },
        ThroughputPoint {
            load: 0.0,
            tokens_per_sec: 100.0,
            verify_cost_per_token: f32::NAN,
        },
        ThroughputPoint {
            load: 0.0,
            tokens_per_sec: f32::INFINITY,
            verify_cost_per_token: 0.5,
        },
        ThroughputPoint {
            load: -1.0,
            tokens_per_sec: 100.0,
            verify_cost_per_token: 0.5,
        },
        ThroughputPoint {
            load: 0.0,
            tokens_per_sec: 0.0,
            verify_cost_per_token: 0.5,
        },
        ThroughputPoint {
            load: 0.0,
            tokens_per_sec: 100.0,
            verify_cost_per_token: 0.0,
        },
        ThroughputPoint {
            load: 0.0,
            tokens_per_sec: 100.0,
            verify_cost_per_token: 1.5,
        },
    ];
    for (index, point) in bad_points.into_iter().enumerate() {
        assert!(
            matches!(
                ThroughputTable::new(vec![point]).unwrap_err(),
                SpecError::InvalidThroughputPoint { .. }
            ),
            "bad point {index} was accepted"
        );
    }
}

#[test]
fn schedule_rejects_nan_survival() {
    // REGRESSION: schedule_for_load accepted a NaN survival entry and
    // silently returned a plan computed from garbage (NaN comparisons never
    // beat the running best). Non-probability curve entries are now a typed
    // error.
    let table = single_point_table(0.5);
    let err = schedule_for_load(&table, 0.0, &[0.9, f32::NAN]).unwrap_err();
    assert!(matches!(
        err,
        SpecError::InvalidSurvival { index: 1, value } if value.is_nan()
    ));
}

#[test]
fn schedule_rejects_out_of_range_survival() {
    // A negative entry used to produce a plan with negative
    // expected_accepted — nonsense output for typed-valid input.
    let table = single_point_table(0.5);
    assert!(matches!(
        schedule_for_load(&table, 0.0, &[0.9, -0.1]).unwrap_err(),
        SpecError::InvalidSurvival { index: 1, .. }
    ));
    assert!(matches!(
        schedule_for_load(&table, 0.0, &[0.9, 1.5]).unwrap_err(),
        SpecError::InvalidSurvival { index: 1, .. }
    ));
}

#[test]
fn schedule_tie_prefers_shorter_verify() {
    // k=1: 0.8/2 = 0.4; k=2: 1.2/3 = 0.4 — exact tie, shorter must win
    // (less engine time for the same expectation).
    let table = single_point_table(1.0);
    let plan = schedule_for_load(&table, 0.0, &[0.8, 0.4]).unwrap();
    assert_eq!(plan.verify_len, 1);
    assert!((plan.expected_accepted - 0.8).abs() < 1e-6);
}

#[test]
fn threshold_one_verifies_only_certain_prefix() {
    // Survival 0.99 < 1.0: nothing qualifies, verifier gets an empty slice.
    let out = draft_verify(
        || vec![7, 8],
        |submitted| {
            assert!(submitted.is_empty());
            vec![]
        },
        |_| 0.99,
        1.0,
    )
    .unwrap();
    assert_eq!(out.verified_len, 0);
    assert!(out.accepted.is_empty());
    assert_eq!(out.draft_len, 2);
}

#[test]
fn draft_count_bound_holds_for_huge_tokens() {
    // The bound is on token *count*: 64 one-megabyte tokens are accepted.
    // Per-token byte budget is documented as the caller's responsibility —
    // the module cannot bound bytes of a generic T.
    let big = "x".repeat(1024 * 1024);
    let out = draft_verify(
        || vec![big.clone(); DRAFT_TOKEN_MAX],
        |submitted| vec![true; submitted.len()],
        |_| 1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(out.draft_len, DRAFT_TOKEN_MAX);
    assert_eq!(out.verified_len, DRAFT_TOKEN_MAX);
    assert_eq!(out.accepted.len(), DRAFT_TOKEN_MAX);
}

// ---------------------------------------------------------------------------
// kv_policy attacks
// ---------------------------------------------------------------------------

#[test]
fn validate_truth_table_global_main() {
    // Exhaustive (precision, qat, after_rope) truth table for GlobalMain.
    // A "contradictory" config (QAT and not-QAT at once) is unrepresentable:
    // the fields are single bools, so the closest attack is enumerating
    // every combination and pinning the exact violation set and order.
    struct Case {
        precision: Precision,
        qat: bool,
        after_rope: bool,
        expected_rules: Vec<&'static str>,
    }
    let cases = [
        Case {
            precision: Precision::Fp4,
            qat: true,
            after_rope: true,
            expected_rules: vec![],
        },
        Case {
            precision: Precision::Fp4,
            qat: true,
            after_rope: false,
            expected_rules: vec!["quantize_after_rope"],
        },
        Case {
            precision: Precision::Fp4,
            qat: false,
            after_rope: true,
            expected_rules: vec!["fp4_requires_qat"],
        },
        Case {
            precision: Precision::Fp4,
            qat: false,
            after_rope: false,
            expected_rules: vec!["fp4_requires_qat", "quantize_after_rope"],
        },
        Case {
            precision: Precision::Fp8,
            qat: true,
            after_rope: true,
            expected_rules: vec![],
        },
        Case {
            precision: Precision::Fp8,
            qat: true,
            after_rope: false,
            expected_rules: vec!["quantize_after_rope"],
        },
        Case {
            precision: Precision::Fp8,
            qat: false,
            after_rope: true,
            expected_rules: vec![],
        },
        Case {
            precision: Precision::Fp8,
            qat: false,
            after_rope: false,
            expected_rules: vec!["quantize_after_rope"],
        },
    ];
    for case in cases {
        let policy = QuantPolicy {
            component: KvComponent::GlobalMain,
            precision: case.precision,
            quantize_after_rope: case.after_rope,
            trained_with_qat: case.qat,
        };
        let rules: Vec<&str> = validate(&policy).iter().map(|v| v.rule).collect();
        assert_eq!(
            rules, case.expected_rules,
            "precision={} qat={} after_rope={}",
            case.precision, case.qat, case.after_rope
        );
    }
    // SWA + FP4 + QAT: the SWA rule fires alone (QAT does not excuse it).
    let swa = QuantPolicy {
        component: KvComponent::LocalSwa,
        precision: Precision::Fp4,
        quantize_after_rope: true,
        trained_with_qat: true,
    };
    let rules: Vec<&str> = validate(&swa).iter().map(|v| v.rule).collect();
    assert_eq!(rules, vec!["swa_keeps_fp8_or_higher"]);
}

#[test]
fn recommend_covers_every_component_with_clean_policy() {
    // Exhaustiveness is compiler-enforced (match on KvComponent), but pin
    // the runtime contract too: every variant recommends, and every
    // recommendation validates clean.
    for component in [
        KvComponent::GlobalMain,
        KvComponent::LocalSwa,
        KvComponent::Indexer,
    ] {
        let policy = recommend(component);
        assert_eq!(policy.component, component);
        assert!(
            validate(&policy).is_empty(),
            "recommendation for {component} must validate clean"
        );
    }
}

#[test]
fn bytes_per_token_stays_finite_at_u32_max() {
    let policy = recommend(KvComponent::GlobalMain); // Fp4
    let bytes = bytes_per_token(&policy, u32::MAX);
    assert!(bytes.is_finite() && bytes > 0.0);
    let fp32 = QuantPolicy {
        component: KvComponent::GlobalMain,
        precision: Precision::Fp32,
        quantize_after_rope: true,
        trained_with_qat: false,
    };
    assert!(bytes_per_token(&fp32, u32::MAX).is_finite());
}
