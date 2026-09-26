//! Coarse-to-fine candidate ranking.
//!
//! Plain words: scoring every candidate with the expensive scorer is
//! wasteful. Instead, a cheap scorer ranks *blocks* of candidates, the best
//! blocks form a bounded candidate pool, and the expensive scorer only ever
//! runs on pool items. This is the serving idea behind the DeepSeek-V4.1-Flash
//! Hierarchical Sparse Indexer (blockwise max-score selection into a shared
//! candidate pool, then top-K within the pool) — minus the training, which
//! made the paper's version quality-neutral. Here the contract is purely
//! computational: the fine scorer is guaranteed to see only pool members.
//!
//! # Bounds
//!
//! - `block_size >= 1`, `candidate_pool_size >= 1`, `top_k >= 1` (validated).
//! - The pool holds whole blocks, so its size is bounded by
//!   `candidate_pool_size + block_size - 1` (documented, not silent).
//! - `top_k` must not exceed the pool size; otherwise ranking would invent
//!   results beyond what was finely scored.
//!
//! # Score policy
//!
//! A `NaN` score is treated as negative infinity: a scorer that fails to
//! produce a number never outranks one that did. Ties break by original
//! candidate index, so output order is deterministic.

use std::fmt;

/// Configuration for [`rank`].
#[derive(Debug, Clone, Copy)]
pub struct TwoStageConfig {
    /// Candidates per coarse block. Block score is the max member score.
    pub block_size: usize,
    /// Target candidate-pool size. The pool holds whole blocks, so its
    /// actual size is at most `candidate_pool_size + block_size - 1`.
    pub candidate_pool_size: usize,
    /// How many of the pool's best, by fine score, to return.
    pub top_k: usize,
}

impl TwoStageConfig {
    fn validate(&self) -> Result<(), RankError> {
        if self.block_size == 0 {
            return Err(RankError::InvalidConfig {
                field: "block_size",
            });
        }
        if self.candidate_pool_size == 0 {
            return Err(RankError::InvalidConfig {
                field: "candidate_pool_size",
            });
        }
        if self.top_k == 0 {
            return Err(RankError::InvalidConfig { field: "top_k" });
        }
        Ok(())
    }
}

/// Failures of [`rank`]. All are caller errors, not internal faults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankError {
    /// A config field was zero.
    InvalidConfig {
        /// Which field was invalid.
        field: &'static str,
    },
    /// `top_k` exceeded the number of finely scored pool items.
    TopKExceedsPool {
        /// Requested `top_k`.
        top_k: usize,
        /// Actual pool size.
        pool_len: usize,
    },
}

impl fmt::Display for RankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RankError::InvalidConfig { field } => {
                write!(
                    f,
                    "two-stage rank: invalid config field `{field}` (must be >= 1)"
                )
            }
            RankError::TopKExceedsPool { top_k, pool_len } => {
                write!(
                    f,
                    "two-stage rank: top_k {top_k} exceeds finely scored pool of {pool_len}"
                )
            }
        }
    }
}

impl std::error::Error for RankError {}

/// NaN sorts as negative infinity: unscored never outranks scored.
fn sanitize(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// Rank candidates coarse-to-fine.
///
/// Returns the indices of the `top_k` best candidates in rank order (best
/// first), where "best" is decided by `fine_score` restricted to the
/// candidate pool. `coarse_score` runs once per candidate; `fine_score`
/// runs once per pool member and never on an out-of-pool candidate — the
/// scoring loop iterates the pool itself, so the guarantee is structural
/// and holds in release builds too (covered by tests in both modes).
///
/// An empty candidate list yields an empty ranking, even with `top_k >= 1`.
/// Total work is O(`candidates.len()`) coarse scores plus O(pool) fine
/// scores, with the pool bounded as documented on [`TwoStageConfig`].
///
/// # Errors
///
/// - [`RankError::InvalidConfig`] on a zero config field.
/// - [`RankError::TopKExceedsPool`] when `top_k` exceeds the pool size.
pub fn rank<T, CS, FS>(
    candidates: &[T],
    config: &TwoStageConfig,
    coarse_score: CS,
    fine_score: FS,
) -> Result<Vec<usize>, RankError>
where
    CS: Fn(&T) -> f32,
    FS: Fn(&T) -> f32,
{
    config.validate()?;
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Stage 1: cheap score for every candidate.
    let coarse: Vec<f32> = candidates
        .iter()
        .map(|c| sanitize(coarse_score(c)))
        .collect();

    // Group into blocks; each block's score is its members' max.
    let block_count = candidates.len().div_ceil(config.block_size);
    let block_score = |block: usize| -> f32 {
        let start = block * config.block_size;
        let end = (start + config.block_size).min(candidates.len());
        coarse[start..end]
            .iter()
            .fold(f32::NEG_INFINITY, |best, &s| f32::max(best, s))
    };
    let mut block_order: Vec<usize> = (0..block_count).collect();
    block_order.sort_by(|&a, &b| block_score(b).total_cmp(&block_score(a)));

    // Stage 2: candidate pool from the best whole blocks.
    let mut pool: Vec<usize> = Vec::new();
    for &block in &block_order {
        if pool.len() >= config.candidate_pool_size {
            break;
        }
        let start = block * config.block_size;
        let end = (start + config.block_size).min(candidates.len());
        pool.extend(start..end);
    }
    if config.top_k > pool.len() {
        return Err(RankError::TopKExceedsPool {
            top_k: config.top_k,
            pool_len: pool.len(),
        });
    }

    // Stage 3: expensive score, pool members only. Membership is
    // structural — the loop below iterates `pool` itself — so the guarantee
    // holds identically in debug and release builds. The debug assertions
    // check the pool's own invariants (range, uniqueness) instead of a
    // tautological self-membership test.
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            pool.iter().all(|&i| i < candidates.len()),
            "pool index out of candidate range"
        );
        let mut ordered = pool.clone();
        ordered.sort_unstable();
        ordered.dedup();
        debug_assert_eq!(ordered.len(), pool.len(), "pool holds a duplicate index");
    }
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(pool.len());
    for &idx in &pool {
        scored.push((idx, sanitize(fine_score(&candidates[idx]))));
    }
    // Best fine score first; ties resolve by original index for determinism.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(scored
        .into_iter()
        .take(config.top_k)
        .map(|(idx, _)| idx)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn config() -> TwoStageConfig {
        TwoStageConfig {
            block_size: 8,
            candidate_pool_size: 32,
            top_k: 5,
        }
    }

    /// Cheap scorer that agrees with the fine scorer: in a real deployment
    /// the coarse pass is a cheap approximation of the fine one.
    fn peak_score(c: &i32) -> f32 {
        const PEAKS: [usize; 6] = [7, 42, 90, 13, 55, 3];
        if PEAKS.contains(&(*c as usize)) {
            1000.0 - *c as f32
        } else {
            *c as f32
        }
    }

    #[test]
    fn rank_returns_top_k_by_fine_score() {
        // 100 candidates; fine score peaks at known indices. The pool takes
        // the 4 best blocks of 8 (blocks 0, 1, 5, 6 hold five of the peaks).
        let candidates: Vec<i32> = (0..100).collect();
        let out = rank(&candidates, &config(), peak_score, peak_score).unwrap();
        assert_eq!(out.len(), 5);
        // Best fine scores first: 1000-3 > 1000-7 > 1000-13 > 1000-42 > 1000-55.
        assert_eq!(out, vec![3, 7, 13, 42, 55]);
    }

    #[test]
    fn fine_scorer_never_sees_out_of_pool_items() {
        let candidates: Vec<i32> = (0..64).collect();
        let cfg = TwoStageConfig {
            block_size: 8,
            candidate_pool_size: 16,
            top_k: 4,
        };
        let seen = RefCell::new(Vec::new());
        let out = rank(
            &candidates,
            &cfg,
            |c| *c as f32,
            |c| {
                seen.borrow_mut().push(*c);
                *c as f32
            },
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        // Pool = two best blocks by coarse max = indices 48..64.
        let seen = seen.borrow();
        assert_eq!(seen.len(), 16);
        assert!(seen.iter().all(|c| (48..64).contains(c)));
    }

    #[test]
    fn empty_candidates_yield_empty_ranking() {
        let out: Vec<usize> =
            rank(&Vec::<i32>::new(), &config(), |c| *c as f32, |c| *c as f32).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn top_k_beyond_pool_is_an_error() {
        // block_size 2 divides the pool target exactly, so the pool holds
        // precisely 8 candidates and top_k 9 is rejected.
        let candidates: Vec<i32> = (0..10).collect();
        let cfg = TwoStageConfig {
            block_size: 2,
            candidate_pool_size: 8,
            top_k: 9,
        };
        let err = rank(&candidates, &cfg, |c| *c as f32, |c| *c as f32).unwrap_err();
        assert_eq!(
            err,
            RankError::TopKExceedsPool {
                top_k: 9,
                pool_len: 8
            }
        );
    }

    #[test]
    fn invalid_configs_rejected() {
        let candidates = vec![1, 2, 3];
        for bad in [
            TwoStageConfig {
                block_size: 0,
                candidate_pool_size: 8,
                top_k: 1,
            },
            TwoStageConfig {
                block_size: 4,
                candidate_pool_size: 0,
                top_k: 1,
            },
            TwoStageConfig {
                block_size: 4,
                candidate_pool_size: 8,
                top_k: 0,
            },
        ] {
            assert!(matches!(
                rank(&candidates, &bad, |c| *c as f32, |c| *c as f32).unwrap_err(),
                RankError::InvalidConfig { .. }
            ));
        }
    }

    #[test]
    fn pool_larger_than_candidates_uses_all() {
        let candidates: Vec<i32> = (0..5).collect();
        let cfg = TwoStageConfig {
            block_size: 8,
            candidate_pool_size: 100,
            top_k: 5,
        };
        let out = rank(&candidates, &cfg, |c| *c as f32, |c| -(*c as f32)).unwrap();
        assert_eq!(out, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn nan_coarse_score_never_wins_a_block() {
        // Candidate 0 has NaN coarse score; it must not beat real scores.
        let candidates: Vec<i32> = (0..16).collect();
        let cfg = TwoStageConfig {
            block_size: 8,
            candidate_pool_size: 8,
            top_k: 8,
        };
        let out = rank(
            &candidates,
            &cfg,
            |c| if *c == 0 { f32::NAN } else { *c as f32 },
            |c| *c as f32,
        )
        .unwrap();
        // Pool holds exactly one block of 8: the block 8..16 (max 15 beats NaN-block's max 7).
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&i| i >= 8));
    }

    #[test]
    fn nan_fine_score_ranks_last() {
        let candidates: Vec<i32> = (0..8).collect();
        let cfg = TwoStageConfig {
            block_size: 8,
            candidate_pool_size: 8,
            top_k: 3,
        };
        let out = rank(
            &candidates,
            &cfg,
            |c| *c as f32,
            |c| if *c == 7 { f32::NAN } else { *c as f32 },
        )
        .unwrap();
        assert_eq!(out, vec![6, 5, 4]);
    }

    #[test]
    fn ties_break_by_index_deterministically() {
        let candidates = vec!["a", "b", "c", "d"];
        let cfg = TwoStageConfig {
            block_size: 2,
            candidate_pool_size: 4,
            top_k: 4,
        };
        let first = rank(&candidates, &cfg, |_| 1.0, |_| 1.0).unwrap();
        let second = rank(&candidates, &cfg, |_| 1.0, |_| 1.0).unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec![0, 1, 2, 3]);
    }

    #[test]
    fn block_size_one_pool_is_exact() {
        let candidates: Vec<i32> = (0..10).collect();
        let cfg = TwoStageConfig {
            block_size: 1,
            candidate_pool_size: 3,
            top_k: 3,
        };
        let out = rank(&candidates, &cfg, |c| *c as f32, |c| *c as f32).unwrap();
        assert_eq!(out, vec![9, 8, 7]);
    }
}
