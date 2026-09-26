//! Draft-then-verify orchestration with confidence scheduling.
//!
//! Plain words: instead of decoding one token per forward pass, a cheap
//! drafter proposes several tokens at once and the main engine verifies
//! them in a single parallel pass, accepting the longest correct prefix.
//! A confidence head predicts each draft position's acceptance probability;
//! the verify length is scheduled from those predictions plus a profiled
//! throughput table. This is the serving idea behind DSpark (semi-
//! autoregressive drafting with confidence-scheduled verification) from the
//! DeepSeek-V4.1-Flash paper — without the trained drafter, which we do
//! not have. The caller supplies all three closures, so any draft source
//! (n-gram table, small model, heuristics) plugs in.
//!
//! # Cost model
//!
//! Verifying `k` tokens costs `VERIFY_FIXED_COST_SLOTS + k * c` engine
//! token-slots, where `c = verify_cost_per_token` comes from the profiled
//! table: parallel verification is sublinear (`c < 1`) when the engine is
//! idle and approaches 1 slot per token under saturation. The scheduler
//! picks the length maximizing expected accepted tokens per slot. Because
//! `c` rises with load, the schedule shortens under saturation — shedding
//! speculative work exactly when the engine can least afford it.
//!
//! # Sync only
//!
//! All closures are synchronous. Wiring this into an async runtime (tokio)
//! is future work; the orchestration logic itself does not depend on one.

use std::fmt;

/// Maximum draft tokens per round. Bounds drafter output and the verify
/// submission alike.
pub const DRAFT_TOKEN_MAX: usize = 64;
/// Maximum tokens submitted to one verify call.
pub const VERIFY_LEN_MAX: usize = 64;
/// Maximum entries in a [`ThroughputTable`]. Keeps table scans bounded.
pub const THROUGHPUT_TABLE_ENTRY_MAX: usize = 256;
/// Fixed scheduling overhead of one verify call, in token-slots (kernel
/// launch + scheduling around the parallel pass).
pub const VERIFY_FIXED_COST_SLOTS: f32 = 1.0;

/// Outcome of one [`draft_verify`] round.
#[derive(Debug, Clone)]
pub struct Verified<T> {
    /// Longest accepted draft prefix (may be empty).
    pub accepted: Vec<T>,
    /// How many tokens the drafter proposed.
    pub draft_len: usize,
    /// How many tokens were submitted for verification.
    pub verified_len: usize,
}

/// One profiled operating point of the serving engine.
#[derive(Debug, Clone, Copy)]
pub struct ThroughputPoint {
    /// Offered load at which this point was profiled (>= 0, finite).
    pub load: f32,
    /// Sustained engine throughput at this load, tokens/sec (> 0).
    pub tokens_per_sec: f32,
    /// Profiled engine cost per verified token, in decode-step fractions.
    /// Sublinear (< 1) when verification parallelizes well; approaches 1
    /// under saturation. Must lie in (0, 1].
    pub verify_cost_per_token: f32,
}

/// Profiled throughput curve, sorted by ascending load. Drives
/// [`schedule_for_load`].
#[derive(Debug, Clone)]
pub struct ThroughputTable {
    points: Vec<ThroughputPoint>,
}

impl ThroughputTable {
    /// Build a table from profiled points. Points are sorted by load;
    /// lookup is stepwise (last point at or below the queried load).
    ///
    /// # Errors
    ///
    /// - [`SpecError::EmptyThroughputTable`] when no points are given.
    /// - [`SpecError::InvalidThroughputPoint`] on non-finite, negative, or
    ///   out-of-range fields, or more than [`THROUGHPUT_TABLE_ENTRY_MAX`]
    ///   points.
    pub fn new(mut points: Vec<ThroughputPoint>) -> Result<Self, SpecError> {
        if points.is_empty() {
            return Err(SpecError::EmptyThroughputTable);
        }
        if points.len() > THROUGHPUT_TABLE_ENTRY_MAX {
            return Err(SpecError::TooManyThroughputPoints { len: points.len() });
        }
        for (index, point) in points.iter().enumerate() {
            if !point.load.is_finite() || point.load < 0.0 {
                return Err(SpecError::InvalidThroughputPoint { index });
            }
            if !point.tokens_per_sec.is_finite() || point.tokens_per_sec <= 0.0 {
                return Err(SpecError::InvalidThroughputPoint { index });
            }
            if !point.verify_cost_per_token.is_finite()
                || point.verify_cost_per_token <= 0.0
                || point.verify_cost_per_token > 1.0
            {
                return Err(SpecError::InvalidThroughputPoint { index });
            }
        }
        points.sort_by(|a, b| a.load.total_cmp(&b.load));
        Ok(ThroughputTable { points })
    }

    /// Stepwise lookup: the last point with `load <= current_load`, or the
    /// first point when the load is below every profiled point.
    pub fn throughput_at(&self, current_load: f32) -> Result<ThroughputPoint, SpecError> {
        if !current_load.is_finite() || current_load < 0.0 {
            return Err(SpecError::InvalidLoad {
                value: current_load,
            });
        }
        let mut best = self.points[0];
        for point in &self.points {
            if point.load <= current_load {
                best = *point;
            } else {
                break;
            }
        }
        Ok(best)
    }
}

/// Plan produced by [`schedule_for_load`].
#[derive(Debug, Clone, Copy)]
pub struct VerifyPlan {
    /// Tokens to submit for verification (0 = skip verification this round).
    pub verify_len: usize,
    /// Expected accepted tokens at that length.
    pub expected_accepted: f32,
    /// Expected accepted tokens per engine token-slot at that length.
    pub expected_goodput_slots: f32,
    /// Profiled engine throughput at the queried load, for the caller's
    /// own admission accounting.
    pub profiled_tokens_per_sec: f32,
}

/// Everything that can go wrong in draft-verify orchestration.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecError {
    /// Acceptance threshold outside [0, 1] or non-finite.
    InvalidThreshold {
        /// Offending value.
        value: f32,
    },
    /// Drafter proposed more than [`DRAFT_TOKEN_MAX`] tokens.
    DraftTooLarge {
        /// Actual draft length.
        len: usize,
    },
    /// Confidence head returned a non-probability.
    InvalidConfidence {
        /// Draft position.
        index: usize,
        /// Offending value.
        value: f32,
    },
    /// Load for table lookup was negative or non-finite.
    InvalidLoad {
        /// Offending value.
        value: f32,
    },
    /// No profiled points supplied.
    EmptyThroughputTable,
    /// Too many profiled points.
    TooManyThroughputPoints {
        /// Actual count.
        len: usize,
    },
    /// A profiled point failed field validation.
    InvalidThroughputPoint {
        /// Index of the bad point.
        index: usize,
    },
    /// A survival-curve entry was not a finite probability in [0, 1].
    InvalidSurvival {
        /// Curve position.
        index: usize,
        /// Offending value.
        value: f32,
    },
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecError::InvalidThreshold { value } => {
                write!(f, "speculative: threshold {value} outside [0, 1]")
            }
            SpecError::DraftTooLarge { len } => {
                write!(f, "speculative: draft of {len} exceeds {DRAFT_TOKEN_MAX}")
            }
            SpecError::InvalidConfidence { index, value } => {
                write!(
                    f,
                    "speculative: confidence {value} at position {index} is not a probability"
                )
            }
            SpecError::InvalidLoad { value } => {
                write!(f, "speculative: load {value} invalid for throughput lookup")
            }
            SpecError::EmptyThroughputTable => {
                write!(f, "speculative: throughput table needs at least one point")
            }
            SpecError::TooManyThroughputPoints { len } => {
                write!(
                    f,
                    "speculative: {len} throughput points exceed {THROUGHPUT_TABLE_ENTRY_MAX}"
                )
            }
            SpecError::InvalidThroughputPoint { index } => {
                write!(f, "speculative: throughput point {index} failed validation")
            }
            SpecError::InvalidSurvival { index, value } => {
                write!(
                    f,
                    "speculative: survival {value} at position {index} \
                     is not a probability in [0, 1]"
                )
            }
        }
    }
}

impl std::error::Error for SpecError {}

/// Prefix survival curve from per-position conditional acceptance
/// probabilities: `survival[k]` is P(first `k+1` drafts all accepted).
///
/// Each confidence must be a finite probability in [0, 1]; anything else is
/// a caller bug and fails closed.
pub fn survival_curve(confidence: &[f32]) -> Result<Vec<f32>, SpecError> {
    let mut survival = Vec::with_capacity(confidence.len());
    let mut running = 1.0f32;
    for (index, &c) in confidence.iter().enumerate() {
        if !c.is_finite() || c < 0.0 || c > 1.0 {
            return Err(SpecError::InvalidConfidence { index, value: c });
        }
        running *= c;
        survival.push(running);
    }
    Ok(survival)
}

/// Run one draft-then-verify round.
///
/// 1. `draft_fn` proposes up to [`DRAFT_TOKEN_MAX`] tokens.
/// 2. `confidence_fn(i)` predicts P(token `i` accepted | earlier accepted);
///    the verify length is the longest prefix whose survival probability
///    stays at or above `threshold` (confidence-scheduled acceptance).
/// 3. `verify_fn` judges the submitted prefix; the longest all-accepted
///    prefix is returned. Judgement stops at the first rejection, and at
///    the end of `verify_fn`'s answer if it is short.
///
/// `threshold` must lie in [0, 1]. A threshold of 0 verifies the whole
/// draft; a threshold of 1 verifies only while survival is certain.
///
/// # Errors
///
/// - [`SpecError::InvalidThreshold`] on a bad threshold.
/// - [`SpecError::DraftTooLarge`] when the drafter exceeds its bound.
/// - [`SpecError::InvalidConfidence`] on a non-probability confidence.
pub fn draft_verify<T, D, V, C>(
    draft_fn: D,
    verify_fn: V,
    confidence_fn: C,
    threshold: f32,
) -> Result<Verified<T>, SpecError>
where
    T: Clone,
    D: FnOnce() -> Vec<T>,
    V: FnOnce(&[T]) -> Vec<bool>,
    C: Fn(usize) -> f32,
{
    if !threshold.is_finite() || threshold < 0.0 || threshold > 1.0 {
        return Err(SpecError::InvalidThreshold { value: threshold });
    }
    let drafts = draft_fn();
    if drafts.len() > DRAFT_TOKEN_MAX {
        return Err(SpecError::DraftTooLarge { len: drafts.len() });
    }
    if drafts.is_empty() {
        return Ok(Verified {
            accepted: Vec::new(),
            draft_len: 0,
            verified_len: 0,
        });
    }

    let confidence: Vec<f32> = (0..drafts.len()).map(&confidence_fn).collect();
    let survival = survival_curve(&confidence)?;
    // Survival is non-increasing (products of probabilities), so the
    // qualifying set is a prefix; its length is the verify length.
    let mut verify_len = 0;
    for &s in &survival {
        if s >= threshold {
            verify_len += 1;
        } else {
            break;
        }
    }
    verify_len = verify_len.min(VERIFY_LEN_MAX);

    let decisions = verify_fn(&drafts[..verify_len]);
    let mut accepted = Vec::new();
    for (token, &ok) in drafts[..verify_len].iter().zip(decisions.iter()) {
        if !ok {
            break;
        }
        accepted.push(token.clone());
    }
    Ok(Verified {
        accepted,
        draft_len: drafts.len(),
        verified_len: verify_len,
    })
}

/// Choose the verify length maximizing expected accepted tokens per engine
/// token-slot at the current load.
///
/// `survival` is the prefix survival curve (see [`survival_curve`]).
/// Expected accepted tokens for length `k` is the curve's partial sum;
/// engine cost is `VERIFY_FIXED_COST_SLOTS + k * verify_cost_per_token`
/// token-slots, with the per-token cost read from the profiled table at
/// `current_load`. Ties resolve to the shorter length (less engine time
/// for the same expectation).
///
/// Returns a plan with `verify_len = 0` when there is nothing to verify.
///
/// # Errors
///
/// - [`SpecError::InvalidLoad`] when the load is negative or non-finite.
/// - [`SpecError::InvalidSurvival`] when a curve entry is not a finite
///   probability in [0, 1].
pub fn schedule_for_load(
    table: &ThroughputTable,
    current_load: f32,
    survival: &[f32],
) -> Result<VerifyPlan, SpecError> {
    let point = table.throughput_at(current_load)?;
    // The curve must be genuine: a NaN or out-of-range entry would silently
    // corrupt the expected-value arithmetic below (NaN comparisons never
    // beat the running best; negatives yield nonsense expectations).
    for (index, &s) in survival.iter().enumerate() {
        if !s.is_finite() || s < 0.0 || s > 1.0 {
            return Err(SpecError::InvalidSurvival { index, value: s });
        }
    }
    let cap = survival.len().min(VERIFY_LEN_MAX);
    if cap == 0 {
        return Ok(VerifyPlan {
            verify_len: 0,
            expected_accepted: 0.0,
            expected_goodput_slots: 0.0,
            profiled_tokens_per_sec: point.tokens_per_sec,
        });
    }
    let mut best_len = 1;
    let mut best_expected = survival[0];
    let mut best_goodput = best_expected / (VERIFY_FIXED_COST_SLOTS + point.verify_cost_per_token);
    let mut expected = 0.0f32;
    for (index, &s) in survival[..cap].iter().enumerate() {
        let k = index + 1;
        expected += s;
        let cost = VERIFY_FIXED_COST_SLOTS + k as f32 * point.verify_cost_per_token;
        let goodput = expected / cost;
        if goodput > best_goodput {
            best_goodput = goodput;
            best_expected = expected;
            best_len = k;
        }
    }
    Ok(VerifyPlan {
        verify_len: best_len,
        expected_accepted: best_expected,
        expected_goodput_slots: best_goodput,
        profiled_tokens_per_sec: point.tokens_per_sec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(load: f32, tps: f32, cost: f32) -> ThroughputPoint {
        ThroughputPoint {
            load,
            tokens_per_sec: tps,
            verify_cost_per_token: cost,
        }
    }

    #[test]
    fn full_acceptance_round() {
        let out = draft_verify(
            || vec!['a', 'b', 'c', 'd', 'e'],
            |_| vec![true; 5],
            |_| 0.9,
            0.5,
        )
        .unwrap();
        assert_eq!(out.accepted, vec!['a', 'b', 'c', 'd', 'e']);
        assert_eq!(out.draft_len, 5);
        assert_eq!(out.verified_len, 5);
    }

    #[test]
    fn threshold_truncates_verify_length() {
        // Survival: .9, .81, .729, .6561, .59049. Threshold .7 -> length 3.
        let out = draft_verify(
            || vec![1, 2, 3, 4, 5],
            |submitted| {
                assert_eq!(submitted.len(), 3);
                vec![true; 3]
            },
            |_| 0.9,
            0.7,
        )
        .unwrap();
        assert_eq!(out.verified_len, 3);
        assert_eq!(out.accepted, vec![1, 2, 3]);
    }

    #[test]
    fn rejection_stops_the_accepted_prefix() {
        let out = draft_verify(
            || vec![1, 2, 3, 4, 5],
            |_| vec![true, true, false, true, true],
            |_| 1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(out.accepted, vec![1, 2]);
        assert_eq!(out.verified_len, 5);
    }

    #[test]
    fn short_verdict_stops_gracefully() {
        let out = draft_verify(|| vec![1, 2, 3, 4], |_| vec![true, true], |_| 1.0, 0.0).unwrap();
        assert_eq!(out.accepted, vec![1, 2]);
    }

    #[test]
    fn empty_draft_is_empty_result() {
        let out = draft_verify(Vec::<u8>::new, |_| vec![], |_| 1.0, 0.5).unwrap();
        assert!(out.accepted.is_empty());
        assert_eq!(out.draft_len, 0);
        assert_eq!(out.verified_len, 0);
    }

    #[test]
    fn bad_thresholds_rejected() {
        // NOTE: assert_eq cannot be used here because NaN != NaN.
        for bad in [f32::NAN, -0.1, 1.1, f32::INFINITY] {
            let err = draft_verify(|| vec![1], |_| vec![true], |_| 1.0, bad).unwrap_err();
            assert!(
                matches!(err, SpecError::InvalidThreshold { .. }),
                "threshold {bad} was accepted"
            );
        }
        // Boundaries are valid.
        assert!(draft_verify(|| vec![1], |_| vec![true], |_| 1.0, 0.0).is_ok());
        assert!(draft_verify(|| vec![1], |_| vec![true], |_| 1.0, 1.0).is_ok());
    }

    #[test]
    fn oversized_draft_rejected() {
        let err =
            draft_verify(|| vec![0u8; DRAFT_TOKEN_MAX + 1], |_| vec![], |_| 1.0, 0.0).unwrap_err();
        assert_eq!(
            err,
            SpecError::DraftTooLarge {
                len: DRAFT_TOKEN_MAX + 1
            }
        );
    }

    #[test]
    fn invalid_confidence_fails_closed() {
        let err = draft_verify(
            || vec![1, 2],
            |_| vec![true, true],
            |i| if i == 1 { 2.0 } else { 0.5 },
            0.0,
        )
        .unwrap_err();
        assert_eq!(
            err,
            SpecError::InvalidConfidence {
                index: 1,
                value: 2.0
            }
        );
    }

    #[test]
    fn survival_curve_is_prefix_products() {
        let s = survival_curve(&[0.5, 0.5, 1.0]).unwrap();
        assert_eq!(s.len(), 3);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 0.25).abs() < 1e-6);
        assert!((s[2] - 0.25).abs() < 1e-6);
        assert!(survival_curve(&[0.5, f32::NAN]).is_err());
    }

    #[test]
    fn table_validation() {
        assert_eq!(
            ThroughputTable::new(vec![]).unwrap_err(),
            SpecError::EmptyThroughputTable
        );
        assert!(matches!(
            ThroughputTable::new(vec![point(-1.0, 100.0, 0.5)]).unwrap_err(),
            SpecError::InvalidThroughputPoint { index: 0 }
        ));
        assert!(matches!(
            ThroughputTable::new(vec![point(0.0, 0.0, 0.5)]).unwrap_err(),
            SpecError::InvalidThroughputPoint { index: 0 }
        ));
        assert!(matches!(
            ThroughputTable::new(vec![point(0.0, 100.0, 1.5)]).unwrap_err(),
            SpecError::InvalidThroughputPoint { index: 0 }
        ));
        let too_many = vec![point(0.0, 100.0, 0.5); THROUGHPUT_TABLE_ENTRY_MAX + 1];
        assert!(matches!(
            ThroughputTable::new(too_many).unwrap_err(),
            SpecError::TooManyThroughputPoints { .. }
        ));
    }

    #[test]
    fn throughput_lookup_is_stepwise() {
        let table = ThroughputTable::new(vec![
            point(10.0, 50.0, 0.8),
            point(0.0, 200.0, 0.2),
            point(5.0, 100.0, 0.5),
        ])
        .unwrap();
        assert_eq!(table.throughput_at(0.0).unwrap().tokens_per_sec, 200.0);
        assert_eq!(table.throughput_at(7.0).unwrap().tokens_per_sec, 100.0);
        assert_eq!(table.throughput_at(99.0).unwrap().tokens_per_sec, 50.0);
        assert!(table.throughput_at(-1.0).is_err());
        assert!(table.throughput_at(f32::NAN).is_err());
    }

    #[test]
    fn scheduler_shortens_under_saturation() {
        // Decaying survival: longer verifies help less and less.
        // Survival curve: [0.9, 0.495, 0.2475, 0.12375]; idle (cost 0.2)
        // picks length 3, saturated (cost 0.95) picks length 2.
        let survival = survival_curve(&[0.9, 0.55, 0.5, 0.5]).unwrap();
        let idle = ThroughputTable::new(vec![point(0.0, 200.0, 0.2)]).unwrap();
        let saturated = ThroughputTable::new(vec![point(0.0, 200.0, 0.95)]).unwrap();
        let plan_idle = schedule_for_load(&idle, 0.0, &survival).unwrap();
        let plan_sat = schedule_for_load(&saturated, 0.0, &survival).unwrap();
        assert_eq!(plan_idle.verify_len, 3);
        assert_eq!(plan_sat.verify_len, 2);
        assert!(plan_idle.verify_len >= plan_sat.verify_len);
        assert_eq!(plan_sat.profiled_tokens_per_sec, 200.0);
    }

    #[test]
    fn scheduler_empty_survival_plans_nothing() {
        let table = ThroughputTable::new(vec![point(0.0, 100.0, 0.5)]).unwrap();
        let plan = schedule_for_load(&table, 0.0, &[]).unwrap();
        assert_eq!(plan.verify_len, 0);
        assert_eq!(plan.expected_accepted, 0.0);
    }
}
