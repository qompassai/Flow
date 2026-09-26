//! KV-cache quantization precision policy, as data.
//!
//! Plain words: not every KV tensor survives aggressive quantization. The
//! DeepSeek-V4.1-Flash paper's serving recipe is: 4-bit for the long-lived
//! global (main) KV — but only with quantization-aware training (QAT) —
//! while the local sliding-window KV stays at 8-bit because it is more
//! quantization-sensitive, and quantization happens *after* RoPE. The
//! sparse indexer uses 4-bit queries/keys, again with QAT.
//!
//! This module does not quantize anything (there is no model here to
//! quantize). It captures the policy as plain data so serving code can ask
//! two honest questions: "does this configuration violate the paper's
//! rules?" ([`validate`]) and "what should I use for this component?"
//! ([`recommend`]). Anything the engine cannot actually do must surface as
//! a violation here, never as a silent approximation.

use std::fmt;

/// Which KV tensor the policy applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvComponent {
    /// Long-lived global (main) KV. Tolerates 4-bit with QAT.
    GlobalMain,
    /// Short-lived local sliding-window KV. Quantization-sensitive: stays
    /// at 8-bit or higher.
    LocalSwa,
    /// Sparse indexer queries/keys. Tolerates 4-bit with QAT.
    Indexer,
}

/// Storage precision for one KV tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit float. Always safe, always wasteful for KV.
    Fp32,
    /// 16-bit float.
    Fp16,
    /// 16-bit brain float.
    Bf16,
    /// 8-bit float. The floor for local/SWA KV.
    Fp8,
    /// 4-bit float. Global/indexer only, and only with QAT.
    Fp4,
}

impl Precision {
    /// Bits per stored value.
    pub fn bits(self) -> u8 {
        match self {
            Precision::Fp32 => 32,
            Precision::Fp16 => 16,
            Precision::Bf16 => 16,
            Precision::Fp8 => 8,
            Precision::Fp4 => 4,
        }
    }

    /// Bytes per stored value (fractional for sub-byte precisions).
    pub fn bytes_per_value(self) -> f32 {
        f32::from(self.bits()) / 8.0
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
            Precision::Fp8 => "fp8",
            Precision::Fp4 => "fp4",
        };
        write!(f, "{name}")
    }
}

impl fmt::Display for KvComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            KvComponent::GlobalMain => "global-main",
            KvComponent::LocalSwa => "local-swa",
            KvComponent::Indexer => "indexer",
        };
        write!(f, "{name}")
    }
}

/// One quantization policy: precision plus the two training/procedure
/// preconditions the paper's recipe depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantPolicy {
    /// Which tensor this policy governs.
    pub component: KvComponent,
    /// Storage precision for the tensor.
    pub precision: Precision,
    /// Whether quantization is applied after RoPE (the paper's procedure).
    /// Quantizing before RoPE adds decode-time overhead for marginal gain.
    pub quantize_after_rope: bool,
    /// Whether the model was trained with quantization-aware training for
    /// this precision. Required for 4-bit anywhere it is allowed.
    pub trained_with_qat: bool,
}

/// One rule violation found by [`validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyViolation {
    /// The tensor the violated rule governs.
    pub component: KvComponent,
    /// Machine-oriented rule identifier.
    pub rule: &'static str,
    /// Human-readable explanation.
    pub detail: &'static str,
}

impl fmt::Display for PolicyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kv-policy violation [{}] {}: {}",
            self.component, self.rule, self.detail
        )
    }
}

/// Check a policy against the paper's rules. Returns every violation found;
/// an empty vector means the policy is compliant.
///
/// Rules:
/// - `fp4_requires_qat`: 4-bit is only valid with quantization-aware
///   training, on any component.
/// - `swa_keeps_fp8_or_higher`: local sliding-window KV is too
///   quantization-sensitive for 4-bit, even with QAT.
/// - `quantize_after_rope`: quantization must happen after RoPE, not
///   before.
pub fn validate(policy: &QuantPolicy) -> Vec<PolicyViolation> {
    let mut violations = Vec::new();
    if policy.precision == Precision::Fp4 {
        if policy.component == KvComponent::LocalSwa {
            violations.push(PolicyViolation {
                component: policy.component,
                rule: "swa_keeps_fp8_or_higher",
                detail: "local sliding-window KV is quantization-sensitive; \
                         4-bit is not permitted even with QAT",
            });
        }
        if !policy.trained_with_qat {
            violations.push(PolicyViolation {
                component: policy.component,
                rule: "fp4_requires_qat",
                detail: "4-bit KV requires quantization-aware training",
            });
        }
    }
    if !policy.quantize_after_rope {
        violations.push(PolicyViolation {
            component: policy.component,
            rule: "quantize_after_rope",
            detail: "quantize after RoPE; pre-RoPE quantization adds \
                     decode overhead for marginal gain",
        });
    }
    violations
}

/// The paper's recommended policy per component. Every recommendation
/// passes [`validate`] with zero violations.
pub fn recommend(component: KvComponent) -> QuantPolicy {
    match component {
        // Paper: FP4 main KV via QAT, quantized after RoPE.
        KvComponent::GlobalMain => QuantPolicy {
            component,
            precision: Precision::Fp4,
            quantize_after_rope: true,
            trained_with_qat: true,
        },
        // Paper: SWA KV retains FP8 due to quantization sensitivity.
        KvComponent::LocalSwa => QuantPolicy {
            component,
            precision: Precision::Fp8,
            quantize_after_rope: true,
            trained_with_qat: false,
        },
        // Paper: FP4 indexer queries/keys via QAT.
        KvComponent::Indexer => QuantPolicy {
            component,
            precision: Precision::Fp4,
            quantize_after_rope: true,
            trained_with_qat: true,
        },
    }
}

/// Estimate the KV footprint for one token: `values_per_token` quantized
/// values (e.g. `2 * kv_heads * head_dim` for K and V) at the policy's
/// precision. Returns bytes as `f32` because sub-byte precisions produce
/// fractional bytes per value.
pub fn bytes_per_token(policy: &QuantPolicy, values_per_token: u32) -> f32 {
    values_per_token as f32 * policy.precision.bytes_per_value()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommendations_are_compliant() {
        for component in [
            KvComponent::GlobalMain,
            KvComponent::LocalSwa,
            KvComponent::Indexer,
        ] {
            let policy = recommend(component);
            assert_eq!(policy.component, component);
            assert!(
                validate(&policy).is_empty(),
                "recommendation for {component} should validate clean"
            );
        }
    }

    #[test]
    fn recommendation_precisions_match_paper() {
        assert_eq!(recommend(KvComponent::GlobalMain).precision, Precision::Fp4);
        assert_eq!(recommend(KvComponent::LocalSwa).precision, Precision::Fp8);
        assert_eq!(recommend(KvComponent::Indexer).precision, Precision::Fp4);
        assert!(recommend(KvComponent::GlobalMain).trained_with_qat);
        assert!(recommend(KvComponent::GlobalMain).quantize_after_rope);
    }

    #[test]
    fn fp4_without_qat_violates() {
        let policy = QuantPolicy {
            component: KvComponent::GlobalMain,
            precision: Precision::Fp4,
            quantize_after_rope: true,
            trained_with_qat: false,
        };
        let violations = validate(&policy);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule, "fp4_requires_qat");
    }

    #[test]
    fn fp4_on_swa_violates_even_with_qat() {
        let policy = QuantPolicy {
            component: KvComponent::LocalSwa,
            precision: Precision::Fp4,
            quantize_after_rope: true,
            trained_with_qat: true,
        };
        let violations = validate(&policy);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule, "swa_keeps_fp8_or_higher");
    }

    #[test]
    fn pre_rope_quantization_violates() {
        let policy = QuantPolicy {
            component: KvComponent::GlobalMain,
            precision: Precision::Fp8,
            quantize_after_rope: false,
            trained_with_qat: false,
        };
        let violations = validate(&policy);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule, "quantize_after_rope");
    }

    #[test]
    fn violations_accumulate() {
        // FP4 + no QAT + pre-RoPE on SWA: all three rules fire.
        let policy = QuantPolicy {
            component: KvComponent::LocalSwa,
            precision: Precision::Fp4,
            quantize_after_rope: false,
            trained_with_qat: false,
        };
        let violations = validate(&policy);
        assert_eq!(violations.len(), 3);
        let rules: Vec<&str> = violations.iter().map(|v| v.rule).collect();
        assert!(rules.contains(&"swa_keeps_fp8_or_higher"));
        assert!(rules.contains(&"fp4_requires_qat"));
        assert!(rules.contains(&"quantize_after_rope"));
    }

    #[test]
    fn conservative_precisions_are_compliant() {
        // FP8 global without QAT is safe (just less compact); FP16 SWA too.
        for policy in [
            QuantPolicy {
                component: KvComponent::GlobalMain,
                precision: Precision::Fp8,
                quantize_after_rope: true,
                trained_with_qat: false,
            },
            QuantPolicy {
                component: KvComponent::LocalSwa,
                precision: Precision::Fp16,
                quantize_after_rope: true,
                trained_with_qat: false,
            },
        ] {
            assert!(validate(&policy).is_empty());
        }
    }

    #[test]
    fn precision_widths() {
        assert_eq!(Precision::Fp32.bits(), 32);
        assert_eq!(Precision::Fp16.bits(), 16);
        assert_eq!(Precision::Bf16.bits(), 16);
        assert_eq!(Precision::Fp8.bits(), 8);
        assert_eq!(Precision::Fp4.bits(), 4);
        assert!((Precision::Fp4.bytes_per_value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn footprint_estimate() {
        // 2048 K+V values per token at FP4 -> 1024 bytes/token.
        let policy = recommend(KvComponent::GlobalMain);
        assert!((bytes_per_token(&policy, 2048) - 1024.0).abs() < 1e-3);
        let swa = recommend(KvComponent::LocalSwa);
        assert!((bytes_per_token(&swa, 2048) - 2048.0).abs() < 1e-3);
        assert_eq!(bytes_per_token(&policy, 0), 0.0);
    }

    #[test]
    fn violation_display_is_informative() {
        let policy = QuantPolicy {
            component: KvComponent::Indexer,
            precision: Precision::Fp4,
            quantize_after_rope: true,
            trained_with_qat: false,
        };
        let text = validate(&policy)[0].to_string();
        assert!(text.contains("indexer"));
        assert!(text.contains("fp4_requires_qat"));
    }
}
