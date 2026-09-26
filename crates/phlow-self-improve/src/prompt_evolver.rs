//! The disabled prompt evolver.
//!
//! Ports `flow/self_improve/prompt_evolver.py`. The Python evolver rewrote
//! operator prompts and committed the result to Git automatically; both
//! are disabled in the safe runtime. [`PromptEvolver::evolve`] always
//! fails with the exact Python denial, so no code path can mutate prompts
//! or create commits implicitly.

use crate::error::SelfImproveError;

/// The prompt evolver. Construction succeeds; evolution never does.
#[derive(Debug, Default)]
pub struct PromptEvolver;

impl PromptEvolver {
    /// Build the evolver. Building is harmless; only evolving is disabled.
    pub fn new() -> PromptEvolver {
        PromptEvolver
    }

    /// Evolve prompts from feedback. Always fails: automatic prompt
    /// evolution and implicit Git commits are disabled. Review and edit
    /// operator-owned prompts manually.
    pub fn evolve(&self) -> Result<(), SelfImproveError> {
        Err(SelfImproveError::EvolutionDisabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact Python denial, from flow/self_improve/prompt_evolver.py.
    const PYTHON_DENIAL: &str = "Automatic prompt evolution and implicit Git commits are disabled. Review and edit operator-owned prompts manually.";

    #[test]
    fn evolve_always_fails_with_exact_python_denial() {
        let evolver = PromptEvolver::new();
        let error = evolver.evolve().unwrap_err();
        assert!(matches!(error, SelfImproveError::EvolutionDisabled));
        assert_eq!(error.to_string(), PYTHON_DENIAL);
    }
}
