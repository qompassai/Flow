//! The read-only skill store.
//!
//! Ports `flow/self_improve/skill_store.py`. The store reads through the
//! root-pinned, read-only [`Workspace`](phlow_workspace::Workspace):
//! [`SkillStore::load`] reads the exact relative path given (a directory
//! fails with a directory error, as Python's `IsADirectoryError`), and
//! [`SkillStore::list_skills`] returns the workspace's bounded recursive
//! file list. Missing skills return absence (`None` / empty list), never
//! an error. Mutation is disabled: [`SkillStore::save_skill`] and
//! [`SkillStore::revert_skill`] always fail with the exact Python
//! denials, and [`SkillStore::history`] always returns an empty list
//! instead of reading Git history.

use std::io::ErrorKind;
use std::path::Path;

use phlow_workspace::{Workspace, WorkspaceError};

use crate::error::SelfImproveError;

/// A skill directory name longer than this is rejected; names come from
/// model input and must stay short labels.
pub const SKILL_NAME_CHARS_MAX: usize = 128;

/// Read-only access to a skills directory.
pub struct SkillStore {
    workspace: Option<Workspace>,
}

impl SkillStore {
    /// Open the skill store over `skills_dir`. A missing directory is not
    /// an error: reads return absence, as in Python.
    pub fn open(skills_dir: &Path) -> Result<SkillStore, SelfImproveError> {
        if !skills_dir.is_dir() {
            return Ok(SkillStore { workspace: None });
        }
        let workspace = Workspace::open(skills_dir, false, &[])?;
        Ok(SkillStore {
            workspace: Some(workspace),
        })
    }

    /// Load a skill file's content at the exact relative path given — no
    /// implicit `SKILL.md` is appended. Returns `None` when the path is
    /// missing, mirroring Python's `FileNotFoundError` catch; reading a
    /// directory fails with a directory error, mirroring Python's
    /// `IsADirectoryError`.
    pub fn load(&self, name: &str) -> Result<Option<String>, SelfImproveError> {
        let name = check_name(name)?;
        let Some(workspace) = &self.workspace else {
            return Ok(None);
        };
        match workspace.read(&name) {
            Ok(result) => Ok(Some(result.content)),
            Err(WorkspaceError::Io(error)) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// The workspace's bounded recursive file list, exactly as Python's
    /// `list_skills` returned `workspace.list()["files"]`: files sorted
    /// within each directory, directories visited in sorted order, at most
    /// 500 entries (the workspace's `LIST_FILES_MAX`, as in Python).
    /// Empty when the skills directory is missing, as in Python.
    pub fn list_skills(&self) -> Result<Vec<String>, SelfImproveError> {
        let Some(workspace) = &self.workspace else {
            return Ok(Vec::new());
        };
        Ok(workspace.list(".")?.files)
    }

    /// Skill history is not read from Git; always an empty list, as the
    /// Python implementation returned `[]`.
    pub fn history(&self, _name: &str) -> Vec<serde_json::Value> {
        Vec::new()
    }

    /// Saving a skill is disabled: automatic skill mutation would let the
    /// agent rewrite its own capabilities.
    pub fn save_skill(&self, _name: &str, _content: &str) -> Result<(), SelfImproveError> {
        Err(SelfImproveError::SkillMutationDisabled)
    }

    /// Reverting a skill via Git is disabled: automatic Git mutation would
    /// let the agent rewrite repository history.
    pub fn revert_skill(&self, _name: &str) -> Result<(), SelfImproveError> {
        Err(SelfImproveError::GitMutationDisabled)
    }
}

/// Skill names are short operator-chosen labels.
fn check_name(name: &str) -> Result<String, SelfImproveError> {
    if name.chars().count() > SKILL_NAME_CHARS_MAX {
        return Err(SelfImproveError::TooLong {
            field: "skill name",
            max_chars: SKILL_NAME_CHARS_MAX,
        });
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlow_workspace::LIST_FILES_MAX;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Exact Python denials from flow/self_improve/skill_store.py.
    const PYTHON_SAVE_DENIAL: &str = "Automatic skill mutation is disabled; edit manually";
    const PYTHON_REVERT_DENIAL: &str = "Automatic Git mutation is disabled; revert manually";

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phlow-self-improve-skills-test-{prefix}-{}-{}",
            std::process::id(),
            TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("test setup: create temp dir");
        dir
    }

    fn skill_dir() -> PathBuf {
        let dir = temp_dir("store");
        for name in ["skill-b", "skill-a"] {
            let skill = dir.join(name);
            std::fs::create_dir_all(&skill).unwrap();
            std::fs::write(skill.join("SKILL.md"), format!("# {name}\n")).unwrap();
        }
        // A plain file is not a skill.
        std::fs::write(dir.join("notes.txt"), "not a skill").unwrap();
        dir
    }

    #[test]
    fn load_returns_skill_content() {
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        assert_eq!(
            store.load("skill-a/SKILL.md").unwrap(),
            Some("# skill-a\n".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_directory_fails_with_directory_error() {
        // Python's SkillStore.load("skill-a") raises IsADirectoryError: the
        // exact path is read, no SKILL.md is appended. Verified against the
        // real Python 2026-09-26.
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        let error = store.load("skill-a").unwrap_err();
        assert!(
            matches!(
                error,
                SelfImproveError::Workspace(WorkspaceError::NotRegularFile)
            ),
            "unexpected error: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_skill_returns_none() {
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        assert_eq!(store.load("nope").unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_skills_dir_yields_absence_not_error() {
        let dir = temp_dir("gone");
        let _ = std::fs::remove_dir_all(&dir);
        let store = SkillStore::open(&dir).unwrap();
        assert_eq!(store.load("skill-a").unwrap(), None);
        assert_eq!(store.list_skills().unwrap(), Vec::<String>::new());
    }

    #[test]
    fn list_skills_returns_python_recursive_file_list() {
        // Differential against the real Python SkillStore.list_skills()
        // over the same fixture layout (captured 2026-09-26):
        // ['notes.txt', 'skill-a/SKILL.md', 'skill-b/SKILL.md'] —
        // the workspace's bounded recursive file list, not bare directory
        // names.
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        assert_eq!(
            store.list_skills().unwrap(),
            vec!["notes.txt", "skill-a/SKILL.md", "skill-b/SKILL.md"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_bound_matches_python_workspace() {
        // Python's list_skills returns workspace.list()["files"], whose cap
        // is flow/workspace.py LIST_FILES_MAX = 500.
        assert_eq!(LIST_FILES_MAX, 500);
    }

    #[test]
    fn history_is_always_empty() {
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        assert!(store.history("skill-a").is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_skill_fails_with_exact_python_denial() {
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        let error = store.save_skill("skill-a", "# new\n").unwrap_err();
        assert!(matches!(error, SelfImproveError::SkillMutationDisabled));
        assert_eq!(error.to_string(), PYTHON_SAVE_DENIAL);
        // Nothing was written.
        assert_eq!(
            store.load("skill-a/SKILL.md").unwrap(),
            Some("# skill-a\n".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn revert_skill_fails_with_exact_python_denial() {
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        let error = store.revert_skill("skill-a").unwrap_err();
        assert!(matches!(error, SelfImproveError::GitMutationDisabled));
        assert_eq!(error.to_string(), PYTHON_REVERT_DENIAL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn overlong_skill_name_is_rejected() {
        let dir = skill_dir();
        let store = SkillStore::open(&dir).unwrap();
        let long = "x".repeat(SKILL_NAME_CHARS_MAX + 1);
        assert!(store.load(&long).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
