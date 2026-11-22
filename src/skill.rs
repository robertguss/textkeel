use std::fs;
use std::path::Path;

pub const SKILL_MD: &str = include_str!("../SKILL.md");

pub fn handle_skill(install_dir: Option<&Path>) -> Result<(), String> {
    if let Some(dir) = install_dir {
        let target_dir = dir.join("textkeel");
        fs::create_dir_all(&target_dir)
            .map_err(|e| format!("failed to create {}: {e}", target_dir.display()))?;
        let target_file = target_dir.join("SKILL.md");
        fs::write(&target_file, SKILL_MD)
            .map_err(|e| format!("failed to write {}: {e}", target_file.display()))?;
        eprintln!("Installed textkeel skill -> {}", target_file.display());
    } else {
        print!("{SKILL_MD}");
    }
    Ok(())
}
