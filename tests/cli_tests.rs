use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

#[test]
fn train_classify_and_export_browser_playground() {
    let dir = tempdir().unwrap();
    let csv_path = dir.path().join("notes.csv");
    let model_path = dir.path().join("notes.tk.json");
    let html_path = dir.path().join("playground.html");

    let mut rows = String::from("id,note_text,category\n");
    let fin_samples = [
        "tuition balance hold delayed spring registration scholarship needed",
        "student asked about payment plan financial aid grant disbursement",
        "unable to pay tuition invoice requested bursar hold extension",
        "financial hardship emergency scholarship grant application submitted",
        "billing balance overdue payment plan adjustment requested",
        "financial aid package tuition assistance scholarship renewal inquiry",
        "bursar balance hold cannot register until tuition payment clears",
        "unexpected medical expense affecting tuition payment plan schedule",
        "asked financial aid office about scholarship eligibility tuition gap",
        "tuition invoice balance payment plan extension needed before drop",
        "scholarship aid application tuition balance payment plan hold",
        "financial hardship tuition payment bursar hold grant inquiry",
        "bursar billing balance tuition payment plan financial aid office",
    ];
    let acad_samples = [
        "struggling with greek grammar paradigms quiz scores falling behind",
        "hebrew vocabulary exam difficulty requested tutoring writing center",
        "overwhelmed by systematic theology exegetical paper workload",
        "missed two assignments greek syntax quiz needs academic tutoring",
        "falling behind on weekly lectures exegetical paper outline overdue",
        "requested writing center support for church history research paper",
        "greek grammar exam retake tutoring sessions scheduled weekly",
        "hebrew exegesis homework behind schedule needs study plan",
        "canvas quiz scores low struggling with reading load exam prep",
        "academic workload heavy asked professor for essay deadline extension",
        "greek grammar homework behind schedule requested academic tutoring",
        "hebrew quiz scores low struggling with exam preparation workload",
        "systematic theology paper reading load overwhelming needs tutoring",
    ];
    let min_samples = [
        "pastoral ministry travel conference pulpit supply busy church season",
        "church planting launch schedule conflict with live seminar hours",
        "full time youth pastor ministry duties easter week preaching load",
        "mission trip overseas internet access limited during ministry week",
        "senior pastor sabbatical coverage church council meetings evening",
        "bi-vocational pastor preaching schedule conflicts with weekly seminar",
        "ministry internship hospital visitation pastoral care busy week",
        "church retreat leadership travel delayed discussion board posts",
        "associate pastor sermon preparation youth camp ministry travel",
        "church session meetings pastoral counseling schedule very full",
        "pastoral travel preaching ministry conference busy church schedule",
        "church youth ministry retreat travel conflict with evening seminar",
        "pastor preaching schedule church leadership meetings heavy load",
    ];
    let mut id = 1;
    for s in fin_samples {
        rows.push_str(&format!("N{id},\"{s}\",financial\n"));
        id += 1;
    }
    for s in acad_samples {
        rows.push_str(&format!("N{id},\"{s}\",academic\n"));
        id += 1;
    }
    for s in min_samples {
        rows.push_str(&format!("N{id},\"{s}\",ministry_schedule\n"));
        id += 1;
    }
    fs::write(&csv_path, rows).unwrap();

    let train_out = Command::cargo_bin("textkeel")
        .unwrap()
        .args([
            "train",
            csv_path.to_str().unwrap(),
            "--text",
            "note_text",
            "--label",
            "category",
            "--out",
            model_path.to_str().unwrap(),
            "--min-macro-f1",
            "0.85",
            "--export-browser",
            html_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let train_json: serde_json::Value = serde_json::from_slice(&train_out).unwrap();
    assert_eq!(train_json["gate_passed"], true);
    assert!(train_json["eval"]["macro_f1"].as_f64().unwrap() >= 0.85);
    assert!(fs::read_to_string(&html_path).unwrap().contains("const MODEL ="));

    // Classify an unseen text locally
    let cls_out = Command::cargo_bin("textkeel")
        .unwrap()
        .args([
            "classify",
            model_path.to_str().unwrap(),
            "--text-str",
            "student has a bursar tuition balance hold and asked about scholarship payment plan",
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let cls_json: serde_json::Value = serde_json::from_slice(&cls_out).unwrap();
    assert_eq!(cls_json["predictions"][0]["predicted_label"], "financial");
    assert!(cls_json["predictions"][0]["confidence"].as_f64().unwrap() > 0.70);
}

#[test]
fn skill_prints_and_installs() {
    let dir = tempdir().unwrap();
    Command::cargo_bin("textkeel")
        .unwrap()
        .args(["skill", "--install", dir.path().to_str().unwrap(), "--format", "human"])
        .assert()
        .success();
    assert!(dir.path().join("textkeel/SKILL.md").exists());
}
