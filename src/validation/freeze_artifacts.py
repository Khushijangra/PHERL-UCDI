#!/usr/bin/env python3
"""
Phase 4E: Artifact Freeze & Cryptographic Manifest Generator
Generates SHA-256 checksums for all critical pipeline artifacts,
produces artifact_manifest.json, and prints a reproducibility summary.
Run after all training and validation phases are complete.
"""
import os
import json
import hashlib
import datetime

def get_base_dir():
    # src/validation/freeze_artifacts.py -> src/validation -> src -> project_root
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def freeze_artifacts():
    base_dir = get_base_dir()
    
    # Define exactly which artifacts must be present for a valid release
    REQUIRED_ARTIFACTS = {
        "models": [
            "best_model.pt",
            "best_model_metadata.json",
            "gcn.pt",
            "gcn_config.json",
            "graphsage.pt",
            "gat.pt",
            "physics_gnn.pt",
            "xgboost.pkl",
            "lightgbm.pkl",
            "catboost.pkl",
            "randomforest.pkl",
        ],
        "reports": [
            "comparison_table.csv",
            "surrogate_report.md",
            "baseline_results.csv",
            "gnn_results.csv",
            "feature_importance.png",
        ],
        "data/processed": [
            "training_dataset.pt",
            "digital_twin_sanity_report.md",
        ],
        "configs": [
            "experiment.json",
        ],
    }

    manifest = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "project": "PHERL-UCDI++",
        "version": "v1.0-digital-twin-validated",
        "artifacts": {},
        "missing": [],
        "status": "UNKNOWN"
    }

    all_present = True

    for subdir, files in REQUIRED_ARTIFACTS.items():
        dir_path = os.path.join(base_dir, subdir)
        for fname in files:
            fpath = os.path.join(dir_path, fname)
            rel_key = f"{subdir}/{fname}"
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                sha = sha256_file(fpath)
                manifest["artifacts"][rel_key] = {
                    "sha256": sha,
                    "size_bytes": size,
                    "path": fpath
                }
                print(f"  [OK] {rel_key} ({size:,} bytes) - {sha[:16]}...")
            else:
                manifest["missing"].append(rel_key)
                print(f"  [MISSING]: {rel_key}")
                all_present = False

    # Optional artifacts (generated but not required for release)
    OPTIONAL_ARTIFACTS = [
        ("reports", "counterfactual_analysis.png"),
        ("reports", "counterfactual_results.csv"),
        ("reports", "shap_summary.png"),
        ("reports", "residual_plot.png"),
        ("reports", "learning_curve.png"),
        ("reports", "feature_importance.csv"),
        ("reports", "intervention_summary.csv"),
    ]
    for subdir, fname in OPTIONAL_ARTIFACTS:
        fpath = os.path.join(base_dir, subdir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            sha = sha256_file(fpath)
            manifest["artifacts"][f"{subdir}/{fname}"] = {
                "sha256": sha,
                "size_bytes": size,
                "path": fpath,
                "optional": True
            }
            print(f"  [OK] [optional] {subdir}/{fname}")

    manifest["status"] = "COMPLETE" if all_present else "INCOMPLETE"
    manifest["total_artifacts"] = len(manifest["artifacts"])

    out_path = os.path.join(base_dir, 'artifact_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STATUS: {manifest['status']}")
    print(f"Total artifacts hashed: {manifest['total_artifacts']}")
    if manifest['missing']:
        print(f"Missing artifacts: {manifest['missing']}")
    print(f"Manifest saved to: {out_path}")
    print(f"{'='*60}")
    return manifest

if __name__ == "__main__":
    print("PHERL-UCDI++ Artifact Freeze & Manifest Generator")
    print("=" * 60)
    freeze_artifacts()
