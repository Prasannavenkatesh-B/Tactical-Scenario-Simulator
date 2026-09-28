"""Automated deliverable packaging utility for DRDO Tactical Scenario Simulator (TSS) v1.0.0."""

import hashlib
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent

# Files and directories required in the final deliverable package
REQUIRED_FILES = [
    # Top level
    "RELEASE_NOTES.md",
    "README.md",
    "COLLEGE_PROJECT_REPORT.docx",
    "TSS_Colab_Training_5000.ipynb",
    "simulator.html",
    # Docs (12 Markdown Files)
    "docs/README.md",
    "docs/ARCHITECTURE.md",
    "docs/TECHNICAL_REPORT.md",
    "docs/USER_GUIDE.md",
    "docs/API_REFERENCE.md",
    "docs/DATABASE_SCHEMA.md",
    "docs/TROUBLESHOOTING.md",
    "docs/CHANGELOG.md",
    "docs/INTEGRATION.md",
    "docs/NON_DETERMINISM_REPORT.md",
    "docs/REALISM_VALIDATION_REPORT.md",
    "docs/COLLEGE_PROJECT_REPORT.md",
    # Proposal (10 DRDO Forms)
    "proposal/FORM_1_SUMMARY.md",
    "proposal/FORM_2_TECHNICAL_BRIEF.md",
    "proposal/FORM_3A_LAB_RECOMMENDATION.md",
    "proposal/FORM_3B_COORDINATING_LAB.md",
    "proposal/FORM_4_EXTENDED_TECHNICAL.md",
    "proposal/FORM_7A_CERTIFICATE.md",
    "proposal/EXECUTIVE_SUMMARY.md",
    "proposal/FIGURES_OF_MERIT.md",
    "proposal/TRAINING_ROADMAP.md",
    "proposal/BUDGET_AND_TIMELINE.md",
    # Reports
    "reports/realism/report.json",
    "reports/realism/report.md",
    "reports/realism_stage_b/report.json",
    "reports/realism_stage_b/report.md",
    "reports/non_determinism/report.json",
    "reports/non_determinism/report.md",
    # Checkpoints
    "checkpoints/final/checkpoint_final.pt",
    "checkpoints/final/checkpoint_stage_a_final.pt",
    "checkpoints/final/checkpoint_stage_b_iter_01000.pt",
    # Web App
    "web/package.json",
    "web/index.html",
    "web/vite.config.ts",
    "web/src/App.tsx",
    "web/dist/index.html",
]

INCLUDED_DIRS = [
    "src",
    "scripts",
    "tests",
    "docs",
    "proposal",
    "checkpoints/final",
    "reports",
    "web",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_deliverables() -> bool:
    """Verify that all mandatory deliverable files exist on disk."""
    print("=" * 80)
    print("  VERIFYING DRDO TSS MANDATORY DELIVERABLE ARTIFACTS")
    print("=" * 80)
    missing = []
    for rel_path in REQUIRED_FILES:
        full_path = ROOT / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  [OK] {rel_path:48} ({size:>8} bytes)")
        else:
            print(f"  [MISSING] {rel_path}")
            missing.append(rel_path)

    if missing:
        print(f"\n[!] Verification FAILED: {len(missing)} required files missing.")
        return False

    print(f"\n[+] All {len(REQUIRED_FILES)} core deliverable documents verified successfully.")
    return True


def bundle_deliverable_zip(output_zip: Path) -> int:
    """Create deliverable.zip bundle containing all project source and documents."""
    print("\n" + "=" * 80)
    print(f"  BUNDLING DELIVERABLE PACKAGE: {output_zip.name}")
    print("=" * 80)

    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files and launch scripts
        root_files = [
            "RELEASE_NOTES.md",
            "README.md",
            "COLLEGE_PROJECT_REPORT.docx",
            "TSS_Colab_Training_5000.ipynb",
            "simulator.html",
            "LAUNCH_SIMULATOR.bat",
            "LAUNCH_TACTICAL_SIMULATOR.bat",
            "LAUNCH_REACT_C4ISR_CONSOLE.bat",
            "OPEN_WEB_SIMULATOR.bat",
            ".gitignore",
        ]
        for rf in root_files:
            p = ROOT / rf
            if p.exists():
                zf.write(p, arcname=rf)
                file_count += 1
                total_uncompressed_bytes += p.stat().st_size

        # Add included directories
        for d in INCLUDED_DIRS:
            dir_path = ROOT / d
            if not dir_path.exists():
                continue
            for root, _, files in os.walk(dir_path):
                # Skip heavy temporary and third-party files
                if any(x in root for x in ["__pycache__", "node_modules", ".git", ".mypy_cache", ".pytest_cache"]):
                    continue
                for f in files:
                    if f.endswith((".pyc", ".pyo", ".tmp")):
                        continue
                    full_p = Path(root) / f
                    arcname = str(full_p.relative_to(ROOT))
                    zf.write(full_p, arcname=arcname)
                    file_count += 1
                    total_uncompressed_bytes += full_p.stat().st_size

    archive_size = output_zip.stat().st_size
    checksum = compute_sha256(output_zip)

    print(f"[+] Total Files Bundled:          {file_count}")
    print(f"[+] Uncompressed Size:            {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"[+] Compressed Archive Size:      {archive_size / (1024 * 1024):.2f} MB")
    print(f"[+] Package SHA-256 Checksum:     {checksum}")
    print(f"[+] Output File Location:         {output_zip.resolve()}")
    print("=" * 80)
    return file_count


def main() -> None:
    if not verify_deliverables():
        sys.exit(1)

    out_zip = ROOT / "deliverable.zip"
    bundle_deliverable_zip(out_zip)
    print("\nDRDO TSS deliverable.zip bundle updated successfully.")


if __name__ == "__main__":
    main()
