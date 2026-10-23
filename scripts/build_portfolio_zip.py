"""
RoadRisk AI — Clean Portfolio Packaging Script
==============================================
Creates a distribution ZIP archive excluding virtual environments, git history,
temporary caches, local credentials, raw large datasets, and heavy model binaries.
Verifies the archive contents and file size upon completion.
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIR_NAMES = {
    ".venv", "venv", ".git", ".pytest_cache", "__pycache__",
    "dist", "scratch", "temp", ".idea", ".vscode"
}

EXCLUDED_EXTENSIONS = {
    ".pyc", ".pyo", ".pth", ".pt", ".joblib", ".db", ".sqlite", ".sqlite3", ".log"
}

EXCLUDED_FILE_PATTERNS = {
    ".env", "secrets.toml"
}

EXCLUDED_PATHS = [
    "data/raw",
    "data/train",
    "data/val",
    "data/test",
]


def should_include_path(path: Path) -> bool:
    """Determine whether a file should be packaged into the portfolio ZIP."""
    rel = path.relative_to(PROJECT_ROOT).as_posix()

    # Exclude directories
    for part in path.parts:
        if part in EXCLUDED_DIR_NAMES:
            return False

    # Exclude sensitive files
    if path.name in EXCLUDED_FILE_PATTERNS or path.name.startswith(".env.") and path.name != ".env.example":
        return False

    # Exclude compiled & binary extensions
    if path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False

    # Exclude heavy raw data and raw images
    for ex_prefix in EXCLUDED_PATHS:
        if rel.startswith(ex_prefix):
            return False

    return True


def build_portfolio_zip() -> Path:
    """Package the clean project into a verified ZIP archive."""
    dist_dir = PROJECT_ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = "RoadRisk_AI_Portfolio_Package.zip"
    zip_path = dist_dir / zip_filename

    print(f"Building portfolio archive: {zip_path}")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]

            for file in files:
                file_path = Path(root) / file
                if should_include_path(file_path):
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                    zf.write(file_path, arcname=rel_path)
                    file_count += 1

    size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)
    size_kb = round(zip_path.stat().st_size / 1024, 2)
    print(f"Successfully packaged {file_count} files into {zip_path.name}")
    print(f"Archive Size: {size_kb} KB ({size_mb} MB)")

    # Integrity verification
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Verify critical files exist
        assert "README.md" in names
        assert "app.py" in names
        assert "requirements.txt" in names
        assert "docs/architecture.md" in names
        assert "docs/qa_report.md" in names
        assert "docs/model_card.md" in names
        assert "src/ai_assistant.py" in names

        # Verify forbidden files and directories are absent
        for n in names:
            assert not n.startswith(".venv/") and "/.venv/" not in n
            assert not n.startswith(".git/") and "/.git/" not in n
            assert not n.endswith(".db")
            assert not n.endswith(".sqlite")
            assert not n.endswith(".pth")
            assert not n.endswith(".joblib")
            assert "secrets.toml" not in n or n.endswith(".example")

    print("Verification Gate Passed: Archive is completely clean and ready for submission!")
    return zip_path


if __name__ == "__main__":
    build_portfolio_zip()
