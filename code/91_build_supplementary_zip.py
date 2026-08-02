"""
Build draft/latex/supplementary_material.zip from a clean checkout of the
anonymized public repository.

Two anonymity properties are asserted here rather than trusted:

1. No file in the archive contains an identifying string. The archive is
   built from `git ls-files` in the anonymized staging clone, which does not
   track main.tex (that file carries a de-anonymized \\author block that the
   TMLR class hides at render time but that would leak in source form).

2. The Zip ARCHIVE COMMENT is empty. This is the vector that a previous
   revision missed: the comment field is stored in the end-of-central-
   directory record, is not a file, and therefore does not appear in `unzip
   -l`, `zipinfo`, or any file listing. An earlier archive carried the
   project's git commit hash there, which would have identified the
   repository. `zipfile.ZipFile` writes an empty comment unless one is
   explicitly assigned, so the fix is to never assign it -- and then verify.

The identifying strings themselves are NOT hardcoded here -- writing them into
a file that ships inside the anonymized archive would recreate the very leak
this script exists to prevent. Supply them at call time:

    ANON_PATTERNS='name1|name2|@institution' \
        python3 91_build_supplementary_zip.py <staging_repo> <out.zip>

If ANON_PATTERNS is unset the content scan is skipped (and says so loudly);
the archive-comment assertion always runs.

Usage:
    python3 91_build_supplementary_zip.py <staging_repo> <out.zip>
"""
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

_pat = os.environ.get("ANON_PATTERNS", "").strip()
IDENTITY = re.compile(_pat.encode(), re.IGNORECASE) if _pat else None
# Text-ish extensions worth scanning for identity strings.
SCAN_SUFFIXES = {".py", ".md", ".txt", ".json", ".tex", ".yml", ".yaml",
                 ".cfg", ".toml", ".sh", ".ipynb", ".bst", ".sty", ""}


def main():
    repo = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()

    files = subprocess.run(["git", "-C", str(repo), "ls-files"],
                           capture_output=True, text=True, check=True
                           ).stdout.split()
    assert files, "no tracked files found"
    assert not any(f.endswith("main.tex") for f in files), \
        "main.tex is tracked in the anonymized repo -- it carries a de-anonymized author block"

    if IDENTITY is None:
        print("WARNING: ANON_PATTERNS unset -- skipping the content identity scan.")
        n_scanned = 0
    else:
        leaks = []
        n_scanned = 0
        for rel in files:
            p = repo / rel
            if p.suffix.lower() in SCAN_SUFFIXES:
                n_scanned += 1
                if IDENTITY.search(p.read_bytes()):
                    leaks.append(rel)
        assert not leaks, f"identity strings found in: {leaks}"

    if out.exists():
        out.unlink()
    # NOTE: .comment is deliberately never assigned -> stays b"".
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(files):
            z.write(repo / rel, arcname=rel)

    # Verify after the fact, on a freshly opened handle.
    with zipfile.ZipFile(out) as z:
        assert z.comment == b"", f"archive comment is not empty: {z.comment!r}"
        names = z.namelist()
        bad = z.testzip()
        assert bad is None, f"corrupt entry: {bad}"

    print(f"Built {out}")
    print(f"  entries:        {len(names)}")
    print(f"  archive comment: {zipfile.ZipFile(out).comment!r} (empty)")
    print(f"  identity scan:   clean over {n_scanned} scanned files"
          if IDENTITY is not None else "  identity scan:   SKIPPED")


if __name__ == "__main__":
    sys.exit(main())
