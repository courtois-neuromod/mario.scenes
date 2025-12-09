#!/usr/bin/env python
"""Unpack per-session gamelogs.tar archives into gamelogs/ directories.

Looks for ``sub-XX/ses-YYY/gamelogs.tar`` and extracts each one back to
``sub-XX/ses-YYY/gamelogs/``.  Sessions whose ``gamelogs/`` directory already
exists are skipped unless ``--force`` is passed.

Usage::

    python decompress.py -o /path/to/mario.scenes
    python decompress.py -o /path/to/mario.scenes --subjects sub-01
    python decompress.py -o /path/to/mario.scenes --sessions ses-001 ses-002 --force
"""

import argparse
import logging
import os
import os.path as op
import tarfile

from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_archives(root, subjects=None, sessions=None):
    """Yield (sub, ses, archive_path) tuples for all matching gamelogs.tar files."""
    for sub in sorted(os.listdir(root)):
        if not sub.startswith("sub-"):
            continue
        if subjects and sub not in subjects:
            continue
        sub_dir = op.join(root, sub)
        if not op.isdir(sub_dir):
            continue
        for ses in sorted(os.listdir(sub_dir)):
            if not ses.startswith("ses-"):
                continue
            if sessions and ses not in sessions:
                continue
            archive = op.join(sub_dir, ses, "gamelogs.tar")
            if op.exists(archive):
                yield sub, ses, archive


def _decompress_session(sub, ses, archive_path, force=False):
    """Extract gamelogs.tar for one session.  Returns True if extracted."""
    session_dir = op.dirname(archive_path)
    gamelogs_dir = op.join(session_dir, "gamelogs")

    if op.isdir(gamelogs_dir) and not force:
        log.info("  skipping %s/%s — gamelogs/ already exists (use --force to overwrite)", sub, ses)
        return False

    log.info("  extracting %s/%s ...", sub, ses)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=session_dir)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Unpack per-session gamelogs.tar archives into gamelogs/ directories."
    )
    p.add_argument(
        "-o", "--output", default=".",
        help="Root of the mario.scenes dataset (default: current directory)",
    )
    p.add_argument(
        "--subjects", nargs="+", metavar="SUB",
        help="Restrict to these subjects (e.g. sub-01 sub-02)",
    )
    p.add_argument(
        "--sessions", nargs="+", metavar="SES",
        help="Restrict to these sessions (e.g. ses-001 ses-002)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-extract even if gamelogs/ already exists",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    level = logging.WARNING if args.verbose == 0 else (
        logging.INFO if args.verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(format="%(levelname)s %(message)s", level=level)

    root = op.abspath(args.output)
    archives = list(_find_archives(root, args.subjects, args.sessions))

    if not archives:
        log.warning("No gamelogs.tar archives found matching the given filters.")
        return

    n_extracted = 0
    for sub, ses, archive_path in tqdm(archives, desc="Decompressing sessions"):
        try:
            extracted = _decompress_session(sub, ses, archive_path, force=args.force)
            if extracted:
                n_extracted += 1
        except Exception as exc:
            log.error("  failed %s/%s: %s", sub, ses, exc)

    print(f"Done. {n_extracted}/{len(archives)} session(s) extracted.")


if __name__ == "__main__":
    main()
