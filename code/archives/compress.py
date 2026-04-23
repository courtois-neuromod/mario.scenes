#!/usr/bin/env python
"""Pack each session's gamelogs/ directory into a single .tar archive.

Archives are written to ``sub-XX/ses-YYY/gamelogs.tar`` alongside the
existing ``func/`` directory.  The original ``gamelogs/`` directory is left
intact unless ``--remove-source`` is passed.

Usage::

    python compress.py -o /path/to/mario.scenes
    python compress.py -o /path/to/mario.scenes --subjects sub-01 sub-02
    python compress.py -o /path/to/mario.scenes --sessions ses-001 --remove-source
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

def _find_sessions(root, subjects=None, sessions=None):
    """Yield (sub, ses, gamelogs_dir) tuples for all matching sessions."""
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
            gamelogs = op.join(sub_dir, ses, "gamelogs")
            if op.isdir(gamelogs):
                yield sub, ses, gamelogs


def _compress_session(sub, ses, gamelogs_dir, root, force=False, remove_source=False):
    """Create gamelogs.tar for one session.  Returns True if archive was written."""
    archive_path = op.join(root, sub, ses, "gamelogs.tar")

    if op.exists(archive_path) and not force:
        log.info("  skipping %s/%s — archive already exists (use --force to overwrite)", sub, ses)
        return False

    log.info("  packing %s/%s ...", sub, ses)
    tmp_path = archive_path + ".tmp"
    try:
        with tarfile.open(tmp_path, "w") as tar:
            tar.add(gamelogs_dir, arcname="gamelogs")
        os.replace(tmp_path, archive_path)
    except Exception:
        if op.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if remove_source:
        import shutil
        shutil.rmtree(gamelogs_dir)
        log.info("  removed source %s", gamelogs_dir)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Pack gamelogs/ directories into per-session .tar archives."
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
        help="Overwrite existing archives",
    )
    p.add_argument(
        "--remove-source", action="store_true",
        help="Delete gamelogs/ after a successful archive (irreversible)",
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
    sessions = list(_find_sessions(root, args.subjects, args.sessions))

    if not sessions:
        log.warning("No gamelogs/ directories found matching the given filters.")
        return

    n_packed = 0
    for sub, ses, gamelogs_dir in tqdm(sessions, desc="Compressing sessions"):
        try:
            packed = _compress_session(
                sub, ses, gamelogs_dir, root,
                force=args.force,
                remove_source=args.remove_source,
            )
            if packed:
                n_packed += 1
        except Exception as exc:
            log.error("  failed %s/%s: %s", sub, ses, exc)

    print(f"Done. {n_packed}/{len(sessions)} session(s) archived.")


if __name__ == "__main__":
    main()
