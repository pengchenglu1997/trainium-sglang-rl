#!/usr/bin/env python3
"""End-of-session step: record the size in bytes and the file count of each compile-cache folder.

    python 82_cache_record.py --results DIR --tag t1 --cache DIR [--cache DIR ...]
    python 82_cache_record.py --results DIR --tag t1 --root DIR        every sub-folder of DIR whose name holds "cache", except cache_archives

Run it after the servers that write into the folders have stopped: a folder a container is still writing to gives a count
that is not final (the record says nothing about that; stop the servers first). Nothing is changed or deleted.

Result file DIR/82_cache_record_<tag>.jsonl (rl_common format, last line {"done": true}), one line per folder:
  {"event": "folder", "name", "path", "exists", "bytes", "files", "dirs", "unreadable", "by_top": {first-level entry: {"bytes",
   "files"}}, "profile_entries": [first-level entries named profile_* or devprof_*]}
and a last {"event": "summary", "folders", "bytes", "files", "unix"}.
bytes = sum of the sizes of all files below the folder (a symbolic link counts as itself, its target is not followed);
files = number of files and links (folders are not counted, they are in "dirs"); "unreadable" = entries that could not be
read, so a number with unreadable > 0 is a lower bound. "profile_entries" lists device or system profile output still inside
the cache folder (the profile scripts of this kit write outside it, see 59_decode_profile.py); non-empty means profile data
is mixed into the count.
A folder that does not exist is recorded with exists false and no numbers; a missing --cache is not an error, so the step
can run after a session that never started the server.
--root skips a sub-folder named cache_archives (tar archives of the caches, not a cache); give it with --cache if it is wanted."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402

PROFILE_PREFIXES = ("profile_", "devprof_")
# sub-folders of --root that hold "cache" in the name but are not compile caches: the save procedure writes tar archives of the caches there
NOT_CACHE = ("cache_archives",)


def scan(folder: Path) -> dict:
    """Walk folder without following links; return bytes, files, dirs, unreadable and the first-level breakdown."""
    errors = [0]

    def onerr(_e):
        errors[0] += 1

    total_b = total_f = total_d = 0
    by_top: dict[str, dict] = {}
    for root, dirs, files in os.walk(folder, followlinks=False, onerror=onerr):
        rel = Path(root).relative_to(folder)
        total_d += len(dirs)
        for name in files:
            top = rel.parts[0] if rel.parts else name
            try:
                size = os.lstat(os.path.join(root, name)).st_size
            except OSError:
                errors[0] += 1
                continue
            total_b += size
            total_f += 1
            t = by_top.setdefault(top, {"bytes": 0, "files": 0})
            t["bytes"] += size
            t["files"] += 1
        # symbolic links to folders are listed in dirs by os.walk but not entered: count them as files
        for name in list(dirs):
            p = os.path.join(root, name)
            if os.path.islink(p):
                try:
                    size = os.lstat(p).st_size
                except OSError:
                    errors[0] += 1
                    continue
                dirs.remove(name)
                total_d -= 1
                total_b += size
                total_f += 1
                top = rel.parts[0] if rel.parts else name
                t = by_top.setdefault(top, {"bytes": 0, "files": 0})
                t["bytes"] += size
                t["files"] += 1
    return {"bytes": total_b, "files": total_f, "dirs": total_d, "unreadable": errors[0], "by_top": by_top}


def profile_entries(folder: Path) -> list[str]:
    try:
        return sorted(e for e in os.listdir(folder) if e.startswith(PROFILE_PREFIXES))
    except OSError:
        return []


def record(folder: Path) -> dict:
    row = {"event": "folder", "name": folder.name, "path": str(folder), "exists": folder.is_dir()}
    if row["exists"]:
        row.update(scan(folder))
        row["profile_entries"] = profile_entries(folder)
    return row


def folders_from(args) -> list[Path]:
    out = [Path(c) for c in args.cache]
    for r in args.root:
        root = Path(r)
        if root.is_dir():
            out += sorted(p for p in root.iterdir() if p.is_dir() and "cache" in p.name and p.name not in NOT_CACHE)
    seen, uniq = set(), []
    for p in out:
        k = os.path.abspath(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", action="append", default=[], help="a compile-cache folder (repeatable)")
    ap.add_argument("--root", action="append", default=[], help="a folder whose sub-folders with 'cache' in the name are recorded")
    a = ap.parse_args(argv)
    folders = folders_from(a)
    res = C.Results(a.results, f"82_cache_record_{a.tag}")
    rows = []
    try:
        if not folders:
            raise RuntimeError("no cache folder given (--cache DIR or --root DIR with a sub-folder that has 'cache' in its name)")
        for f in folders:
            row = record(f)
            rows.append(row)
            res.write(**row)
            print(f"{row['name']}: " + (f"{row['bytes']} bytes, {row['files']} files, unreadable {row['unreadable']}"
                                        + (f", profile entries inside: {row['profile_entries']}" if row["profile_entries"] else "")
                                        if row["exists"] else "absent"))
        res.write(event="summary", folders=len(rows), bytes=sum(r.get("bytes", 0) for r in rows),
                  files=sum(r.get("files", 0) for r in rows), unix=round(time.time(), 1))
    except BaseException as e:  # noqa: BLE001
        res.fail("cache_record", e)
        res.done()
        print(f"FAILED: {e}")
        return 1
    res.done()
    return 0


if __name__ == "__main__":
    sys.exit(main())
