#!/usr/bin/env python3
"""Scrub check for result folders that go into the public repository.

    python scrub_check.py DIR [--words FILE] [--allow FILE] [--report OUT]

Scans every file under DIR (text and binary) for things that must not be published:
  instance ids (i- and 8 to 17 hex digits), spot request ids (sir-), 12-digit numbers (account ids),
  IPv4 addresses other than 127.0.0.1 and 0.0.0.0, host names ip-N-N-N-N, e-mail addresses, "arn:aws",
  access-key-like strings (AKIA / ASIA), Windows paths (a drive letter and a slash), home folders other than
  /home/ubuntu, "user@host" forms such as "ubuntu@", key file names (.pem, .ppk, id_rsa), prices (an amount per hour or
  day, USD, spot or on-demand price), a carriage return (the files are kept with LF line ends),
  and the project-internal words of a private word list (one word or phrase per line, matched without
  regard to case, kept outside the repository, because the list itself holds names).
A file that is not valid UTF-8 is reported as a hit. The exit status is 1 when anything was found.
This script and scrub_allow.txt hold the patterns as text and are not scanned.
--allow FILE: lines "path<TAB>pattern name<TAB>reason" for hits that are accepted on purpose (the report lists them).
"""
import argparse
import os
import re
import sys

PATTERNS = [
    ("instance-id", re.compile(r"\bi-[0-9a-f]{8,17}\b")),
    ("spot-request-id", re.compile(r"\bsir-[0-9a-z]{8}\b")),
    ("account-id-12-digits", re.compile(r"(?<![0-9A-Za-z.])[0-9]{12}(?![0-9A-Za-z])")),
    ("ipv4", re.compile(r"(?<![0-9.])(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]?[0-9])){3}(?!\.?[0-9])")),
    ("ip-hostname", re.compile(r"\bip-[0-9]+-[0-9]+-[0-9]+-[0-9]+\b")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("user-at-host", re.compile(r"\b[A-Za-z0-9_-]+@(?=[A-Za-z0-9._-]+\b)")),
    ("arn", re.compile(r"arn:aws")),
    ("access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{8,}\b")),
    ("windows-path", re.compile(r"(?<![A-Za-z0-9\\])[A-Za-z]:[\\/]")),
    ("other-home-folder", re.compile(r"/(?:home|Users)/(?!ubuntu\b)[A-Za-z0-9._-]+")),
    ("key-file", re.compile(r"\.pem\b|\.ppk\b|\bid_rsa\b|\bid_ed25519\b")),
    ("price", re.compile(r"\bUSD\b|spot price|on-demand|\bhourly\b|\$\s?[0-9][0-9,.]*\s*(?:/\s*h|per hour|an hour|a day|/day)", re.I)),
]


def load_words(path):
    words = []
    if path:
        for line in open(path, encoding="utf-8"):
            w = line.strip()
            if w and not w.startswith("#"):
                words.append(w)
    return words


def load_allow(path):
    allow = {}
    if path:
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                allow[(parts[0], parts[1])] = parts[2]
    return allow


def scan(root, words, allow):
    hits, allowed, nfiles, nbytes = [], [], 0, 0
    wre = [(w, re.compile(re.escape(w), re.I)) for w in words]
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).replace("\\", "/")
            if rel in ("scrub_report.txt", "scrub_check.py", "scrub_allow.txt"):   # these hold the patterns / accepted hits as text
                continue
            raw = open(p, "rb").read()
            nfiles += 1
            nbytes += len(raw)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                hits.append((rel, 0, "not-utf8", ""))
                continue
            if "\r" in text:
                hits.append((rel, 0, "carriage-return", ""))
            for ln, line in enumerate(text.split("\n"), 1):
                for name, rx in PATTERNS:
                    mo = rx.search(line)
                    if mo:
                        rec = (rel, ln, name, mo.group(0))
                        (allowed if (rel, name) in allow else hits).append(rec)
                for w, rx in wre:
                    if rx.search(line):
                        rec = (rel, ln, "private-word", "<word %d of the list>" % (words.index(w) + 1))
                        (allowed if (rel, "private-word") in allow else hits).append(rec)
    return hits, allowed, nfiles, nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--words")
    ap.add_argument("--allow")
    ap.add_argument("--report")
    a = ap.parse_args()
    words = load_words(a.words)
    allow = load_allow(a.allow)
    hits, allowed, nfiles, nbytes = scan(a.dir, words, allow)
    lines = ["scrub_check: %d files, %d bytes scanned" % (nfiles, nbytes),
             "patterns: %s" % ", ".join(n for n, _ in PATTERNS),
             "private word list: %d entries" % len(words),
             "hits: %d" % len(hits)]
    for h in hits[:200]:
        lines.append("  HIT %s:%d %s %r" % h)
    if len(hits) > 200:
        lines.append("  ... %d more" % (len(hits) - 200))
    lines.append("accepted on purpose: %d" % len(allowed))
    for rel, ln, name, val in allowed[:50]:
        lines.append("  OK  %s:%d %s %r  (%s)" % (rel, ln, name, val, allow[(rel, name)]))
    txt = "\n".join(lines) + "\n"
    sys.stdout.write(txt)
    if a.report:
        with open(a.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
