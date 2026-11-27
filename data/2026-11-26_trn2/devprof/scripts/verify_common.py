"""Shared loaders for the s4 device-profile analysis (adapted from s3 analysis_devprof/scripts/verify_common.py).
Profiles are addressed by tag (q06_rel_nd, ...); parquet under devprof/parquet/<tag>/<session>/*.parquet."""
import glob, os, sys
import pandas as pd, numpy as np
import pyarrow.parquet as pq
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = r"<local-path>/results_s4_2026-11-26\devprof"
BASE = ROOT + r"\parquet"
OUT = ROOT + r"\out"
SCR = r"<local-path>"
TAGS = ["q06_rel_nd", "q06_main_nd", "q06_topk_nd", "q06_lmh_off_nd", "q06_lmh_on_nd", "q8_rel_nd", "q8_main_nd",
        "q06_topk_kon", "q06_topk_koff"]


def cache_path(p):
    return os.path.join(SCR, "s4instr_%s.pkl" % p)


def sess(p):
    g = glob.glob(os.path.join(BASE, p, "i-*"))
    assert len(g) == 1, (p, g)
    return g[0]


def tbl(p, name, cols=None):
    f = os.path.join(sess(p), name + ".parquet")
    return pd.read_parquet(f, columns=cols)


def cols_of(p, name):
    return pq.ParquetFile(os.path.join(sess(p), name + ".parquet")).schema_arrow.names


def execs(p):
    return tbl(p, "ExecutionInfo")


INSTR_COLS = ["execution_index", "start_ts", "end_ts", "duration_ns", "engine", "execution_unit", "pcore_idx", "opcode",
              "compiler_pc", "layer", "tensor_instruction_type", "spill_save_bytes", "spill_reload_bytes",
              "flops_transpose", "raw_flops", "nki_source_location", "hlo_name", "evt_wait_time_ns"]


def instr(p):
    c = cache_path(p)
    if os.path.exists(c):
        return pd.read_pickle(c)
    have = set(cols_of(p, "Instruction"))
    d = tbl(p, "Instruction", [x for x in INSTR_COLS if x in have])

    def _f(x):
        if x is None or isinstance(x, float):
            return None
        if isinstance(x, str):
            return x
        x = list(x)
        return x[0] if x else None
    d["nki_source_location"] = d["nki_source_location"].astype(object).map(_f)
    d.to_pickle(c)
    return d


def union_len(starts, ends):
    if len(starts) == 0:
        return 0.0
    o = np.argsort(starts, kind="stable")
    s = np.asarray(starts, dtype=np.int64)[o]
    e = np.asarray(ends, dtype=np.int64)[o]
    em = np.maximum.accumulate(e)
    new = np.empty(len(s), bool)
    new[0] = True
    new[1:] = s[1:] > em[:-1]
    idx = np.flatnonzero(new)
    return float((np.maximum.reduceat(e, idx) - s[idx]).sum())
