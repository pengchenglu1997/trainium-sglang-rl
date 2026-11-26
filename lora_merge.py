"""NKI kernel: merge a low-rank LoRA delta into a resident bf16 base weight with ONE rounding.

    out[IN, OUT] = round_bf16( float32(base[IN, OUT]) + a_stack[K, IN].T @ bt[K, OUT] )

Layouts (all input-major, as the sglang-neuron dense Qwen3 graph stores its per-rank shards):
  qkv_weight [H, q+2kv], o_weight [heads*d, H], gate_up_weight [H, 2*I/tp], down_weight [I/tp, H]
  are all [IN, OUT]; this kernel supports exactly that layout and no [OUT, IN] layout.
  a_stack [K, IN]  : the LoRA A factors of the c fused segments stacked on K (K = c*r, c in 1..3)
  bt      [K, OUT] : the LoRA B factors transposed, with the scale (alpha/r) already folded in
K is the partition axis of both matmul operands, so no transpose is needed on the device.

Numerics:
  * bf16 factors: every bf16 x bf16 product is exact in fp32 (8+8 bit mantissas); the K-sum runs in
    the tensor engine's fp32 accumulator; one fp32 add of the base tile and one cast to bf16.
  * fp32 factors: nc_matmul accepts fp32 stationary and moving when BOTH are fp32. In the simulator
    the result is the numpy float32 matmul. Real tensor-engine fp32 precision is a hardware question.
  * "split" mode (split_hi_lo): each fp32 factor x becomes bf16 hi = bf16(x) and bf16 lo = bf16(x - hi),
    and the kernel runs plain bf16 on the stacked factors [A_hi; A_hi; A_lo] and [B_hi; B_lo; B_hi]
    (K' = 3K, up to 144, done as two accumulating matmuls). This keeps ~16 mantissa bits per factor.
    It is an alternative to fp32 operands if fp32 matmul turns out unusable or slow on hardware.

Tiling: loops over IN tiles of at most 128 rows (stationary free dim = PSUM partitions) and OUT tiles of
at most 512 columns (one PSUM bank of fp32). The last tile in each direction may be smaller, so any
positive IN and OUT are accepted by the kernel; IN % 128 == 0 and OUT % 512 == 0 is the efficient case.
a_stack and bt are loaded once into SBUF and stay resident across the tile loop.

LNC: with LNC=2 the kernel body runs once per program (nl.program_id(0) = 0, 1; nl.num_programs(0) = 2).
IN tiles are dealt round-robin to the programs; both write disjoint rows of the same shared-HBM output.
"""
import ml_dtypes
import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

BF16 = ml_dtypes.bfloat16

P_MAX = 128          # partitions; max rows of one IN tile and max K per matmul
N_MAX = 512          # fp32 elements of one PSUM bank; max OUT columns per matmul
K_MAX = 256          # K accepted by this kernel (two accumulating matmuls); real use is <= 48 (<= 144 split)
RESIDENT_BYTES_PER_PARTITION = 128 * 1024   # budget for the resident factors (SBUF is 224 KiB/partition on trn2)


def lora_merge_reference(base, a_stack, bt):
    """Pure numpy reference: float32 accumulate, one cast to bf16.  Shapes as in the module doc."""
    base = np.asarray(base)
    prod = np.asarray(a_stack, dtype=np.float32).T @ np.asarray(bt, dtype=np.float32)
    return (base.astype(np.float32) + prod).astype(BF16)


def shape_guard(base, a_stack, bt):
    """Return None if the kernel accepts these arguments, else a string saying why not."""
    if base.ndim != 2 or a_stack.ndim != 2 or bt.ndim != 2:
        return "base, a_stack and bt must all be 2-D"
    in_dim, out_dim = base.shape
    k, in_a = a_stack.shape
    k_b, out_b = bt.shape
    if base.dtype != BF16:
        return f"base must be bfloat16, got {base.dtype}"
    if a_stack.dtype != bt.dtype:
        return f"a_stack and bt must have the same dtype (fp32 cannot be mixed with bf16 in nc_matmul), got {a_stack.dtype} and {bt.dtype}"
    if a_stack.dtype not in (np.dtype(np.float32), np.dtype(BF16)):
        return f"factor dtype must be float32 or bfloat16, got {a_stack.dtype}"
    if in_a != in_dim or out_b != out_dim or k != k_b:
        return f"shape mismatch: base {base.shape}, a_stack {a_stack.shape}, bt {bt.shape}"
    if not (1 <= k <= K_MAX):
        return f"K={k} outside 1..{K_MAX}"
    if in_dim < 1 or out_dim < 1:
        return "empty base"
    n_chunks = -(-k // P_MAX)
    resident = n_chunks * (in_dim + out_dim) * a_stack.dtype.itemsize
    if resident > RESIDENT_BYTES_PER_PARTITION:
        return f"resident factors need {resident} bytes per partition, budget {RESIDENT_BYTES_PER_PARTITION}"
    return None


def efficient_shape(in_dim, out_dim):
    """True when no partial tile exists (full 128 x 512 tiles only; 512 divides OUT)."""
    return in_dim % P_MAX == 0 and out_dim % N_MAX == 0


K_PRODUCTION = {"fp32": 128, "split": 144}


def prepare_factors(a_stack, bt, mode="fp32"):
    """The one place that fixes the factor dtype at the builder -> kernel interface.

    The factor builders return float32. 'fp32' passes float32 through (native fp32 matmul);
    'split' returns the bf16 hi/lo stack (K' = 3K). A plain cast of float32 factors to bf16 is never
    done here: it loses half of the mantissa of every factor and was measured to be far less accurate.
    """
    a = np.ascontiguousarray(np.asarray(a_stack))
    b = np.ascontiguousarray(np.asarray(bt))
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise TypeError(f"factors must arrive as float32 from the builders, got {a.dtype} and {b.dtype}")
    if mode == "fp32":
        return a, b
    if mode == "split":
        return split_hi_lo(a, b)
    raise ValueError(f"unknown mode {mode!r}")


def production_guard(base, a_stack, bt, mode="fp32"):
    """Guard for the serving path: only what the tests cover on full tiles. None = use the kernel;
    a string = why not (the caller then takes the host path).

    Stricter than shape_guard on purpose: partial tiles and K above one matmul chunk pass in the
    simulator but have not been compiled or run on hardware.
    """
    why = shape_guard(base, a_stack, bt)
    if why is not None:
        return why
    in_dim, out_dim = base.shape
    if not efficient_shape(in_dim, out_dim):
        return f"shard {in_dim}x{out_dim} is not a multiple of {P_MAX}x{N_MAX}"
    k = a_stack.shape[0]
    if k > K_PRODUCTION[mode]:
        return f"K={k} above {K_PRODUCTION[mode]} for mode {mode}"
    want = np.dtype(np.float32) if mode == "fp32" else np.dtype(BF16)
    if a_stack.dtype != want:
        return f"mode {mode} needs {want} factors from prepare_factors, got {a_stack.dtype}"
    return None


def split_hi_lo(a_stack, bt):
    """Host helper for 'split' mode.  fp32 factors -> bf16 stacked factors with K' = 3K.

    A_hi*B_hi + A_hi*B_lo + A_lo*B_hi  with  x = hi + lo + O(2^-17 x).
    Returns (a_cat [3K, IN] bf16, bt_cat [3K, OUT] bf16).
    """
    a = np.asarray(a_stack, dtype=np.float32)
    b = np.asarray(bt, dtype=np.float32)
    a_hi = a.astype(BF16)
    a_lo = (a - a_hi.astype(np.float32)).astype(BF16)
    b_hi = b.astype(BF16)
    b_lo = (b - b_hi.astype(np.float32)).astype(BF16)
    return (np.concatenate([a_hi, a_hi, a_lo], axis=0),
            np.concatenate([b_hi, b_lo, b_hi], axis=0))


@nki.jit
def lora_merge(base, a_stack, bt):
    in_dim, out_dim = base.shape
    k = a_stack.shape[0]
    assert a_stack.shape[1] == in_dim and bt.shape[0] == k and bt.shape[1] == out_dim
    assert 1 <= k <= K_MAX

    out = nl.ndarray((in_dim, out_dim), dtype=base.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    nprog = nl.num_programs(0)

    # resident factors: one SBUF tile per chunk of at most 128 rows of K
    n_chunks = (k + P_MAX - 1) // P_MAX
    a_tiles = []
    b_tiles = []
    for c in range(n_chunks):
        k0 = c * P_MAX
        kc = min(P_MAX, k - k0)
        a_sb = nl.ndarray((kc, in_dim), dtype=a_stack.dtype, buffer=nl.sbuf)
        b_sb = nl.ndarray((kc, out_dim), dtype=bt.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=a_stack[k0:k0 + kc, 0:in_dim])
        nisa.dma_copy(dst=b_sb, src=bt[k0:k0 + kc, 0:out_dim])
        a_tiles.append(a_sb)
        b_tiles.append(b_sb)

    n_in_tiles = (in_dim + P_MAX - 1) // P_MAX
    n_out_tiles = (out_dim + N_MAX - 1) // N_MAX
    for ti in range(n_in_tiles):
        if ti % nprog != pid:
            continue
        i0 = ti * P_MAX
        m = min(P_MAX, in_dim - i0)
        for tj in range(n_out_tiles):
            o0 = tj * N_MAX
            n = min(N_MAX, out_dim - o0)
            w_sb = nl.ndarray((m, n), dtype=base.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb, src=base[i0:i0 + m, o0:o0 + n])
            ps = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.psum)
            for c in range(n_chunks):
                nisa.nc_matmul(dst=ps,
                               stationary=a_tiles[c][:, i0:i0 + m],
                               moving=b_tiles[c][:, o0:o0 + n],
                               accumulate=(c > 0))
            res = nl.ndarray((m, n), dtype=base.dtype, buffer=nl.sbuf)
            # fp32 add of bf16 base and fp32 PSUM, one cast to bf16
            nisa.tensor_tensor(dst=res, data1=w_sb, data2=ps, op=nl.add)
            nisa.dma_copy(dst=out[i0:i0 + m, o0:o0 + n], src=res)
    return out
