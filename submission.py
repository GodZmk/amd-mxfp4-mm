#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
FP4 quant + FP4 GEMM: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Quantization is implemented as a custom Triton kernel.
"""
import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _mxfp4_quant_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    M, K,
    stride_xm,
    GROUP_SIZE: tl.constexpr = 32,
    BLOCK_M: tl.constexpr = 1,
):
    row = tl.program_id(0)
    group_id = tl.program_id(1)

    k_start = group_id * GROUP_SIZE
    half_offs = tl.arange(0, GROUP_SIZE // 2)

    # Load even-indexed (lo nibble) and odd-indexed (hi nibble) elements separately
    k_even = k_start + half_offs * 2
    k_odd  = k_start + half_offs * 2 + 1
    mask_e = k_even < K
    mask_o = k_odd  < K

    x_even = tl.load(x_ptr + row * stride_xm + k_even, mask=mask_e, other=0.0).to(tl.float32)
    x_odd  = tl.load(x_ptr + row * stride_xm + k_odd,  mask=mask_o, other=0.0).to(tl.float32)

    # E8M0 scale: match reference _mxfp4_quant_op (aiter/ops/triton/_triton_kernels/quant/quant.py)
    abs_max = tl.maximum(tl.max(tl.abs(x_even), axis=0),
                         tl.max(tl.abs(x_odd),  axis=0))

    # Match reference _mxfp4_quant_op scale exactly via bitwise rounding
    abs_max = tl.maximum(abs_max, 1e-38).to(tl.float32)
    abs_max_int = abs_max.to(tl.int32, bitcast=True)
    abs_max_rounded = ((abs_max_int + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.floor(tl.math.log2(abs_max_rounded)).to(tl.int32) - 2
    scale_e8m0_unbiased = tl.minimum(tl.maximum(scale_e8m0_unbiased, -127), 127)
    e8m0_exp = (scale_e8m0_unbiased + 127).to(tl.uint8)
    quant_scale = tl.math.exp2(-scale_e8m0_unbiased.to(tl.float32))

    tl.store(scale_ptr + row * (K // GROUP_SIZE) + group_id, e8m0_exp)

    # E2M1 bit-manipulation encoding matching ROCm/aiter _mxfp4_quant_op exactly
    # val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << (MBITS_F32 - MBITS_FP4 - 1)) - 1
    # = ((1 - 127) << 23) + (1 << 21) - 1 = -1056964608 + 2097152 - 1 = -1054867457
    # DENORM_MASK_INT = 149 << 23 = 0x4A800000; as float = 2^22 = 4194304.0

    # Even elements -> lo nibbles
    xs_e = x_even * quant_scale
    xs_e_uint = xs_e.to(tl.int32, bitcast=True).to(tl.uint32)
    s_e = xs_e_uint & 0x80000000
    xs_e_pos_uint = xs_e_uint ^ s_e
    xs_e_pos = xs_e_pos_uint.to(tl.float32, bitcast=True)
    sat_e = xs_e_pos >= 6.0
    den_e = xs_e_pos < 1.0
    mant_odd_e = (xs_e_pos_uint >> 22) & 1
    norm_e = ((xs_e_pos_uint.to(tl.int32) + (-1054867457)) + mant_odd_e.to(tl.int32)) >> 22
    norm_e = norm_e.to(tl.uint8)
    den_val_e = (xs_e_pos + 4194304.0).to(tl.int32, bitcast=True) - 0x4A800000
    den_val_e = den_val_e.to(tl.uint8)
    q_e = tl.full(xs_e.shape, 7, dtype=tl.uint8)
    q_e = tl.where(~sat_e, norm_e, q_e)
    q_e = tl.where(den_e, den_val_e, q_e)
    sign_e_lp = (s_e >> 28).to(tl.uint8)
    lo = (q_e | sign_e_lp) & 0xF

    # Odd elements -> hi nibbles
    xs_o = x_odd * quant_scale
    xs_o_uint = xs_o.to(tl.int32, bitcast=True).to(tl.uint32)
    s_o = xs_o_uint & 0x80000000
    xs_o_pos_uint = xs_o_uint ^ s_o
    xs_o_pos = xs_o_pos_uint.to(tl.float32, bitcast=True)
    sat_o = xs_o_pos >= 6.0
    den_o = xs_o_pos < 1.0
    mant_odd_o = (xs_o_pos_uint >> 22) & 1
    norm_o = ((xs_o_pos_uint.to(tl.int32) + (-1054867457)) + mant_odd_o.to(tl.int32)) >> 22
    norm_o = norm_o.to(tl.uint8)
    den_val_o = (xs_o_pos + 4194304.0).to(tl.int32, bitcast=True) - 0x4A800000
    den_val_o = den_val_o.to(tl.uint8)
    q_o = tl.full(xs_o.shape, 7, dtype=tl.uint8)
    q_o = tl.where(~sat_o, norm_o, q_o)
    q_o = tl.where(den_o, den_val_o, q_o)
    sign_o_lp = (s_o >> 28).to(tl.uint8)
    hi = ((q_o | sign_o_lp) & 0xF) << 4

    # Pack two FP4 nibbles per byte: lo nibble = even index, hi nibble = odd index
    packed = lo | hi

    out_base = row * (K // 2) + k_start // 2
    tl.store(out_ptr + out_base + half_offs, packed.to(tl.uint8))


def _triton_mxfp4_quant(x: torch.Tensor):
    """x: [M, K] bf16 -> (fp4_packed [M, K//2] uint8, e8m0_scale [M, K//32] uint8)"""
    M, K = x.shape
    assert K % 32 == 0, "K must be a multiple of 32 for per-1x32 MXFP4 quant"
    x = x.contiguous()

    out = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)
    scale = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)

    grid = (M, K // 32)
    _mxfp4_quant_kernel[grid](
        x, out, scale,
        M, K,
        x.stride(0),
    )
    return out, scale


def custom_kernel(data: input_t) -> output_t:
    import aiter
    from aiter import dtypes
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape

    # Quantize A with the Triton kernel
    A_fp4, A_scale = _triton_mxfp4_quant(A)

    # Shuffle scale to match gemm_a4w4's expected layout
    A_scale_sh = e8m0_shuffle(A_scale.view(torch.uint8))

    out_gemm = aiter.gemm_a4w4(
        A_fp4.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return out_gemm
