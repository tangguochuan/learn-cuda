import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from triton.testing import do_bench

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

CURRENT_DIR = Path(__file__).parent

module = load(
    "my_ext",
    sources=[str(CURRENT_DIR / "attention.cpp"), str(CURRENT_DIR / "attention_v6.cu")],
    extra_cuda_cflags=["-lineinfo", "--ptxas-options=-v"],
    verbose=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--nh", type=int, default=8)
    parser.add_argument("--kv", type=int, default=8)  # number of KV heads
    parser.add_argument("--lq", type=int, default=4096)
    parser.add_argument("--lkv", type=int, default=8192)
    args = parser.parse_args()

    bs = args.bs
    nh = args.nh
    kv = args.kv
    lq = args.lq
    lkv = args.lkv
    head_dim = 64

    # add a small offset so that output does not have a mean of zero,
    # which will result in large relative error
    def generate_input(*shape):
        return torch.randn(shape).add(0.5).bfloat16().cuda()

    Q = generate_input(bs, nh, lq, head_dim)
    K = generate_input(bs, kv, lkv, head_dim)
    V = generate_input(bs, kv, lkv, head_dim)

    if args.profile is not None:
        if args.profile == "fa":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V)

        elif args.profile == "cudnn":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                F.scaled_dot_product_attention(Q, K, V)

        elif args.profile == "v6":
            # For GQA reference, expand KV heads to match Q heads
            # if kv != nh:
            #     q_kv_ratio = nh // kv
            #     K_expanded = K.repeat_interleave(q_kv_ratio, dim=1)
            #     V_expanded = V.repeat_interleave(q_kv_ratio, dim=1)
            #     out_ref = F.scaled_dot_product_attention(Q, K_expanded, V_expanded)
            # else:
            out_ref = F.scaled_dot_product_attention(Q, K, V, enable_gqa= True)
            out = module.sdpa_v6(Q, K, V)
            torch.testing.assert_close(out, out_ref)
            print(f"Correctness test passed! (nh={nh}, kv={kv})")

            # Benchmark PyTorch
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                torch_latency_ms = do_bench(lambda: F.scaled_dot_product_attention(Q, K, V, enable_gqa=True), return_mode="median")
            torch_tflops = 4 * bs * nh * lq * lkv * head_dim / torch_latency_ms / 1e9
            print(f"PyTorch:   {torch_latency_ms:.4f} ms, TFLOPS: {torch_tflops:.2f}")

            # Benchmark v6
            latency_ms = do_bench(lambda: module.sdpa_v6(Q, K, V), return_mode="median")
            tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
            print(f"v6:        {latency_ms:.4f} ms, TFLOPS: {tflops:.2f}")
            # Manual attention: Q @ K^T, softmax, softmax @ V
            # For GQA, expand KV heads to match Q heads
            if kv != nh:
                q_kv_ratio = nh // kv
                K_exp = K.repeat_interleave(q_kv_ratio, dim=1)
                V_exp = V.repeat_interleave(q_kv_ratio, dim=1)
            else:
                K_exp = K
                V_exp = V

            def manual_attention(q, k, v):
                scale = 1.0 / (head_dim ** 0.5)
                attn_weights = torch.einsum('bqhd,bkhd->bhqk', q, k) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)
                out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
                return out

            torch.cuda.synchronize()

            # Benchmark: Matmul1 (Q @ K^T)
            # Q: (bs, nh, lq, head_dim), K_exp: (bs, nh, lkv, head_dim)
            # einsum: 'b n q d, b n k d -> b n q k'
            matmul1_latency = do_bench(
                lambda: torch.einsum('bnqd,bnkd->bnqk', Q, K_exp) * (head_dim ** -0.5),
                return_mode="median"
            )
            matmul1_tflops = 2 * bs * nh * lq * lkv * head_dim / matmul1_latency / 1e9

            # Benchmark: Softmax
            softmax_latency = do_bench(
                lambda: F.softmax(torch.einsum('bnqd,bnkd->bnqk', Q, K_exp) * (head_dim ** -0.5), dim=-1),
                return_mode="median"
            )

            # Benchmark: Matmul2 (P @ V)
            matmul2_latency = do_bench(
                lambda: torch.einsum('bnqk,bnkd->bnqd',
                    F.softmax(torch.einsum('bnqd,bnkd->bnqk', Q, K_exp) * (head_dim ** -0.5), dim=-1), V_exp),
                return_mode="median"
            )
            matmul2_tflops = 2 * bs * nh * lq * lkv * head_dim / matmul2_latency / 1e9

            print(f"\n--- Manual attention breakdown ---")
            print(f"Matmul1 (Q@K^T): {matmul1_latency:.4f} ms, TFLOPS: {matmul1_tflops:.2f}")
            print(f"Softmax:         {softmax_latency:.4f} ms")
            print(f"Matmul2 (P@V):   {matmul2_latency:.4f} ms, TFLOPS: {matmul2_tflops:.2f}")
            print(f"Manual total:    {matmul1_latency + softmax_latency + matmul2_latency:.4f} ms")

            # Benchmark PyTorch fused
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                torch_latency_ms = do_bench(lambda: F.scaled_dot_product_attention(Q, K, V, enable_gqa=True), return_mode="median")
            torch_tflops = 4 * bs * nh * lq * lkv * head_dim / torch_latency_ms / 1e9
            print(f"\nPyTorch fused:  {torch_latency_ms:.4f} ms, TFLOPS: {torch_tflops:.2f}")

            # Benchmark v6
            latency_ms = do_bench(lambda: module.sdpa_v6(Q, K, V), return_mode="median")
            tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
            print(f"v6:             {latency_ms:.4f} ms, TFLOPS: {tflops:.2f}")
            print(f"\nvs PyTorch:     {torch_latency_ms / latency_ms:.2f}x")
            print(f"vs Manual:      {(matmul1_latency + softmax_latency + matmul2_latency) / latency_ms:.2f}x")

        else:
            f = getattr(module, f"sdpa_v{args.profile}")
            f(Q, K, V)

        torch.cuda.synchronize()
        return

    SOL_LOOKUP = {
        "NVIDIA GeForce RTX 5090": 209.5,
    }
    sol = SOL_LOOKUP.get(torch.cuda.get_device_name(), 0)

    results = []

    def bench_and_print(f, name, *args):
        # sleep to stabilize thermal
        time.sleep(1)

        latency_ms = do_bench(lambda: f(*args), return_mode="median")
        tflops = 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9
        pct_sol = tflops / sol * 100
        results.append([name, round(latency_ms, 4), round(tflops, 2), round(pct_sol, 2)])

    out_ref = F.scaled_dot_product_attention(Q, K, V)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - FA", Q, K, V)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        bench_and_print(F.scaled_dot_product_attention, "F.sdpa() - CuDNN", Q, K, V)

    if flash_attn_func is not None:
        out = flash_attn_func(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)).transpose(1, 2)
        torch.testing.assert_close(out, out_ref)
        bench_and_print(
            flash_attn_func,
            "flash-attn",
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
        )

    f = module.sdpa_v6
    out = f(Q, K, V)
    torch.testing.assert_close(out, out_ref)
    bench_and_print(f, "v6", Q, K, V)

    df = pd.DataFrame(results, columns=["Kernel", "Latency (ms)", "TFLOPS", "% SOL"])
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
