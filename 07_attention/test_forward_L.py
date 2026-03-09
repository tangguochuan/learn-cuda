import torch
from pathlib import Path
from torch.utils.cpp_extension import load

CURRENT_DIR = Path(__file__).parent
module = load(
    "attention_bp",
    sources=[
        str(CURRENT_DIR / "attention_v6.cu"),
        str(CURRENT_DIR / "attention_v6_bp.cu"),
        str(CURRENT_DIR / "attention_bp.cpp"),
    ],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)


def expand_kv(K, q_head):
    kv_head = K.shape[1]
    if q_head == kv_head:
        return K
    return K.repeat_interleave(q_head // kv_head, dim=1)


def ref_L(Q_bf16, K_bf16, scale, is_causal=False):
    """Reference logsumexp: L = logsumexp(S * scale, dim=-1)"""
    Q_f = Q_bf16.float()
    K_f = expand_kv(K_bf16, Q_f.shape[1]).float()
    S = torch.einsum("bhqd,bhkd->bhqk", Q_f, K_f) * scale
    if is_causal:
        q_len, kv_len = S.shape[-2], S.shape[-1]
        mask = torch.triu(torch.ones(q_len, kv_len, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    return torch.logsumexp(S, dim=-1)


def run_test(batch_size, q_head, kv_head, q_len, kv_len, head_dim, is_causal, label):
    torch.manual_seed(42)
    scale = 1.0 / (head_dim ** 0.5)

    Q = (torch.randn(batch_size, q_head, q_len, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    K = (torch.randn(batch_size, kv_head, kv_len, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    V = (torch.randn(batch_size, kv_head, kv_len, head_dim, device="cuda") * 0.1).to(torch.bfloat16)

    # Our forward
    O_ours, L_ours = module.sdpa_v6_forward(Q, K, V, is_causal)

    # Reference O
    Q_f = Q.float()
    K_exp_f = expand_kv(K, q_head).float()
    V_exp_f = expand_kv(V, q_head).float()
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q_f, K_exp_f, V_exp_f, scale=scale, is_causal=is_causal
    )

    # Reference L
    L_ref = ref_L(Q, K, scale, is_causal=is_causal)

    # Compare O
    O_diff = (O_ours.float() - O_ref).abs()
    print(f"  O  - max: {O_diff.max().item():.6f}, mean: {O_diff.mean().item():.6f}")

    # Compare L
    L_diff = (L_ours.float() - L_ref).abs()
    print(f"  L  - max: {L_diff.max().item():.6f}, mean: {L_diff.mean().item():.6f}")

    max_err = max(O_diff.max().item(), L_diff.max().item())
    ok = max_err < 0.15
    print(f"  [{label}] max_err={max_err:.6f} {'PASSED' if ok else 'FAILED'}")
    assert ok, f"max error too large: {max_err}"


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: MHA, no causal, 64x64")
    run_test(1, 1, 1, 64, 64, 64, False, "MHA")

    print("\n" + "=" * 60)
    print("Test 2: MHA, causal, 64x64")
    run_test(1, 1, 1, 64, 64, 64, True, "MHA-causal")

    print("\n" + "=" * 60)
    print("Test 3: MHA, no causal, 128x128")
    run_test(1, 1, 1, 128, 128, 64, False, "MHA-128")

    print("\n" + "=" * 60)
    print("Test 4: GQA, no causal, q_head=4, kv_head=1")
    run_test(1, 4, 1, 64, 64, 64, False, "GQA")

    print("\n" + "=" * 60)
    print("Test 5: GQA, causal, 128x128, q_head=4, kv_head=2")
    run_test(1, 4, 2, 128, 128, 64, True, "GQA-causal")

    print("\n" + "=" * 60)
    print("Test 6: batch=2, GQA causal")
    run_test(2, 4, 2, 128, 128, 64, True, "batch-GQA-causal")

    print("\nAll tests passed!")
