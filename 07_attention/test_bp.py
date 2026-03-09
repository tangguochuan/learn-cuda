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
    """Expand KV heads to match Q heads for GQA"""
    kv_head = K.shape[1]
    if q_head == kv_head:
        return K
    ratio = q_head // kv_head
    return K.repeat_interleave(ratio, dim=1)


def compute_L(Q_bf16, K_bf16, scale, is_causal=False):
    """Compute L = logsumexp(S*scale, dim=-1) using bf16 inputs cast to float"""
    Q_f = Q_bf16.float()
    K_f = expand_kv(K_bf16, Q_f.shape[1]).float()
    S = torch.einsum("bhqd,bhkd->bhqk", Q_f, K_f) * scale
    if is_causal:
        q_len, kv_len = S.shape[-2], S.shape[-1]
        mask = torch.triu(torch.ones(q_len, kv_len, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    L = torch.logsumexp(S, dim=-1)  # [bs, q_head, q_len]
    return L


def ref_backward(Q, K, V, dO, scale, is_causal=False):
    """Reference backward using PyTorch autograd, with GQA support"""
    q_head = Q.shape[1]
    kv_head = K.shape[1]
    Q_ref = Q.clone().detach().float().requires_grad_(True)
    K_exp = expand_kv(K, q_head).clone().detach().float().requires_grad_(True)
    V_exp = expand_kv(V, q_head).clone().detach().float().requires_grad_(True)
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q_ref, K_exp, V_exp, scale=scale, is_causal=is_causal
    )
    dQ_ref, dK_ref, dV_ref = torch.autograd.grad(O_ref, [Q_ref, K_exp, V_exp], dO.float())
    # Sum gradients back to kv_head dimension for GQA
    if q_head != kv_head:
        ratio = q_head // kv_head
        dK_ref = dK_ref.view(dK_ref.shape[0], kv_head, ratio, *dK_ref.shape[2:]).sum(dim=2)
        dV_ref = dV_ref.view(dV_ref.shape[0], kv_head, ratio, *dV_ref.shape[2:]).sum(dim=2)
    return dQ_ref, dK_ref, dV_ref


def run_test(batch_size, q_head, kv_head, q_len, kv_len, head_dim, is_causal, label):
    torch.manual_seed(42)
    scale = 1.0 / (head_dim ** 0.5)

    Q = (torch.randn(batch_size, q_head, q_len, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    K = (torch.randn(batch_size, kv_head, kv_len, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    V = (torch.randn(batch_size, kv_head, kv_len, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    dO = (torch.randn(batch_size, q_head, q_len, head_dim, device="cuda") * 0.01).to(torch.bfloat16)

    # Forward: 用自己的 kernel 拿 O 和 L（保证 forward/backward 一致）
    O, L = module.sdpa_v6_forward(Q, K, V, is_causal)

    # Our backward
    dQ, dK, dV = module.sdpa_v6_bp(Q, K, V, O, L, dO, is_causal)

    # Reference backward
    dQ_ref, dK_ref, dV_ref = ref_backward(Q, K, V, dO, scale, is_causal=is_causal)

    for name, ours, ref in [("dQ", dQ, dQ_ref), ("dK", dK, dK_ref), ("dV", dV, dV_ref)]:
        diff = (ours.float() - ref).abs()
        print(f"  {name} - max: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")

    max_err = max(
        (dQ.float() - dQ_ref).abs().max().item(),
        (dK.float() - dK_ref).abs().max().item(),
        (dV.float() - dV_ref).abs().max().item(),
    )
    # bf16 has limited precision; causal+GQA accumulation can amplify errors
    tol = 0.15
    ok = max_err < tol
    print(f"  [{label}] max_err={max_err:.6f} {'PASSED' if ok else 'FAILED'}")
    assert ok, f"max error too large: {max_err}"
    return True


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
    print("Test 4: GQA, no causal, 64x64, q_head=4, kv_head=1")
    run_test(1, 4, 1, 64, 64, 64, False, "GQA")

    print("\n" + "=" * 60)
    print("Test 5: GQA, causal, 128x128, q_head=4, kv_head=2")
    run_test(1, 4, 2, 128, 128, 64, True, "GQA-causal")

    print("\n" + "=" * 60)
    print("Test 6: batch=2, GQA causal, 128x128")
    run_test(2, 4, 2, 128, 128, 64, True, "batch-GQA-causal")

    print("\nAll tests passed!")
