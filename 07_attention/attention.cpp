#include <torch/extension.h>
#include <cuda_bf16.h>

// Forward declaration with full signature
void attention_v6(
  const nv_bfloat16 *Q,
  const nv_bfloat16 *K,
  const nv_bfloat16 *V,
  nv_bfloat16 *O,
  int bs,
  int q_head,
  int kv_head,
  int len_q,
  int len_kv,
  int dim,
  const nv_bfloat16 *atten_mask,
  bool is_causal,
  float dropout_p,
  bool is_gqa);

at::Tensor sdpa_v6(
  const at::Tensor& Q,
  const at::Tensor& K,
  const at::Tensor& V) {

  const int bs = Q.size(0);
  const int q_head = Q.size(1);
  const int len_q = Q.size(2);
  const int kv_head = K.size(1);
  const int len_kv = K.size(2);
  const int dim = Q.size(3);

  at::Tensor O = at::empty_like(Q);

  auto Q_ptr = reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<const nv_bfloat16 *>(K.data_ptr());
  auto V_ptr = reinterpret_cast<const nv_bfloat16 *>(V.data_ptr());
  auto O_ptr = reinterpret_cast<nv_bfloat16 *>(O.data_ptr());

  attention_v6(Q_ptr, K_ptr, V_ptr, O_ptr, bs, q_head, kv_head, len_q, len_kv, dim, nullptr, false, 0.0f, false);

  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sdpa_v6", &sdpa_v6);
}
