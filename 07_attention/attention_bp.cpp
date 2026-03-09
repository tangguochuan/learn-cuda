#include <torch/extension.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>

void attention_v6_backward(
  const nv_bfloat16 *Q,
  const nv_bfloat16 *K,
  const nv_bfloat16 *V,
  const nv_bfloat16 *O,
  const nv_bfloat16 *L,
  const nv_bfloat16 *dO,
  nv_bfloat16 *dQ,
  nv_bfloat16 *dK,
  nv_bfloat16 *dV,
  int batch_size,
  int q_head,
  int kv_head,
  int q_len,
  int kv_len,
  int head_dim,
  bool is_causal);

std::vector<at::Tensor> sdpa_v6_bp(
  const at::Tensor& Q,
  const at::Tensor& K,
  const at::Tensor& V,
  const at::Tensor& O,
  const at::Tensor& L,
  const at::Tensor& dO,
  bool is_causal) {

  const int bs = Q.size(0);
  const int q_head = Q.size(1);
  const int len_q = Q.size(2);
  const int kv_head = K.size(1);
  const int len_kv = K.size(2);
  const int dim = Q.size(3);

  at::Tensor dQ = at::zeros_like(Q);
  at::Tensor dK = at::zeros_like(K);
  at::Tensor dV = at::zeros_like(V);

  attention_v6_backward(
    reinterpret_cast<const nv_bfloat16 *>(Q.data_ptr()),
    reinterpret_cast<const nv_bfloat16 *>(K.data_ptr()),
    reinterpret_cast<const nv_bfloat16 *>(V.data_ptr()),
    reinterpret_cast<const nv_bfloat16 *>(O.data_ptr()),
    reinterpret_cast<const nv_bfloat16 *>(L.data_ptr()),
    reinterpret_cast<const nv_bfloat16 *>(dO.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(dQ.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(dK.data_ptr()),
    reinterpret_cast<nv_bfloat16 *>(dV.data_ptr()),
    bs, q_head, kv_head, len_q, len_kv, dim, is_causal);

  return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sdpa_v6_bp", &sdpa_v6_bp, "sdpa_v6_backward",
    py::arg("Q"), py::arg("K"), py::arg("V"),
    py::arg("O"), py::arg("L"), py::arg("dO"),
    py::arg("is_causal") = false);
}
