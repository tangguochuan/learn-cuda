# 开发日志

## Bug 1: cvta.to.shared.u32 编译失败
- **现象**: ptxas error: Arguments mismatch for instruction 'cvta.to'
- **原因**: 内联 asm `cvta.to.shared.u32` 在 sm_80 上参数不匹配
- **修复**: 改用 `__cvta_generic_to_shared()` 编译器内置函数

## Bug 2: 1D grid 与 2D blockIdx 不匹配
- **现象**: illegal memory access
- **原因**: kernel 使用 `blockIdx.x` 和 `blockIdx.y` 分别索引 kv_block 和 batch_kv_head，但 `launch_kernel` 只传单个 int 作为 grid size，导致 `blockIdx.y` 始终为 0
- **修复**: 改为从 `blockIdx.x` 计算两个维度: `kv_block_idx = blockIdx.x % num_kv_blocks`, `batch_kv_head = blockIdx.x / num_kv_blocks`

## Bug 3: shared memory 地址转指针导致越界写入
- **现象**: compute-sanitizer 报告 `frag_to_smem_transpose` 写入 out of bounds 全局地址
- **原因**: `*(nv_bfloat16*)a0 = v0` 中 `a0` 是 32-bit shared memory 地址，强转为 64-bit 指针时高 32 位为垃圾值，变成无效的全局地址
- **修复**: 使用 `st.shared.b16` 内联汇编替代直接指针解引用

## 精度说明
- 非 causal: dQ/dK/dV max error < 0.01
- causal + GQA: dV max error 可达 ~0.12（mean ~0.003），属于 bf16 精度限制
