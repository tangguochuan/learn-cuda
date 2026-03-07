#include <cuda_bf16.h>
#include "common.h"

// no gqa + no causal
template <int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__global__ void flash_atten_bakward_1(
const nv_bfloat16 *Q,
    const nv_bfloat16 *K,
    const nv_bfloat16 *V,
    const nv_bfloat16 *O,
    const nv_bfloat16 *L,
    const nv_bfloat16 *dO,
    nv_bfloat16 *dQ,
    nv_bfloat16 *d_temp_K, // [batch_size, q_head, kv_len, dim]
    nv_bfloat16 *d_temp_V, // [batch_size, q_head, kv_len, dim]
    int bs,
    int q_head,
    int kv_head,
    int q_len,
    int kv_len,
    int head_dim,
    int q_kv_ratio = 1
){
    //basic information
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_kv_blocks = cdiv(kv_len, BLOCK_KV);
    const int bs_id = bid / num_kv_blocks;
    const int batch_id = bs_id / q_head;
    const int q_head_id = bs_id % q_head;
    const int kv_head_id = q_head / q_kv_ratio;
    const int kv_block_id = bid % num_kv_blocks;
    const int WARP_KV = BLOCK_KV / NUM_WARPS;
    //当前thread block要处理的初始位置
    Q += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM); // process [q_len, DIM]
    K += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM); // process [BLOCK_KV, DIM]
    V += (batch_id * kv_head * kv_len * DIM + kv_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM); // process [BLOCK_KV, DIM]
    O += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM); // process [q_len, DIM]
    dO += (batch_id * q_head * q_len * DIM + q_head_id * q_len * DIM); // process [q_len, DIM]
    d_temp_K += (batch_id * q_head * kv_len * DIM + q_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM); // process [BLOCK_KV, DIM]
    d_temp_V += (batch_id * q_head * kv_len * DIM + q_head_id * kv_len * DIM + kv_block_id * BLOCK_KV * DIM); // process [BLOCK_KV, DIM]
    L += (batch_id * q_head * q_len + q_head_id * q_len); //process [q_len,]

    // Load K, V HBM -> SRAM [BLOCK_KV, DIM]
    // initialize dK,dV: [BLOCK_KV, DIM]
    extern __shared__ nv_bfloat16 smem[];
    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t Q_smem = K_smem;
    const uint32_t V_smem = Q_smem + max(BLOCK_KV, BLOCK_Q) * DIM * sizeof(nv_bfloat16);
    const uint32_t L_smem = V_smem + BLOCK_KV * DIM * sizeof(nv_bfloat16);
    //for ldmatrix: 计算每个线程要load的行和列，并且要swizzle一下
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        // A tile
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    {
        // B tile trans
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)> (V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
    }
    const int kv_valid_rows = min(BLOCK_KV, kv_len - kv_block_id * BLOCK_KV);
    if(kv_valid_rows == BLOCK_KV){
        gloabl_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid);
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid);
    }
    else{
        gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(K_smem, K, DIM, tid, kv_valid_rows);
        gloabl_to_shared_swizzle_padded<BLOCK_KV, DIM, TB_SIZE>(V_smem, V, DIM, tid, kv_valid_rows);
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // K,V,dK, dV: shared -> registers
    const int MMA_M = 16;
    const int MMA_K = 16;
    const int MMA_N = 8;
    uint32_t K_rmem[WARP_KV / MMA_N][DIM / MMA_K][2];
    // TBD: modify V_rmem
    uint32_t V_rmem[WARP_KV / MMA_N][DIM / MMA_K][2];

    // Load K registers: shared -> register
    for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_N; mma_id_kv++){
        for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
            uint32_t addr = K_smem_thread;
            addr += mma_id_kv * MMA_N * DIM *sizeof(nv_bfloat16);
            addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
            ldmatrix_x2(
                K_rmem[mma_id_kv][mma_id_d], addr
            );
        }
    }
    uint32_t Q_rmem[BLOCK_Q / MMA_M][DIM / MMA_K][4];
    // uint32_t dK_rmem;
    // uint32_t dV_rmem
    int kv_start = kv_block_id * BLOCK_KV + WARP_KV * warp_id;
    for(int off_q = 0; off_q < q_len; off_q+= BLOCK_Q){
        // dK，dV不用在smem中分配，直接分配在registers中
        float dK_rmem[WARP_KV / MMA_N][DIM / MMA_K][2] = {};
        float dV_rmem[WARP_KV / MMA_N][DIM / MMA_K][2] = {};

        //为S，P, L分配 registers
        float S_rmem[BLOCK_Q / MMA_M][WARP_KV / MMA_N][4] = {};
        float P_rmem[BLOCK_Q / MMA_M][WARP_KV / MMA_N][4];
        float L_rmem[BLOCK_Q / MMA_M][2];
        // Load Q: [BLOCK_Q, DIM] from HBM -> shared
        int q_valid_rows = min(BLOCK_Q, q_len - off_q);
        if(q_valid_rows == BLOCK_Q){
            global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid);
        }
        else{
            gloabl_to_shared_swizzle_padded<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, q_valid_rows);
        }
        // Load L: [BLOCK_Q] from HBM -> shared
        for(int i = tid; i < BLOCK_Q; i += TB_SIZE){
            int idx = i + off_q;
            if(idx < q_len){
                asm volatile("cp.async.cg.shared.global [%0], [%1], 4;"
                : "r"(&L_smem[i])
                : "l"(&L[idx]));
            }
        }
        
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        // Load Q: shared -> registers
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
            for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
                uint32_t addr = Q_smem_thread;
                addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);
                addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);
                ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
            }
        }

        // recompute: S = Q @ K.T
        for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q ++){
            for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_N; mma_id_kv++){
                for(int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++){
                    mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_q][mma_id_kv]);
                }
            }
        }
        // apply padding mask to KV and get P (overlap S)
            for(int mma_id_q = 0; mma_id_q < BLOCK_Q / MMA_M; mma_id_q++){
                for(int mma_id_kv = 0; mma_id_kv < WARP_KV / MMA_N; mma_id_kv++){
                    for(int i = 0; i < 4; i++){
                        int mma_col = (lane_id % 4) * 2 + (i & 0x1);
                        int kv_idx = kv_start + mma_id_kv * MMA_N + mma_col;
                        if(kv_idx >= kv_len){
                            S_rmem[mma_id_q][mma_id_kv][i] = -FLT_MAX;
                        }
                        //row_id within BLOCK_Q
                        int row_id = mma_id_q * MMA_M + (lane_id >> 2) + 8 * (i >= 2);
                        // 为了节省现存，S和P复用
                        S_rmem[mma_id_q][mma_id_kv][i] = __expf(S_rmem[mma_id_q][mma_id_kv][i] - L_smem[row_id]);
                    }
                }
            }

    }
}

/**
 * @brief flash attention backward的入口函数
 * @param[in] Q (bs, q_head, q_len, head_dim)
 * @param[in] K (bs, kv_head, kv_len, head_dim)
 * @param[in] V (bs, kv_head, kv_len, head_dim)
 * @param[in] O (bs, q_head, q_len, head_dim)
 * @param[in] L (bs, q_head, q_len)
 * @param[in] dO same shape as O
 */
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
    bool is_causal
){
    ASSERT_NOT_NULL(Q, K, V, O, dO, dQ, dK, dV);
    if(q_head % kv_head){
        ERROR("q_head is %d, kv_head is %d, q_head % kv_head not equal 0", q_head,kv_head);
    }
    // 四种情况
    // 1) 非gqa + 非causal
    // 2) gqa + 非causal
    // 3) 非gqa + causal
    // 4) gqa + causal
    const int KV_BLOCKS = 64;
    const int Q_BLOCKS = 64;
    const int kv_blocks = batch_size * q_head * cdiv(kv_len, KV_BLOCKS);
    const int TB_SIZE = 128;
    const int WARP_SIZE = 32;
    const int NUM_WARPS = 4;
    const int DIM = 64;
    if(is_causal == false && q_head == kv_head){

    }
}