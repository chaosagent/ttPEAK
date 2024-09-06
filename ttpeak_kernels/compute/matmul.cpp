#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"

#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t in0 = tt::CB::c_in0;
    constexpr uint32_t in1 = tt::CB::c_in1;
    constexpr uint32_t out0 = tt::CB::c_out0;
#define BLOCK 4
#define BLOCK_K 4
#ifndef BLOCK
    mm_init(in0, in1, out0);

    for(uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(in0, 1);
        cb_wait_front(in1, 1);
        // Don't care about the output. discard it.
        acquire_dst(tt::DstMode::Half);
        matmul_tiles(in0, in1, 0, 0, 0, false);

        release_dst(tt::DstMode::Half);
        cb_pop_front(in0, 1);
        cb_pop_front(in1, 1);
    }
#else
    mm_block_init(in0, in1, out0, 0, BLOCK, BLOCK, BLOCK_K);
    for(uint32_t i = 0; i < n_tiles/BLOCK/BLOCK_K; i++) {
        DeviceZoneScopedN("MATMUL-INNER");
        cb_wait_front(in0, BLOCK * BLOCK_K);
        cb_wait_front(in1, BLOCK * BLOCK_K);

        acquire_dst(tt::DstMode::Half);
#pragma unroll 4
        for (int j = 0; j < BLOCK_K; j++) {
            matmul_block(in0, in1, 0, 0, 0, false, BLOCK, BLOCK, BLOCK_K);
        }
        release_dst(tt::DstMode::Half);

        cb_pop_front(in0, BLOCK * BLOCK_K);
        cb_pop_front(in1, BLOCK * BLOCK_K);
    }
#endif

}
}

