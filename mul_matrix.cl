__kernel void mul_matrix(__global float const* const a,
                    __global float const* const b,
                    __global float* const c,
                    uint const n,
                    uint const m,
                    uint const k) {
    uint const global_i = get_global_id(1) * ELEMS_PER_THREAD;
    uint const global_j = get_global_id(0);
    uint const tile_i = get_local_id(1) * ELEMS_PER_THREAD;
    uint const tile_j = get_local_id(0);

    local float A_sub[TILE_SIZE][TILE_SIZE];
    local float B_sub[TILE_SIZE][TILE_SIZE];

    float thread_res[ELEMS_PER_THREAD];
    for (uint i = 0; i < ELEMS_PER_THREAD; i++) {
        thread_res[i] = 0;
    }

    uint const tile_cnt = m / TILE_SIZE;
    for (uint tile_id = 0; tile_id < tile_cnt; ++tile_id) {

        for (uint shift = 0; shift < ELEMS_PER_THREAD; shift++) {
            uint tiled_col = tile_id * TILE_SIZE + tile_j + shift;
            uint tiled_row = tile_id * TILE_SIZE + tile_i;

            A_sub[tile_i + shift][tile_j] = a[(global_i + shift) * m + tiled_col];
            B_sub[tile_i + shift][tile_j] = b[tiled_row * k + global_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint shift = 0; shift < ELEMS_PER_THREAD; shift++) {
            for (uint t = 0; t < TILE_SIZE; ++t) {
                thread_res[shift] += A_sub[tile_i + shift][t] * B_sub[t][tile_j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint shift = 0; shift < ELEMS_PER_THREAD; shift++) {
        c[(global_i + shift) * k + global_j] = thread_res[shift];
    }
}
