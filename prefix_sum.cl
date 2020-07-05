__kernel void prefix_sum(__global float const *const a,
                         __global float *const b,
                         __global float *const tmp,
                         ulong const n) {
  const uint j = get_local_id(0);
  const uint sz = get_local_size(0);
  const ulong log2_n = log2((float)n);

  tmp[j] = a[j];

  uint shift = 1;
  for (uint i = 0; i < log2_n + 1; ++i) {
    barrier(CLK_LOCAL_MEM_FENCE);
    uint new_shift = shift ^ 1;
    if ((1u << i) > j) {
      tmp[j + n * shift] = tmp[j + n * new_shift];
    } else {
      tmp[j + n * shift] = tmp[j + n * new_shift] + tmp[j + n * new_shift - (1u << i)];
    }
    shift = new_shift;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint i = 0; i < n; ++i) {
    b[i] = tmp[i + n * (shift ^ 1)];
  }
}
