//! Block-tiled matrix multiplication with SIMD dot products.
//!
//! This follows ggml's convention: `C = mul_mat(A, B)` means `C[i,j] = dot(A[i,:], B[j,:])`.

const BLOCK_M: usize = 16;
const BLOCK_N: usize = 16;

/// Compute C = A × B^T using block-tiling with SIMD dot products.
///
/// # Arguments
///
/// * `a` - Matrix A with shape `[m, k]`
/// * `b` - Matrix B with shape `[n, k]`
/// * `c` - Output matrix with shape `[m, n]` (must be pre-allocated)
/// * `n_threads` - Number of threads for parallel execution
/// * `min_parallel_rows` - Minimum rows of A before parallel dispatch is worthwhile.
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_threads: usize,
    min_parallel_rows: usize,
) {
    assert_eq!(a.len(), m * k, "A must have shape [{m}, {k}]");
    assert_eq!(b.len(), n * k, "B must have shape [{n}, {k}]");
    assert_eq!(c.len(), m * n, "C must have shape [{m}, {n}]");

    let n_threads = if n_threads == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    } else {
        n_threads
    };

    // Don't parallelize tiny matmuls
    if n_threads <= 1 || m < min_parallel_rows {
        // Single-threaded
        matmul_f32_block(a, b, c, n, k, 0, m, 0, n);
        return;
    }

    // Parallel: split rows of A across threads
    let rows_per_thread = m.div_ceil(n_threads);

    // Build row ranges
    let mut ranges = Vec::new();
    for t in 0..n_threads {
        let i_start = (t * rows_per_thread).min(m);
        let i_end = ((t + 1) * rows_per_thread).min(m);
        if i_start < i_end {
            ranges.push((i_start, i_end));
        }
    }

    // Use scoped threads with raw pointers for non-overlapping mutable access
    let c_ptr = c.as_mut_ptr();
    std::thread::scope(|scope| {
        for &(i_start, i_end) in &ranges {
            let c_start = i_start * n;
            let len = (i_end - i_start) * n;
            // Safety: each thread accesses a non-overlapping region of c
            let c_slice = unsafe { std::slice::from_raw_parts_mut(c_ptr.add(c_start), len) };
            scope.spawn(move || {
                matmul_f32_block(a, b, c_slice, n, k, i_start, i_end, 0, n);
            });
        }
    });
}

/// Compute a block of the matrix multiplication.
///
/// `c` is a slice starting at row `i_start` of the full output matrix.
fn matmul_f32_block(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    i_start: usize,
    i_end: usize,
    _j_start: usize,
    j_end: usize,
) {
    for i0 in (i_start..i_end).step_by(BLOCK_M) {
        let i1 = (i0 + BLOCK_M).min(i_end);
        for j0 in (0..j_end).step_by(BLOCK_N) {
            let j1 = (j0 + BLOCK_N).min(j_end);

            for i in i0..i1 {
                // c slice starts at row i_start, so offset by i_start
                let c_row_offset = (i - i_start) * n;
                for j in j0..j1 {
                    let a_row = &a[i * k..(i + 1) * k];
                    let b_row = &b[j * k..(j + 1) * k];
                    c[c_row_offset + j] = crate::simd::dot_f32(a_row, b_row);
                }
            }
        }
    }
}
