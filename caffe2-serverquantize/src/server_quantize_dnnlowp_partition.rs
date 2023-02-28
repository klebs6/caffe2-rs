crate::ix!();

#[inline] pub fn get_work_per_thread(
    work:       usize,
    nthreads:   i32,
    work_align: i32) -> usize {
    
    todo!();
    /*
        return ((work + work_align - 1) / work_align + nthreads - 1) / nthreads *
          work_align;
    */
}

#[inline] pub fn get1d_partition(
    work:       usize,
    nthreads:   i32,
    tid:        i32,
    work_align: Option<i32>) -> (usize,usize) {
    let work_align: i32 = work_align.unwrap_or(1);

    todo!();
    /*
        size_t work_per_thread = GetWorkPerThread_(work, nthreads, work_align);
      size_t work_begin = std::min(tid * work_per_thread, work);
      size_t work_end = std::min(work_begin + work_per_thread, work);
      return {work_begin, work_end};
    */
}

/**
  | 1D-partition m x n 2D work.
  | 
  | First try partitioning m if m >= nthreads.
  | 
  | Otherwise, each row is partitioned
  | by multiple threads.
  | 
  | In this case, each thread only works
  | on a single row.
  | 
  | Optionally, we can force the number
  | of columns assigned per thread is a multiple
  | of n_align.
  |
  */
#[inline] pub fn get1d_partition_of2d(
    m:        i32,
    n:        i32,
    nthreads: i32,
    tid:      i32,
    m_begin:  *mut i32,
    m_end:    *mut i32,
    n_begin:  *mut i32,
    n_end:    *mut i32,
    n_align:  Option<i32>)  {
    let n_align: i32 = n_align.unwrap_or(1);

    todo!();
    /*
        if (m >= nthreads || m == 0) {
        // When m >= nthreads, just parallelize over m.
        std::tie(*m_begin, *m_end) = Get1DPartition(m, nthreads, tid);
        *n_begin = 0;
        *n_end = n;
      } else {
        // Otherwise, each row is parallelized by multiple threads.
        // nthreads_per_row is floor(nthreads / m). If we use ceil, some rows won't
        // be handled by any thread.
        int nthreads_per_row = nthreads / m;
        *m_begin = std::max(std::min(tid / nthreads_per_row, m - 1), 0);
        *m_end = std::min(*m_begin + 1, m);

        int tid_of_m_begin = std::min(*m_begin * nthreads_per_row, nthreads);
        int tid_of_m_end = std::min(
            (*m_end == m) ? nthreads : (tid_of_m_begin + nthreads_per_row),
            nthreads);
        int nthreads_within_row = tid_of_m_end - tid_of_m_begin;
        int tid_within_row = tid - tid_of_m_begin;
        CAFFE_ENFORCE_GE(tid_within_row, 0);
        CAFFE_ENFORCE_LT(tid_within_row, nthreads_within_row);

        std::tie(*n_begin, *n_end) =
            Get1DPartition(n, nthreads_within_row, tid_within_row, n_align);
      }
    */
}
