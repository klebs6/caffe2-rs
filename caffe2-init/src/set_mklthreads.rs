crate::ix!();

#[cfg(caffe2_use_mkl)]
#[inline] pub fn caffe_2set_mklthreads(
    i: *mut i32,
    c: *mut *mut *mut u8) -> bool 
{
    todo!();
    /*
        if (!getenv("MKL_NUM_THREADS")) {
        VLOG(1) << "MKL_NUM_THREADS not passed, defaulting to 1 thread";
        mkl_set_num_threads(1);
      }

      // If caffe2_omp_num_threads is set, we use that for MKL as well.
      if (FLAGS_caffe2_omp_num_threads > 0) {
        VLOG(1) << "Setting mkl_num_threads to " << FLAGS_caffe2_omp_num_threads
                << " as inherited from omp_num_threads.";
        mkl_set_num_threads(FLAGS_caffe2_omp_num_threads);
      }

      // Override omp_num_threads if mkl_num_threads is set.
      if (FLAGS_caffe2_mkl_num_threads > 0) {
        VLOG(1) << "Setting mkl_num_threads to " << FLAGS_caffe2_mkl_num_threads;
        mkl_set_num_threads(FLAGS_caffe2_mkl_num_threads);
      }
      VLOG(1) << "Caffe2 running with " << mkl_get_max_threads() << " MKL threads";
      return true;
    */
}

