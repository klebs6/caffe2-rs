// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelOpenMP.h]

#[cfg(_OPENMP)]
pub const INTRA_OP_PARALLEL: bool = true;

#[inline] pub fn parallel_for<F>(
        begin:      i64,
        end:        i64,
        grain_size: i64,
        f:          &F)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(grain_size >= 0);
      if (begin >= end) {
        return;
      }

    #ifdef _OPENMP
      internal::lazy_init_num_threads();
      const auto numiter = end - begin;
      const bool use_parallel = (
        numiter > grain_size && numiter > 1 &&
        omp_get_max_threads() > 1 && !omp_in_parallel());
      if (!use_parallel) {
        f(begin, end);
        return;
      }

      atomic_flag err_flag = ATOMIC_FLAG_INIT;
      exception_ptr eptr;
      // Work around memory leak when using 1 thread in nested "omp parallel"
      // caused by some buggy OpenMP versions and the fact that omp_in_parallel()
      // returns false when omp_get_max_threads() == 1 inside nested "omp parallel"
      // See issue gh-32284

    #pragma omp parallel
      {
        // choose number of tasks based on grain size and number of threads
        // can't use num_threads clause due to bugs in GOMP's thread pool (See #32008)
        i64 num_threads = omp_get_num_threads();
        if (grain_size > 0) {
          num_threads = min(num_threads, divup((end - begin), grain_size));
        }

        i64 tid = omp_get_thread_num();
        i64 chunk_size = divup((end - begin), num_threads);
        i64 begin_tid = begin + tid * chunk_size;
        if (begin_tid < end) {
          try {
            f(begin_tid, min(end, chunk_size + begin_tid));
          } catch (...) {
            if (!err_flag.test_and_set()) {
              eptr = current_exception();
            }
          }
        }
      }
      if (eptr) {
        rethrow_exception(eptr);
      }
    #else
      f(begin, end);
    #endif
        */
}

#[inline] pub fn parallel_reduce<Scalar, F, SF>(
    begin:      i64,
    end:        i64,
    grain_size: i64,
    ident:      Scalar,
    f:          &F,
    sf:         &SF) -> Scalar {

    todo!();
        /*
            TORCH_CHECK(grain_size >= 0);
      internal::lazy_init_num_threads();
      if (begin >= end) {
        return ident;
      } else if ((end - begin) <= grain_size || in_parallel_region() ||
                 get_num_threads() == 1) {
        return f(begin, end, ident);
      } else {
        const i64 num_results = divup((end - begin), grain_size);
        vector<Scalar> results(num_results);
        Scalar* results_data = results.data();
        atomic_flag err_flag = ATOMIC_FLAG_INIT;
        exception_ptr eptr;
    #pragma omp parallel for
        for (i64 id = 0; id < num_results; id++) {
          i64 i = begin + id * grain_size;
          try {
            results_data[id] = f(i, i + min(end - i, grain_size), ident);
          } catch (...) {
            if (!err_flag.test_and_set()) {
              eptr = current_exception();
            }
          }
        }
        if (eptr) {
          rethrow_exception(eptr);
        }
        Scalar result = ident;
        for (auto partial_result : results) {
          result = sf(result, partial_result);
        }
        return result;
      }
        */
}



//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelOpenMP.cpp]


#[cfg(AT_PARALLEL_OPENMP)]
#[cfg(AT_MKLDNN_ENABLED)]
pub fn clear_computation_cache()  {
    
    todo!();
        /*
        
        */
}

/// Number of threads set by the user
#[cfg(AT_PARALLEL_OPENMP)]
lazy_static!{
    /*
    atomic<int> num_threads{-1};
    */
}

#[cfg(AT_PARALLEL_OPENMP)]
pub fn init_num_threads()  {
    
    todo!();
        /*
            auto nthreads = num_threads.load();
      if (nthreads > 0) {
        set_num_threads(nthreads);
      } else {
    #if defined(_OPENMP) && defined(TH_BLAS_MKL) && !defined(TH_BLAS_MKL_SEQ)
        // If we are using MKL an OpenMP make sure the number of threads match.
        // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
        // size of the OpenMP thread pool, resulting in worse performance (and memory
        // leaks in GCC 5.4)
        omp_set_num_threads(mkl_get_max_threads());
    #elif defined(_OPENMP)
        omp_set_num_threads(intraop_default_num_threads());
    #endif
      }
        */
}

#[cfg(AT_PARALLEL_OPENMP)]
pub fn set_num_threads(nthreads: i32)  {
    
    todo!();
        /*
            TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
      num_threads.store(nthreads);
    #ifdef _OPENMP
      omp_set_num_threads(nthreads);
    #endif
    #ifdef TH_BLAS_MKL
      mkl_set_num_threads(nthreads);

      // because PyTorch uses OpenMP outside of MKL invocations
      // as well, we want this flag to be false, so that
      // threads aren't destroyed and recreated across every
      // MKL / non-MKL boundary of OpenMP usage
      // See https://github.com/pytorch/pytorch/issues/13757
      mkl_set_dynamic(false);
    #endif
    #ifdef USE_PTHREADPOOL
      // because PyTorch uses pthreadpool() in QNNPACK
      PThreadPool* const pool = pthreadpool();
      TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
      pool->set_thread_count(nthreads);
    #endif
    #if AT_MKLDNN_ENABLED()
      native::mkldnn::clear_computation_cache();
    #endif
        */
}

/**
  | Explicitly calling omp_get_max_threads() as the
  | size of the parallel region might be different
  | in the new thread;
  |
  | Use init_num_threads() during thread
  | initialization to ensure consistent size of
  | parallel region in different threads
  */
#[cfg(AT_PARALLEL_OPENMP)]
pub fn get_num_threads() -> i32 {
    
    todo!();
        /*
            #ifdef _OPENMP
      return omp_get_max_threads();
    #else
      return 1;
    #endif
        */
}

#[cfg(AT_PARALLEL_OPENMP)]
pub fn get_thread_num() -> i32 {
    
    todo!();
        /*
            #ifdef _OPENMP
      return omp_get_thread_num();
    #else
      return 0;
    #endif
        */
}

#[cfg(AT_PARALLEL_OPENMP)]
pub fn in_parallel_region() -> bool {
    
    todo!();
        /*
            #ifdef _OPENMP
      return omp_in_parallel();
    #else
      return false;
    #endif
        */
}

#[cfg(AT_PARALLEL_OPENMP)]
pub fn intraop_launch(func: fn() -> ())  {
    
    todo!();
        /*
            // execute inline in openmp case
      func();
        */
}

#[cfg(AT_PARALLEL_OPENMP)]
pub fn intraop_launch_future(func: fn() -> ()) -> IntrusivePtr<Future> {
    
    todo!();
        /*
            func();
      auto future = make_intrusive<Future>(NoneType::get());
      future->markCompleted();
      return future;
        */
}
