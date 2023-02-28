crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelNativeTBB.h]

pub const INTRA_OP_PARALLEL: bool = true;

#[inline] pub fn parallel_for<F>(
        begin:      i64,
        end:        i64,
        grain_size: i64,
        f:          &F)  {

    todo!();
        /*
            TORCH_CHECK(grain_size >= 0);
      internal::lazy_init_num_threads();
      if (begin >= end) {
        return;
      }
      if ((end - begin) < grain_size || get_num_threads() == 1) {
        f(begin, end);
        return;
      }

      // Choose number of tasks based on grain size and number of threads.
      i64 chunk_size = divup((end - begin), get_num_threads());
      // Make sure each task is at least grain_size size.
      chunk_size = max(grain_size, chunk_size);

      atomic_flag err_flag = ATOMIC_FLAG_INIT;
      exception_ptr eptr;
      tbb::parallel_for(tbb::blocked_range<i64>(begin, end, chunk_size),
        [&eptr, &err_flag, f](const tbb::blocked_range<i64>& r) {
          try {
            f(r.begin(), r.end());
          } catch (...) {
            if (!err_flag.test_and_set()) {
              eptr = current_exception();
            }
          }
        });
      if (eptr) {
        rethrow_exception(eptr);
      }
        */
}

#[inline] pub fn parallel_reduce<scalar_t, F, SF>(
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
      }
      if ((end - begin) < grain_size || get_num_threads() == 1) {
        return f(begin, end, ident);
      }

      // Choose number of tasks based on grain size and number of threads.
      i64 chunk_size = divup((end - begin), get_num_threads());
      // Make sure each task is at least grain_size size.
      chunk_size = max(grain_size, chunk_size);

      scalar_t result;
      atomic_flag err_flag = ATOMIC_FLAG_INIT;
      exception_ptr eptr;
      result = tbb::parallel_reduce(
        tbb::blocked_range<i64>(begin, end, chunk_size), ident,
        [&eptr, &err_flag, f]
            (const tbb::blocked_range<i64>& r, scalar_t ident) {
          try {
            return f(r.begin(), r.end(), ident);
          } catch (...) {
            if (!err_flag.test_and_set()) {
              eptr = current_exception();
            }
            return ident;
          }
        },
        sf
      );
      if (eptr) {
        rethrow_exception(eptr);
      }
      return result;
        */
}

pub fn intraop_invoke<F0, F1>(f0: &F0, f1: &F1)  {

    todo!();
        /*
            tbb::parallel_invoke(f0, f1);
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelNativeTBB.cpp]

#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub const TBB_PREVIEW_GLOBAL_CONTROL: usize = 1;

#[cfg(AT_PARALLEL_NATIVE_TBB)]
lazy_static!{
    /*
    static thread_local tbb::task_scheduler_init tbb_init_(intraop_default_num_threads());
    static thread_local tbb::task_group tg_;
    */
}

#[cfg(AT_PARALLEL_NATIVE_TBB)]
lazy_static!{
    /*
    mutex global_thread_mutex_;
    shared_ptr<tbb::global_control> global_thread_limit_ = nullptr;
    atomic<int> num_intraop_threads_{-1};
    */
}

#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn internal_set_num_threads(nthreads: i32)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(nthreads > 0);
      {
        unique_lock<mutex> lk(global_thread_mutex_);
        global_thread_limit_ = make_shared<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, nthreads);
        num_intraop_threads_.store(nthreads);
      }
      if (tbb_init_.is_active()) {
        tbb_init_.terminate();
      }
      tbb_init_.initialize(nthreads);
        */
}

#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn init_num_threads()  {
    
    todo!();
        /*
            #ifdef _OPENMP
      omp_set_num_threads(1);
      #endif

      #ifdef TH_BLAS_MKL
      mkl_set_num_threads(1);
      #endif

      int nthreads = num_intraop_threads_.load();
      if (nthreads < 0) {
        nthreads = intraop_default_num_threads();
      }
      _internal_set_num_threads(nthreads);
        */
}


#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn set_num_threads(nthreads: i32)  {
    
    todo!();
        /*
            TORCH_CHECK(nthreads > 0);

      _internal_set_num_threads(nthreads);
        */
}


#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn get_num_threads() -> i32 {
    
    todo!();
        /*
            return tbb::this_task_arena::max_concurrency();
        */
}


#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn get_thread_num() -> i32 {
    
    todo!();
        /*
            auto tid = tbb::this_task_arena::current_thread_index();
      return max(tid, 0);
        */
}


#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn in_parallel_region() -> bool {
    
    todo!();
        /*
            return tbb::this_task_arena::current_thread_index() >= 0;
        */
}


#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn intraop_launch(func: fn() -> ())  {
    
    todo!();
        /*
            if (get_num_threads() > 1) {
        tg_.run(func);
      } else {
        func();
      }
        */
}

#[cfg(AT_PARALLEL_NATIVE_TBB)]
pub fn intraop_launch_future(func: fn() -> ()) -> IntrusivePtr<Future> {
    
    todo!();
        /*
            auto future = make_intrusive<Future>(NoneType::get());
      if (get_num_threads() > 1) {
        tg_.run(
          [func, future]() {
            func();
            future->markCompleted();
          }
        );
      } else {
        func();
        future->markCompleted();
      }
      return future;
        */
}
