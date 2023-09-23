crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelNative.h]

pub const INTRA_OP_PARALLEL: bool = true;

#[inline] pub fn calc_num_tasks_and_chunk_size(
    begin:      i64,
    end:        i64,
    grain_size: i64) -> (usize,usize) {
    
    todo!();
        /*
            if ((end - begin) < grain_size) {
        return make_tuple(1, max((i64)0, end - begin));
      }
      // Choose number of tasks based on grain size and number of threads.
      usize chunk_size = divup((end - begin), get_num_threads());
      // Make sure each task is at least grain_size size.
      chunk_size = max((usize)grain_size, chunk_size);
      usize num_tasks = divup((end - begin), chunk_size);
      return make_tuple(num_tasks, chunk_size);
        */
}

pub fn parallel_run(
        begin:      i64,
        end:        i64,
        grain_size: i64,
        f:          &fn(
                _0: i64,
                _1: i64,
                _2: usize
        ) -> ())  {
    
    todo!();
        /*
        
        */
}

#[inline] pub fn parallel_for<F>(
        begin:      i64,
        end:        i64,
        grain_size: i64,
        f:          &F)  {

    todo!();
        /*
            TORCH_CHECK(grain_size >= 0);
      if (begin >= end) {
        return;
      }
      if ((end - begin) < grain_size || in_parallel_region()) {
        f(begin, end);
        return;
      }
      internal::_parallel_run(
          begin,
          end,
          grain_size,
          [f](i64 start, i64 end, usize /* unused */) {
            f(start, end);
          }
      );
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
      if (begin >= end) {
        return ident;
      }
      if ((end - begin) < grain_size || in_parallel_region()) {
        return f(begin, end, ident);
      }
      usize num_tasks, chunk_size;
      tie(num_tasks, chunk_size) =
          internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);
      vector<Scalar> results(num_tasks);
      Scalar* results_data = results.data();
      internal::_parallel_run(
          begin,
          end,
          grain_size,
          [f, ident, results_data](i64 start, i64 end, usize task_id) {
            results_data[task_id] = f(start, end, ident);
          }
      );
      Scalar result = ident;
      for (auto partial_result : results) {
        result = sf(result, partial_result);
      }
      return result;
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelNative.cpp]

#[cfg(AT_PARALLEL_NATIVE)]
mod at_parallel_native {
    use super::*;

    /**
      | used with _set_in_parallel_region
      | to mark master thread as in parallel
      | region while executing parallel primitives
      |
      */
    lazy_static!{
        /*
        thread_local bool in_parallel_region_ = false;
        */
    }

    /// thread number (task_id) set by parallel primitive
    ///
    lazy_static!{
        /*
        thread_local usize thread_num_ = 0;
        */
    }

    pub fn set_in_parallel_region(in_region: bool)  {
        
        todo!();
            /*
                in_parallel_region_ = in_region;
            */
    }

    pub fn set_thread_num(thread_num: usize)  {
        
        todo!();
            /*
                thread_num_ = thread_num;
            */
    }

    pub fn unset_thread_num()  {
        
        todo!();
            /*
                thread_num_ = 0;
            */
    }

    #[cfg(not(C10_MOBILE))]
    pub const NOT_SET:  i32 = -1;

    #[cfg(not(C10_MOBILE))]
    pub const CONSUMED: i32 = -2;

    /**
      | Number of threads set by the user
      |
      | NOT_SET -> positive value -> CONSUMED
      |
      | or
      |
      | NOT_SET -> CONSUMED
      |
      | Meaning:
      |
      |  - NOT_SET - pool not initialized, user value
      |  is not set
      |
      |  - positive value - pool not initialized, user
      |  value set
      |
      |  - CONSUMED - pool is initialized
      |
      */
    #[cfg(not(C10_MOBILE))]
    lazy_static!{
        /*
        atomic<int> num_intraop_threads{NOT_SET};
        */
    }

    #[cfg(not(C10_MOBILE))]
    pub fn num_pool_threads(nthreads: i32) -> i32 {
        
        todo!();
            /*
                if (nthreads == NOT_SET) {
            nthreads = intraop_default_num_threads();
          } else {
            TORCH_INTERNAL_ASSERT(nthreads > 0);
          }
          // minus one because of the master thread
          return nthreads - 1;
            */
    }

    #[cfg(not(C10_MOBILE))]
    pub fn get_intraop_pool() -> &mut TaskThreadPoolBase {
        
        todo!();
            /*
                static shared_ptr<TaskThreadPoolBase> pool =
              ThreadPoolRegistry()->Create(
                  "C10",
                  /* device_id */ 0,
                  /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
                  /* create_new */ true); // create a separate thread pool for intra-op
          return *pool;
            */
    }

    /**
      | Run lambda function `fn` over `task_id` in [0,
      | `range`) with threadpool.
      |
      | `fn` will be called with params:
      | (thread_pool_task_id, task_id).
      |
      */
    pub fn run_with_pool(
        fn_:   &fn(_0: i32, _1: usize) -> (),
        range: usize)  {

        todo!();
            /*
                #ifndef C10_MOBILE
          for (usize i = 1; i < range; ++i) {
            _get_intraop_pool().run([fn, i]() { fn((int)i, i); });
          }
          // Run the first task on the current thread directly.
          fn(0, 0);
        #else
          PThreadPool* const pool = pthreadpool();
          TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");

          pool->run(
            // PThreadPool::run() is blocking.  A function [const] reference to
            // this lambda cannot go out of scope before PThreadPool::run() returns.
            [&fn](const usize task_id) {
              fn(0 /* unused */, task_id);
            }, range);
        #endif // C10_MOBILE
            */
    }

    /**
      | RAII guard helps to support in_parallel_region()
      | and get_thread_num() API.
      |
      */
    pub struct ParallelRegionGuard {

    }

    impl ParallelRegionGuard {

        pub fn new(task_id: i64) -> Self {
        
            todo!();
            /*


                _set_thread_num(task_id);
            _set_in_parallel_region(true);
            */
        }
    }

    impl Drop for ParallelRegionGuard {

        fn drop(&mut self) {
            todo!();
            /*
                _set_in_parallel_region(false);
            _unset_thread_num();
            */
        }
    }

    pub fn parallel_run(
            begin:      i64,
            end:        i64,
            grain_size: i64,
            f:          &fn(
                    _0: i64,
                    _1: i64,
                    _2: usize
            ) -> ())  {
        
        todo!();
            /*
                internal::lazy_init_num_threads();

          usize num_tasks, chunk_size;
          tie(num_tasks, chunk_size) =
              internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

          struct {
            atomic_flag err_flag = ATOMIC_FLAG_INIT;
            exception_ptr eptr;
            mutex mutex;
            volatile usize remaining;
            condition_variable cv;
          } state;

          auto task = [f, &state, begin, end, chunk_size]
              (int /* unused */, usize task_id) {
            i64 local_start = begin + task_id * chunk_size;
            if (local_start < end) {
              i64 local_end = min(end, (i64)(chunk_size + local_start));
              try {
                ParallelRegionGuard guard(task_id);
                f(local_start, local_end, task_id);
              } catch (...) {
                if (!state.err_flag.test_and_set()) {
                  state.eptr = current_exception();
                }
              }
            }
            {
              unique_lock<mutex> lk(state.mutex);
              if (--state.remaining == 0) {
                state.cv.notify_one();
              }
            }
          };
          state.remaining = num_tasks;
          _run_with_pool(task, num_tasks);

          // Wait for all tasks to finish.
          {
            unique_lock<mutex> lk(state.mutex);
            if (state.remaining != 0) {
              state.cv.wait(lk);
            }
          }
          if (state.eptr) {
            rethrow_exception(state.eptr);
          }
            */
    }

    pub fn init_num_threads()  {
        
        todo!();
            /*
                #ifdef _OPENMP
          omp_set_num_threads(1);
        #endif

        #ifdef TH_BLAS_MKL
          mkl_set_num_threads(1);
        #endif

        #ifdef C10_MOBILE
          pthreadpool();
        #endif
            */
    }

    pub fn set_num_threads(nthreads: i32)  {
        
        todo!();
            /*
                #ifndef C10_MOBILE
          TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
          int no_value = NOT_SET;
          if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
            // num_intraop_threads either stores a positive integer or CONSUMED,
            // check that requested size is the same as the current one
            int stored_nthreads = num_intraop_threads.load();
            if (stored_nthreads <= 0) {
              // plus one because of master thread
              stored_nthreads = _get_intraop_pool().size() + 1;
            }
            if (stored_nthreads != nthreads) {
              TORCH_WARN(
                "Cannot set number of intraop threads "
                "after parallel work has started or after set_num_threads call "
                "when using native parallel backend");
            }
          }
        #else
          PThreadPool* const pool = pthreadpool();
          TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
          pool->set_thread_count(nthreads);
        #endif // C10_MOBILE
            */
    }

    pub fn get_num_threads() -> i32 {
        
        todo!();
            /*
                #ifndef C10_MOBILE
          // not initializing pool unnecessarily,
          // because pool cannot be resized after initialization
          int nthreads = num_intraop_threads.load();
          if (nthreads > 0) {
            return nthreads;
          } else if (nthreads == NOT_SET) {
            return intraop_default_num_threads();
          } else {
            TORCH_INTERNAL_ASSERT(nthreads == CONSUMED);
            return _get_intraop_pool().size() + 1;
          }
        #else
          PThreadPool* const pool = pthreadpool();
          TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!")
          return in_parallel_region() ? 1 /* current thread */ : pool->get_thread_count();
        #endif // C10_MOBILE
            */
    }

    pub fn get_thread_num() -> i32 {
        
        todo!();
            /*
                return thread_num_;
            */
    }

    pub fn in_parallel_region() -> bool {
        
        todo!();
            /*
                #ifndef C10_MOBILE
          return in_parallel_region_ || (
            num_intraop_threads.load() == CONSUMED &&
            // Needed as intraop_launch() doesn't set in_parallel_region().
            _get_intraop_pool().inThreadPool()
          );
        #else
          return in_parallel_region_;
        #endif // C10_MOBILE
            */
    }

    pub fn intraop_launch(func: fn() -> ())  {
        
        todo!();
            /*
                #ifndef C10_MOBILE
          if (!in_parallel_region() && get_num_threads() > 1) {
            _get_intraop_pool().run(func);
          } else {
            // execute inline if we're in parallel region
            func();
          }
        #else
          // TODO: PThreadPool only provides a data-parallel API.
          // Task parallelism is not currently supported.
          func();
        #endif // C10_MOBILE
            */
    }

    pub fn intraop_launch_future(func: fn() -> ()) -> IntrusivePtr<Future> {
        
        todo!();
            /*
                #ifndef C10_MOBILE
          auto future = make_intrusive<Future>(NoneType::get());
          if (!in_parallel_region() && get_num_threads() > 1) {
            _get_intraop_pool().run(
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
        #else
          // TODO: PThreadPool only provides a data-parallel API.
          // Task parallelism is not currently supported.
          auto future = make_intrusive<Future>(NoneType::get());
          func();
          future->markCompleted();
          return future;
        #endif // C10_MOBILE
            */
    }
}
