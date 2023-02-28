crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/ParallelCommon.cpp]

pub fn get_env_var(
    var_name:  *const u8,
    def_value: *const u8) -> *const u8 {

    let def_value: *const u8 = def_value.unwrap_or(nullptr);

    todo!();
        /*
            const char* value = getenv(var_name);
      return value ? value : def_value;
        */
}

pub fn get_env_num_threads(
    var_name:  *const u8,
    def_value: usize) -> usize {

    let def_value: usize = def_value.unwrap_or(0);

    todo!();
        /*
            try {
        if (auto* value = getenv(var_name)) {
          int nthreads = stoi(value);
          TORCH_CHECK(nthreads > 0);
          return nthreads;
        }
      } catch (const exception& e) {
        ostringstream oss;
        oss << "Invalid " << var_name << " variable value, " << e.what();
        TORCH_WARN(oss.str());
      }
      return def_value;
        */
}

pub fn get_parallel_info() -> String {
    
    todo!();
        /*
            ostringstream ss;

      ss << "ATen/Parallel:\n\tat::get_num_threads() : "
         << get_num_threads() << endl;
      ss << "\tat::get_num_interop_threads() : "
         << get_num_interop_threads() << endl;

      ss << get_openmp_version() << endl;
    #ifdef _OPENMP
      ss << "\tomp_get_max_threads() : " << omp_get_max_threads() << endl;
    #endif

      ss << get_mkl_version() << endl;
    #ifdef TH_BLAS_MKL
      ss << "\tmkl_get_max_threads() : " << mkl_get_max_threads() << endl;
    #endif

      ss << get_mkldnn_version() << endl;

      ss << "thread::hardware_concurrency() : "
         << thread::hardware_concurrency() << endl;

      ss << "Environment variables:" << endl;
      ss << "\tOMP_NUM_THREADS : "
         << get_env_var("OMP_NUM_THREADS", "[not set]") << endl;
      ss << "\tMKL_NUM_THREADS : "
         << get_env_var("MKL_NUM_THREADS", "[not set]") << endl;

      ss << "ATen parallel backend: ";
      #if AT_PARALLEL_OPENMP
      ss << "OpenMP";
      #elif AT_PARALLEL_NATIVE
      ss << "native thread pool";
      #elif AT_PARALLEL_NATIVE_TBB
      ss << "native thread pool and TBB";
      #endif
      #ifdef C10_MOBILE
      ss << " [mobile]";
      #endif
      ss << endl;

      #if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
      ss << "Experimental: single thread pool" << endl;
      #endif

      return ss.str();
        */
}

pub fn intraop_default_num_threads() -> i32 {
    
    todo!();
        /*
            #ifdef C10_MOBILE
      // Intraop thread pool size should be determined by mobile cpuinfo.
      // We should hook up with the logic in caffe2/utils/threadpool if we ever need
      // call this API for mobile.
      TORCH_CHECK(false, "Undefined intraop_default_num_threads on mobile.");
    #else
      usize nthreads = get_env_num_threads("OMP_NUM_THREADS", 0);
      nthreads = get_env_num_threads("MKL_NUM_THREADS", nthreads);
      if (nthreads == 0) {
        nthreads = TaskThreadPoolBase::defaultNumThreads();
      }
      return nthreads;
    #endif
        */
}
