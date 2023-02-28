/*!
  | Note [behavior of cudnnFind and cudnnGet]
  |
  | You'll notice that by default, in the
  | ConvolutionDescriptor, we do the following:
  |
  |     AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(),
  |     CUDNN_DEFAULT_MATH));
  |
  |     if(dataType == CUDNN_DATA_HALF)
  |       AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(),
  |       CUDNN_TENSOR_OP_MATH));
  |
  |     Update: AT_CUDNN_CHECK is updated with
  |        AT_CUDNN_CHECK_WITH_SHAPES, which
  |        automatically prints tensor shapes and
  |        convolution parameters if there is
  |        a cuDNN exception thrown.
  |
  | When cudnnSetConvolutionMathType is called
  | before cudnnGet/cudnnFind, it informs
  | cudnnGet/cudnnFind to iterate/take into account
  | both tensor core and non-tensor-core algos.
  |
  | If you don't call cudnnSetConvolutionMathType
  | before calling cudnnGet/cudnnFind,
  | cudnnGet/cudnnFind may not pick tensor core
  | algos.
  |
  | Now after its run, cudnnGet/cudnnFind comes up
  | with the best pair of algo+mathType with all
  | the initial knowledge its given. It then
  | becomes the user's responsibility to update
  | mathType of the convolution descriptor and call
  | the subsequent cudnn calls with the best algo
  | and the updated descriptor. If we don't update
  | the descriptor but just run with the best algo,
  | under the hood, cudnn will run with the slower
  | kernel since it sees fastest algorithm
  | combination with a sub optimal mathType.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/Conv_v7.cpp]

/**
  | Note [blocklist fft algorithms for strided
  | dgrad]
  |
  | This is a workaround for a CuDNN bug that gave
  | wrong results in certain strided convolution
  | gradient setups. Check Issue #16610 for bug
  | details. Bug is there for CUDNN version < 7.5 .
  */
#[cfg(AT_CUDNN_ENABLED)]
pub fn operator_tib_raw_string_suffix(n: u64) -> usize {
    
    todo!();
        /*
            return usize(n) * 1024 * 1024 * 1024 * 1024;
        */
}

// Convenience struct for passing around descriptors and data
// pointers
#[cfg(AT_CUDNN_ENABLED)]
pub struct ConvolutionArgs {
    handle: CuDnnHandle,
    params: ConvolutionParams,
    idesc:  TensorDescriptor,
    odesc:  TensorDescriptor,
    wdesc:  FilterDescriptor,
    input:  &Tensor,
    output: &Tensor,
    weight: &Tensor,
    cdesc:  ConvolutionDescriptor,
}

#[cfg(AT_CUDNN_ENABLED)]
impl ConvolutionArgs {
    
    pub fn new(
        input:  &Tensor,
        output: &Tensor,
        weight: &Tensor) -> Self {
    
        todo!();
        /*


            : input(input), output(output), weight(weight)
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
impl fmt::Display for ConvolutionArgs {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << repro_from_args(args.params)  // already has a trailing newline
        << args.params                     // already has a trailing newline
        << "input: " << args.idesc         // already has a trailing newline
        << "output: " << args.odesc        // already has a trailing newline
        << "weight: " << args.wdesc        // already has a trailing newline
        << "Pointer addresses: " << "\n"
        << "    input: " << args.input.data_ptr() << "\n"
        << "    output: " << args.output.data_ptr() << "\n"
        << "    weight: " << args.weight.data_ptr() << "\n";

      return out;
        */
    }
}

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// TODO: Use something less heavy duty than a big honking mutex
#[cfg(AT_CUDNN_ENABLED)]
pub struct BenchmarkCache<T> {
    mutex: Mutex,
    map:   HashMap<ConvolutionParams,T,ParamsHash<ConvolutionParams>,ParamsEqual<ConvolutionParams>>,
}

#[cfg(AT_CUDNN_ENABLED)]
impl BenchmarkCache<T> {
    
    pub fn find(&mut self, 
        params:  &ConvolutionParams,
        results: *mut T) -> bool {
        
        todo!();
        /*
            lock_guard<mutex> guard(mutex);
        auto it = map.find(params);
        if (it == map.end()) {
          return false;
        }
        *results = it->second;
        return true;
        */
    }
    
    pub fn insert(&mut self, 
        params:  &ConvolutionParams,
        results: &T)  {
        
        todo!();
        /*
            lock_guard<mutex> guard(mutex);
        map[params] = results;
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
lazy_static!{
    /*
    BenchmarkCache<cudnnConvolutionFwdAlgoPerf_t> fwd_algos;
    BenchmarkCache<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algos;
    BenchmarkCache<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algos;
    */
}

/**
  | TODO: Stop manually allocating CUDA
  | memory; allocate an ATen byte tensor
  | instead.
  |
  */
#[cfg(AT_CUDNN_ENABLED)]
pub struct Workspace {
    size: usize,
    data: *mut void,
}

#[cfg(AT_CUDNN_ENABLED)]
impl Workspace {
    
    pub fn new(size: usize) -> Self {
    
        todo!();
        /*
        : size(size),
        : data(NULL),

            // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
        // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
        // we manually fail with OOM.
        TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
        data = THCudaMalloc(globalContext().lazyInitCUDA(), size);
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
impl Drop for Workspace {
    fn drop(&mut self) {
        todo!();
        /*
            if (data) {
          THCudaFree(globalContext().lazyInitCUDA(), data);
        }
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
pub struct AlgorithmSearch<perf_t> {

}

#[cfg(AT_CUDNN_ENABLED)]
pub fn get_workspace_size(
        args: &ConvolutionArgs,
        algo: cudnn::ConvolutionFwdAlgo,
        sz:   *mut usize) -> cudnn::Status {
    
    todo!();
        /*
            return cudnnGetConvolutionForwardWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            algo,
            sz
        );
        */
}


#[cfg(AT_CUDNN_ENABLED)]
pub fn get_workspace_size(
        args: &ConvolutionArgs,
        algo: cudnn::ConvolutionBwdDataAlgo,
        sz:   *mut usize) -> cudnn::Status {
    
    todo!();
        /*
            return cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            algo,
            sz);
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn get_workspace_size(
        args: &ConvolutionArgs,
        algo: cudnn::ConvolutionBwdFilterAlgo,
        sz:   *mut usize) -> cudnn::Status {
    
    todo!();
        /*
            return cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            algo,
            sz);
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn get_max_workspace_size<algo_t>(
        args:   &ConvolutionArgs,
        algo:   *const Algo,
        n_algo: i32) -> usize {

    todo!();
        /*
            usize max_ws_size = 0;
      usize max_block_size = 0;
      usize tmp_bytes = 0;  // Only used for filling pointer parameters that aren't used later

      int device;
      THCudaCheck(cudaGetDevice(&device));
      CUDACachingAllocator::cacheInfo(device, &tmp_bytes, &max_block_size);

      for (int i = 0; i < n_algo; i++) {
        cudnnStatus_t err;
        usize sz;
        err = getWorkspaceSize(args, algo[i], &sz);
        if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > max_block_size)
          continue;
        max_ws_size = sz;
      }
      return max_ws_size;
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn get_valid_algorithms<perf_t>(
        perf_results: *mut Perf,
        args:         &ConvolutionArgs,
        n_algo:       i32) -> Vec<Perf> {

    todo!();
        /*
            // See Note [blocklist fft algorithms for strided dgrad]
    #if CUDNN_VERSION < 7500
      bool blocklist = is_same<decltype(perfResults[0].algo), cudnnConvolutionBwdDataAlgo_t>::value;
      int stride_dim = args.input.dim() - 2;
      blocklist &= any_of(begin(args.params.stride),
                                begin(args.params.stride) + stride_dim,
                                [=](int n){return n != 1;});
    #endif

      vector<perf_t> result;
      result.reserve(n_algo);
      for (int i = 0; i < n_algo; i++) {
        perf_t perf = perfResults[i];

        // TODO: Shouldn't all returned results be successful?
        // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
        if (perf.status == CUDNN_STATUS_SUCCESS) {
          if (!args.params.deterministic || perf.determinism == CUDNN_DETERMINISTIC) {

            // See Note [blocklist fft algorithms for strided dgrad]
    #if CUDNN_VERSION < 7500
            bool skip = blocklist;
            skip &= (static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                      static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
            if (skip) {
              continue;
            }
    #endif

            result.push_back(perf);
          }
        }
      }
      TORCH_CHECK(result.size() > 0, "no valid convolution algorithms available in CuDNN");
      return result;
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub struct AlgorithmSearchCudnnConvolutionFwdAlgoPerf_t {

}

#[cfg(AT_CUDNN_ENABLED)]
impl AlgorithmSearchCudnnConvolutionFwdAlgoPerf_t {

    pub type Perf = CuDnnConvolutionFwdAlgoPerf;
    pub type Algo = cudnn::ConvolutionFwdAlgo;

    pub const DEFAULT_ALGO: Auto = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    
    pub fn cache() -> &mut BenchmarkCache<Perf> {
        
        todo!();
        /*
            return fwd_algos;
        */
    }
    
    pub fn find_algorithms(
        args:      &ConvolutionArgs,
        benchmark: bool) -> Vec<Perf> {
        
        todo!();
        /*
            static const algo_t algos[] = {
             CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
             CUDNN_CONVOLUTION_FWD_ALGO_FFT,
             CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
             CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
             CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
             CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        };
        static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                      "Missing cuDNN convolution forward algorithms");
        int perf_count;
        unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
        if (!benchmark) {
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardAlgorithm_v7(
              args.handle,
              args.idesc.desc(),
              args.wdesc.desc(),
              args.cdesc.desc(),
              args.odesc.desc(),
              num_algos,
              &perf_count,
              perf_results.get()), args);
        } else {
          usize max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
          Workspace ws(max_ws_size);
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionForwardAlgorithmEx(
              args.handle,
              args.idesc.desc(), args.input.data_ptr(),
              args.wdesc.desc(), args.weight.data_ptr(),
              args.cdesc.desc(),
              args.odesc.desc(), args.output.data_ptr(),
              num_algos,
              &perf_count,
              perf_results.get(),
              ws.data,
              ws.size), args);

          // Free the cached blocks in our caching allocator. They are
          // needed here because the above benchmarking uses a huge amount of memory,
          // e.g. a few GBs.
          CUDACachingAllocator::emptyCache();
        }
        return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
        */
    }
    
    pub fn get_workspace_size(
        args:           &ConvolutionArgs,
        algo:           Algo,
        workspace_size: *mut usize)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            algo,
            workspaceSize), args);
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
pub struct AlgorithmSearchCuDnnConvolutionBwdDataAlgoPerf {

}

#[cfg(AT_CUDNN_ENABLED)]
impl AlgorithmSearchCuDnnConvolutionBwdDataAlgoPerf {

    pub type Perf = CuDnnConvolutionBwdDataAlgoPerf;
    pub type Algo = CuDnnConvolutionBwdDataAlgo;

    pub const DEFAULT_ALGO: Auto = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    
    pub fn cache() -> &mut BenchmarkCache<Perf> {
        
        todo!();
        /*
            return bwd_data_algos;
        */
    }
    
    pub fn find_algorithms(
        args:      &ConvolutionArgs,
        benchmark: bool) -> Vec<Perf> {
        
        todo!();
        /*
            static const algo_t algos[] = {
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
        };
        static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
        static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                      "Missing cuDNN convolution backward data algorithms.");
        int perf_count;
        unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
        if (!benchmark) {
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataAlgorithm_v7(
              args.handle,
              args.wdesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.idesc.desc(),
              num_algos,
              &perf_count,
              perf_results.get()), args);
        } else {
          usize max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
          Workspace ws(max_ws_size);
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionBackwardDataAlgorithmEx(
              args.handle,
              args.wdesc.desc(), args.weight.data_ptr(),
              args.odesc.desc(), args.output.data_ptr(),
              args.cdesc.desc(),
              args.idesc.desc(), args.input.data_ptr(),
              num_algos,
              &perf_count,
              perf_results.get(),
              ws.data,
              ws.size), args);

          // Free the cached blocks in our caching allocator. They are
          // needed here because the above benchmarking uses a huge amount of memory,
          // e.g. a few GBs.
          CUDACachingAllocator::emptyCache();
        }
        return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
        */
    }
    
    pub fn get_workspace_size(
        args:           &ConvolutionArgs,
        algo:           cudnn::ConvolutionBwdDataAlgo,
        workspace_size: *mut usize)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            algo,
            workspaceSize), args);
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
pub struct AlgorithmSearchCuDnnConvolutionBwdFilterAlgoPerf {

}

#[cfg(AT_CUDNN_ENABLED)]
impl AlgorithmSearchCuDnnConvolutionBwdFilterAlgoPerf {

    pub type Perf = CuDnnConvolutionBwdFilterAlgoPerf;
    pub type Algo = CuDnnConvolutionBwdFilterAlgo;

    pub const DEFAULT_ALGO: Auto = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    pub fn cache() -> &mut BenchmarkCache<Perf> {
        
        todo!();
        /*
            return bwd_filter_algos;
        */
    }
    
    pub fn find_algorithms(
        args:      &ConvolutionArgs,
        benchmark: bool) -> Vec<Perf> {
        
        todo!();
        /*
            static const algo_t algos[] = {
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
        };
        // NOTE: - 1 because ALGO_WINOGRAD is not implemented
        static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
        static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                      "Missing cuDNN convolution backward filter algorithms.");
        unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
        int perf_count;
        if (!benchmark) {
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
              args.handle,
              args.idesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              num_algos,
              &perf_count,
              perf_results.get()), args);
        } else {
          usize max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
          Workspace ws(max_ws_size);
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionBackwardFilterAlgorithmEx(
              args.handle,
              args.idesc.desc(), args.input.data_ptr(),
              args.odesc.desc(), args.output.data_ptr(),
              args.cdesc.desc(),
              args.wdesc.desc(), args.weight.data_ptr(),
              num_algos,
              &perf_count,
              perf_results.get(),
              ws.data,
              ws.size), args);

          // Free the cached blocks in our caching allocator. They are
          // needed here because the above benchmarking uses a huge amount of memory,
          // e.g. a few GBs.
          CUDACachingAllocator::emptyCache();
        }
        return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
        */
    }
    
    pub fn get_workspace_size(
        args:           &ConvolutionArgs,
        algo:           Algo,
        workspace_size: *mut usize)  {
        
        todo!();
        /*
            AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            algo,
            workspaceSize), args);
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
pub struct AlgoIterator<perf_t> {
    args:      &ConvolutionArgs,
    benchmark: bool,
}

#[cfg(AT_CUDNN_ENABLED)]
impl AlgoIterator<perf_t> {

    pub type Search = AlgorithmSearch<Perf>;

    pub fn new(
        args:      &ConvolutionArgs,
        benchmark: bool) -> Self {
    
        todo!();
        /*
        : args(args),
        : benchmark(benchmark),

        
        */
    }
    
    pub fn only_default_algorithm(args: &ConvolutionArgs) -> Vec<Perf> {
        
        todo!();
        /*
            vector<perf_t> perfResults(1);
        perfResults[0].algo = search::DEFAULT_ALGO;
        if (args.params.dataType == CUDNN_DATA_HALF) {
          perfResults[0].mathType = CUDNN_TENSOR_OP_MATH;
        } else {
          perfResults[0].mathType = CUDNN_DEFAULT_MATH;
    #if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
          if (args.params.dataType == CUDNN_DATA_FLOAT && !args.params.allow_tf32) {
            perfResults[0].mathType = CUDNN_FMA_MATH;
          }
    #endif
        }
        search::getWorkspaceSize(args, perfResults[0].algo, &(perfResults[0].memory));
        return perfResults;
        */
    }
    
    pub fn try_all(&mut self, f: fn(perf: &Perf) -> ())  {
        
        todo!();
        /*
            bool only_use_default = args.params.deterministic && !benchmark;

        auto& cache = search::cache();
        perf_t algoPerf;
        if (!only_use_default && cache.find(args.params, &algoPerf)) {
          try {
            f(algoPerf);
            return;
          } catch (CUDAOutOfMemoryError &e) {
            cudaGetLastError(); // clear CUDA error
          }
        }

        auto perfResults = only_use_default ? onlyDefaultAlgorithm(args) : search::findAlgorithms(args, benchmark);
        for (auto &algoPerf : perfResults) {
          try {
            f(algoPerf);
            cache.insert(args.params, algoPerf);
            return;
          } catch (CUDAOutOfMemoryError &e) {
            cudaGetLastError(); // clear CUDA error
          } catch (CuDNNError &e) {
            cudaGetLastError(); // clear CUDA error
          }
        }
        TORCH_CHECK(false, "Unable to find a valid cuDNN algorithm to run convolution");
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
#[inline] pub fn allocate_workspace(
        size:  usize,
        other: &Tensor) -> Tensor {
    
    todo!();
        /*
            // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
      // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
      // we manually fail with OOM.
      TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
      return empty({static_cast<i64>(size)}, other.options().dtype(kByte));
        */
}

/*
  | NOTE [ raw_cudnn_convolution_forward_out ]
  |
  |    - raw_cudnn_convolution_forward_out (Tensor)
  |
  |      Functiont that handles tensors that are
  |      too large to use 32bit indexing.
  |
  |      It just split the tensor and dispatches to
  |      `raw_cudnn_convolution_forward_out_32bit`.
  |
  |    - raw_cudnn_convolution_forward_out_32bit
  |    (Tensor)
  |
  |      Low level function which invokes CuDNN,
  |      and takes an output tensor which is
  |      directly written to (thus _out).
  |
  */

// ---------------------------------------------------------------------
//
// Splitting to 32bit
//
// ---------------------------------------------------------------------
#[cfg(AT_CUDNN_ENABLED)]
#[inline] pub fn split_batch_dim_to_32bit_out<func_t>(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool,
        max_worksize:  i64,
        func_32bit:    Func)  {

    todo!();
        /*
            constexpr i64 int_max = int::max;
      const i64 ni = input.numel();
      const i64 no = output.numel();
      // Assume the shape of the tensor is (N, C, D1, D2, ...)
      // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
      if (ni <= int_max && no <= int_max) {
        func_32bit(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
        return;
      }
      // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
      //
      // Here we use a simple heuristics to determine the size of each split
      // We don't max out the 2^31 address space because this number is super
      // large and very likely to get an OOM.
      i64 n = output.size(0);
      i64 max_inner_size = max<i64>(ni, no) / n;
      i64 split_size = max<i64>(max_worksize / max_inner_size, 1L);
      i64 num_splits = (n + split_size - 1) / split_size;
      if (split_size * max_inner_size < int_max) {
        for (i64 i = 0; i < num_splits; i++) {
          i64 start = split_size * i;
          i64 split_size_ = min<i64>(split_size, n - start);
          Tensor input_ = input.narrow(0, start, split_size_);
          Tensor output_ = output.narrow(0, start, split_size_);
          func_32bit(output_, input_, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
        }
        return;
      }
      // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
      // For example, for conv2d, there following questions needs to be considered.
      // - Is the memory layout NCHW or NHWC ?
      // - If the conv is NCHW -> NC'H'W', then should we
      //   - split only NC?
      //   - split only N'C'?
      //   - split both?
      // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
      //   to make sure that the boundary is handled correctly.
      // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
      // Considering the complexity of this issue, it is better not to use cuDNN for this case
      TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
        */
}

#[cfg(AT_CUDNN_ENABLED)]
#[cfg(all(CUDNN_VERSION,CUDNN_VERSION_GTE_8000))]
#[macro_export] macro_rules! assert_correct_precision {
    ($math_type:ident) => {
        /*
        
        if (args.params.dataType == CUDNN_DATA_FLOAT) {                                 
          TORCH_INTERNAL_ASSERT(args.params.allow_tf32 || math_type == CUDNN_FMA_MATH); 
        }
        */
    }
}

#[cfg(AT_CUDNN_ENABLED)]
#[cfg(not(all(CUDNN_VERSION,CUDNN_VERSION_GTE_8000)))]
#[macro_export] macro_rules! assert_correct_precision { ($math_type:ident) => { } }

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

#[cfg(AT_CUDNN_ENABLED)]
#[cfg(not(HAS_CUDNN_V8))]
pub fn raw_cudnn_convolution_forward_out_32bit(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            auto dataType = getCudnnDataType(input);

      ConvolutionArgs args{ input, output, weight };
      args.handle = getCudnnHandle();
      setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
      args.idesc.set(input);
      args.wdesc.set(weight, input.suggest_memory_format(), 0);
      args.odesc.set(output);
      args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);

      // TODO: when we do legacy group convolution support, we'll repeatedly
      // reinitialize the workspace for each convolution we do.  This is
      // wasteful; we'd rather reuse the workspace.  OTOH, legacy group
      // convolution support is already pretty slow, so this might not
      // matter.  (This applies to raw_cudnn_convolution_backward_input as well.)
      AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark).try_all(
        [&](const cudnnConvolutionFwdAlgoPerf_t &fwdAlgPerf){
          Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

          // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
          // whether to use Tensor core kernels or not
          // See Note [behavior of cudnnFind and cudnnGet]
          ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), fwdAlgPerf.mathType), args);

          Constant one(dataType, 1);
          Constant zero(dataType, 0);

          AT_CUDNN_CHECK_WITH_SHAPES(cudnnConvolutionForward(
              args.handle,
              &one, args.idesc.desc(), input.data_ptr(),
              args.wdesc.desc(), weight.data_ptr(),
              args.cdesc.desc(), fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory,
              &zero, args.odesc.desc(), output.data_ptr()),
            args, "Forward algorithm: ", static_cast<int>(fwdAlgPerf.algo), "\n");
          }
      );
        */
}

#[cfg(AT_CUDNN_ENABLED)]
#[cfg(not(HAS_CUDNN_V8))]
pub fn raw_cudnn_convolution_forward_out(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            split_batch_dim_to_32bit_out(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, 1024 * 1024 * 256, raw_cudnn_convolution_forward_out_32bit);
        */
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

#[cfg(AT_CUDNN_ENABLED)]
pub fn raw_cudnn_convolution_backward_input_out_32bit(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            auto dataType = getCudnnDataType(grad_output);

      ConvolutionArgs args{ grad_input, grad_output, weight };
      args.handle = getCudnnHandle();
      setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups, deterministic, allow_tf32);
      args.idesc.set(grad_input);
      args.wdesc.set(weight, grad_output.suggest_memory_format(), 0);
      args.odesc.set(grad_output);
      args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);

      AlgoIterator<cudnnConvolutionBwdDataAlgoPerf_t>(args, benchmark).try_all(
        [&](const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgPerf){
          Tensor workspace = allocate_workspace(bwdDataAlgPerf.memory, grad_output);

          // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
          // whether to use Tensor core kernels or not
          // See Note [behavior of cudnnFind and cudnnGet]
          ASSERT_CORRECT_PRECISION(bwdDataAlgPerf.mathType);
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdDataAlgPerf.mathType), args);

          Constant one(dataType, 1);
          Constant zero(dataType, 0);

          AT_CUDNN_CHECK_WITH_SHAPES(cudnnConvolutionBackwardData(
              args.handle,
              &one, args.wdesc.desc(), weight.data_ptr(),
              args.odesc.desc(), grad_output.data_ptr(),
              args.cdesc.desc(), bwdDataAlgPerf.algo, workspace.data_ptr(), bwdDataAlgPerf.memory,
              &zero, args.idesc.desc(), grad_input.data_ptr()),
            args,
            "Additional pointer addresses: \n",
            "    grad_output: ", grad_output.data_ptr(), "\n",
            "    grad_input: ", grad_input.data_ptr(), "\n",
            "Backward data algorithm: ", static_cast<int>(bwdDataAlgPerf.algo), "\n");
        }
      );
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn raw_cudnn_convolution_backward_input_out(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            split_batch_dim_to_32bit_out(grad_input, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, 1024 * 1024 * 128, raw_cudnn_convolution_backward_input_out_32bit);
        */
}


// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

#[cfg(AT_CUDNN_ENABLED)]
pub fn raw_cudnn_convolution_backward_weight_out_32bit(
        grad_weight:   &Tensor,
        grad_output:   &Tensor,
        input:         &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            auto dataType = getCudnnDataType(input);

      ConvolutionArgs args{ input, grad_output, grad_weight };
      args.handle = getCudnnHandle();
      setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups, deterministic, allow_tf32);
      args.idesc.set(input);
      args.wdesc.set(grad_weight, input.suggest_memory_format(), 0);
      args.odesc.set(grad_output);
      args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);

      AlgoIterator<cudnnConvolutionBwdFilterAlgoPerf_t>(args, benchmark).try_all(
        [&](const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgPerf){
          Tensor workspace = allocate_workspace(bwdFilterAlgPerf.memory, input);

          // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
          // whether to use Tensor core kernels or not
          // See Note [behavior of cudnnFind and cudnnGet]
          ASSERT_CORRECT_PRECISION(bwdFilterAlgPerf.mathType);
          AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdFilterAlgPerf.mathType), args);

          Constant one(dataType, 1);
          Constant zero(dataType, 0);

          AT_CUDNN_CHECK_WITH_SHAPES(cudnnConvolutionBackwardFilter(
              args.handle,
              &one, args.idesc.desc(), input.data_ptr(),
              args.odesc.desc(), grad_output.data_ptr(),
              args.cdesc.desc(), bwdFilterAlgPerf.algo, workspace.data_ptr(), bwdFilterAlgPerf.memory,
              &zero, args.wdesc.desc(), grad_weight.data_ptr()),
            args,
            "Additional pointer addresses: \n",
            "    grad_output: ", grad_output.data_ptr(), "\n",
            "    grad_weight: ", grad_weight.data_ptr(), "\n",
            "Backward filter algorithm: ", static_cast<int>(bwdFilterAlgPerf.algo), "\n");
        }
      );
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn raw_cudnn_convolution_backward_weight_out(
        grad_weight:   &Tensor,
        grad_output:   &Tensor,
        input:         &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            constexpr i64 int_max = int::max;
      const i64 ni = input.numel();
      const i64 no = grad_output.numel();
      // Assume the shape of the tensor is (N, C, D1, D2, ...)
      // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
      if (ni <= int_max && no <= int_max) {
        raw_cudnn_convolution_backward_weight_out_32bit(grad_weight, grad_output, input, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
        return;
      }
      // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
      //
      // Here we use a simple heuristics to determine the size of each split
      // We don't max out the 2^31 address space because this number is super
      // large and very likely to get an OOM.
      i64 n = grad_output.size(0);
      i64 max_inner_size = max<i64>(ni, no) / n;
      i64 split_size = max<i64>(1024 * 1024 * 512 / max_inner_size, 1L);
      i64 num_splits = (n + split_size - 1) / split_size;
      if (split_size * max_inner_size < int_max) {
        for (i64 i = 0; i < num_splits; i++) {
          i64 start = split_size * i;
          i64 split_size_ = min<i64>(split_size, n - start);
          Tensor input_ = input.narrow(0, start, split_size_);
          Tensor grad_output_ = grad_output.narrow(0, start, split_size_);
          Tensor grad_weight_ = empty_like(grad_weight);
          raw_cudnn_convolution_backward_weight_out_32bit(grad_weight_, grad_output_, input_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
          grad_weight.add_(grad_weight_);
        }
        return;
      }
      // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
      // For example, for conv2d, there following questions needs to be considered.
      // - Is the memory layout NCHW or NHWC ?
      // - If the conv is NCHW -> NC'H'W', then should we
      //   - split only NC?
      //   - split only N'C'?
      //   - split both?
      // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
      //   to make sure that the boundary is handled correctly.
      // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
      // Considering the complexity of this issue, it is better not to use cuDNN for this case
      TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
        */
}

#[cfg(AT_CUDNN_ENABLED)]
pub fn raw_cudnn_convolution_add_relu_out(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        z:             &Tensor,
        alpha:         f32,
        bias:          &Tensor,
        stride:        &[i32],
        padding:       &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        allow_tf32:    bool)  {
    
    todo!();
        /*
            auto dataType = getCudnnDataType(input);
      ConvolutionArgs args{input, output, weight};
      args.handle = getCudnnHandle();
      setConvolutionParams(
          &args.params,
          input,
          weight,
          padding,
          stride,
          dilation,
          groups,
          deterministic,
          allow_tf32);
      args.idesc.set(input);
      args.wdesc.set(weight, input.suggest_memory_format(), 0);
      args.odesc.set(output);
      args.cdesc.set(
          dataType,
          input.dim() - 2,
          args.params.padding,
          args.params.stride,
          args.params.dilation,
          args.params.groups,
          args.params.allow_tf32);

      TensorDescriptor zdesc;
      zdesc.set(z);

      TensorDescriptor bdesc;
      bdesc.set(bias.expand({1, bias.size(0)}), output.dim());

      ActivationDescriptor adesc;
      adesc.set(CUDNN_ACTIVATION_RELU);

      AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark)
          .try_all([&](const cudnnConvolutionFwdAlgoPerf_t& fwdAlgPerf) {
            Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

            // update convDesc mathType since cudnn 7.4+ now requires both algo +
            // mathType to figure out whether to use Tensor core kernels or not See
            // Note [behavior of cudnnFind and cudnnGet]
            ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
            AT_CUDNN_CHECK_WITH_SHAPES(
                cudnnSetConvolutionMathType(
                    args.cdesc.mut_desc(), fwdAlgPerf.mathType),
                args);

            Constant one(dataType, 1);
            Constant alpha_(dataType, alpha);

            AT_CUDNN_CHECK_WITH_SHAPES(
                cudnnConvolutionBiasActivationForward(
                    args.handle,
                    &one,
                    args.idesc.desc(),
                    input.data_ptr(),
                    args.wdesc.desc(),
                    weight.data_ptr(),
                    args.cdesc.desc(),
                    fwdAlgPerf.algo,
                    workspace.data_ptr(),
                    fwdAlgPerf.memory,
                    &alpha_,
                    zdesc.desc(),
                    z.data_ptr(),
                    bdesc.desc(),
                    bias.data_ptr(),
                    adesc.desc(),
                    args.odesc.desc(),
                    output.data_ptr()),
                args,
                "cudnnConvolutionBiasActivationForward: ",
                static_cast<int>(fwdAlgPerf.algo),
                "\n");
          });
        */
}
