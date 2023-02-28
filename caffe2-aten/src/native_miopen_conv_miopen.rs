/**
  | TODO: Remove the condition on feature = "rocm"
  | entirely, don't build this file as part
  | of CPU build.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/miopen/Conv_miopen.cpp]


/// See Note [ATen preprocessor philosophy]
#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution(
    input:         &Tensor,
    weight:        &Tensor,
    bias_opt:      &Option<Tensor>,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("miopen_convolution: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_backward_input(
    input_size:    &[i32],
    grad_output:   &Tensor,
    weight:        &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("miopen_convolution_backward_input: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_backward_weight(
    weight_size:   &[i32],
    grad_output:   &Tensor,
    input:         &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("miopen_convolution_backward_weight: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_backward_bias(grad_output: &Tensor) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("miopen_convolution_backward_bias: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_backward(
    input:         &Tensor,
    grad_output:   &Tensor,
    weight:        &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool,
    output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            AT_ERROR("miopen_convolution_backward: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_transpose(
    input:          &Tensor,
    weight:         &Tensor,
    bias_opt:       &Option<Tensor>,
    padding:        &[i32],
    output_padding: &[i32],
    stride:         &[i32],
    dilation:       &[i32],
    groups:         i64,
    benchmark:      bool,
    deterministic:  bool) -> Tensor {

    todo!();
        /*
            AT_ERROR("miopen_convolution_transpose: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_transpose_backward_input(
    grad_output:   &Tensor,
    weight:        &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {
    
    todo!();
        /*
            AT_ERROR("miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_transpose_backward_weight(
    weight_size:   &[i32],
    grad_output:   &Tensor,
    input:         &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {

    todo!();
        /*
            AT_ERROR("miopen_convolution_transpose_backward_weight: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_convolution_transpose_backward(
    input:          &Tensor,
    grad_output:    &Tensor,
    weight:         &Tensor,
    padding:        &[i32],
    output_padding: &[i32],
    stride:         &[i32],
    dilation:       &[i32],
    groups:         i64,
    benchmark:      bool,
    deterministic:  bool,
    output_mask:    [bool; 3]) -> (Tensor,Tensor,Tensor) {

    todo!();
        /*
            AT_ERROR("miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_depthwise_convolution(
    input:         &Tensor,
    weight:        &Tensor,
    bias_opt:      &Option<Tensor>,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {

    todo!();
        /*
            AT_ERROR("miopen_depthwise_convolution: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_depthwise_convolution_backward_input(
    input_size:    &[i32],
    grad_output:   &Tensor,
    weight:        &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {

    todo!();
        /*
            AT_ERROR("miopen_depthwise_convolution_backward_input: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_depthwise_convolution_backward_weight(
    weight_size:   &[i32],
    grad_output:   &Tensor,
    input:         &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool) -> Tensor {

    todo!();
        /*
            AT_ERROR("miopen_depthwise_convolution_backward_weight: ATen not compiled with MIOpen support");
        */
}

#[cfg(not(feature = "rocm"))]
pub fn miopen_depthwise_convolution_backward(
    input:         &Tensor,
    grad_output:   &Tensor,
    weight:        &Tensor,
    padding:       &[i32],
    stride:        &[i32],
    dilation:      &[i32],
    groups:        i64,
    benchmark:     bool,
    deterministic: bool,
    output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            AT_ERROR("miopen_depthwise_convolution_backward: ATen not compiled with MIOpen support");
        */
}

#[cfg(feature = "rocm")]
pub fn narrow_group(
        t:         &Tensor,
        dim:       i32,
        group_idx: i32,
        groups:    i64) -> Tensor {
    
    todo!();
        /*
            auto group_size = t.size(dim) / groups;
      return t.narrow(dim, group_idx * group_size, group_size);
        */
}

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

/// Used on pad, stride and dilation
#[cfg(feature = "rocm")]
pub fn check_args(
        c:             CheckedFrom,
        args:          &[i32],
        expected_size: usize,
        arg_name:      *const u8)  {
    
    todo!();
        /*
            TORCH_CHECK(args.size() <= expected_size,
               "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
               expected_size, " (while checking arguments for ", c, ")");
      TORCH_CHECK(args.size() >= expected_size,
               "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
               expected_size, " (while checking arguments for ", c, ")");

      auto num_negative_values = count_if(args.begin(), args.end(), [](int x){return x < 0;});
      if (num_negative_values > 0){
        stringstream ss;
        ss << arg_name << " should be greater than zero but got (";
        copy(args.begin(), args.end() - 1, ostream_iterator<int>(ss,", "));
        ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
        AT_ERROR(ss.str());
      }
        */
}

/// see NOTE [ Convolution checks] in src/Aten/native/cudnn/Conv.cpp
#[cfg(feature = "rocm")]
pub fn convolution_shape_check(
        c:        CheckedFrom,
        input:    &TensorGeometryArg,
        weight:   &TensorGeometryArg,
        output:   &TensorGeometryArg,
        padding:  &[i32],
        stride:   &[i32],
        dilation: &[i32],
        groups:   i64)  {
    
    todo!();
        /*
            check_args(c, padding, input->dim() - 2, "padding");
      check_args(c, stride, padding.size(), "stride");
      check_args(c, dilation, padding.size(), "dilation");

      // Input
      checkDimRange(c, input, 3, 6 /* exclusive */);
      checkSize(c, input, input_channels_dim, weight->size(1) * groups);

      // Weight
      checkSameDim(c, input, weight);

      checkSameDim(c, input, output);
        */
}

/**
  | This POD struct is used to let us easily
  | compute hashes of the parameters
  |
  */
#[cfg(feature = "rocm")]
pub struct ConvolutionParams {

    handle:        MiOpenHandle,
    data_type:     miopen::DataType,
    input_size:    [i32; 2 + max_dim],
    input_stride:  [i32; 2 + max_dim],
    weight_size:   [i32; 2 + max_dim],
    padding:       [i32; max_dim],
    stride:        [i32; max_dim],
    dilation:      [i32; max_dim],
    groups:        i64,
    deterministic: bool,

    /**
      | This is needed to distinguish between
      | miopen handles of multiple gpus.
      |
      */
    device_id:     i32,


  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
}

/**
  | ConvolutionParams must be a POD because
  | we read out its memory contenst as char*
  | when hashing
  |
  */
#[cfg(feature = "rocm")]
static_assert!(is_pod<ConvolutionParams>::value, "ConvolutionParams not POD");

#[cfg(feature = "rocm")]
pub fn set_convolution_params(
        params:        *mut ConvolutionParams,
        handle:        MiOpenHandle,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        deterministic: bool)  {
    
    todo!();
        /*
            miopenDataType_t dataType = getMiopenDataType(input);
      memset(params, 0, sizeof(ConvolutionParams));
      params->dataType = dataType;
      params->handle = handle;
      // ASSERT(weight.dim() == input.dim())
      for (int i = 0; i != input.dim(); ++i) {
        params->input_size[i] = (int) input.size(i);
        params->input_stride[i] = (int) input.stride(i);
        params->weight_size[i] = (int) weight.size(i);
      }
      // ASSERT(padding.size() == stride.size())
      // ASSERT(padding.size() == dilation.size())
      for (usize i = 0; i != padding.size(); ++i) {
        params->padding[i] = padding[i];
        params->stride[i] = stride[i];
        params->dilation[i] = dilation[i];
      }
      params->groups = groups;
      params->deterministic = deterministic;
      int device_id;
      HIP_CHECK(hipGetDevice(&device_id));
      params->device_id = device_id;
        */
}

/**
  | Convenience struct for passing around
  | descriptors and data pointers
  |
  */
#[cfg(feature = "rocm")]
pub struct ConvolutionArgs {
    handle: MiOpenHandle,
    params: ConvolutionParams,
    idesc:  TensorDescriptor,
    odesc:  TensorDescriptor,
    wdesc:  FilterDescriptor,
    input:  &Tensor,
    output: &Tensor,
    weight: &Tensor,
    cdesc:  ConvolutionDescriptor,
}

#[cfg(feature = "rocm")]
impl ConvolutionArgs {
    
    pub fn new(
        input:  &Tensor,
        output: &Tensor,
        weight: &Tensor) -> Self {
    
        todo!();
        /*
        : input(input),
        : output(output),
        : weight(weight),

        
        */
    }
}

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

/// Hashing machinery for ConvolutionParams
///
#[cfg(feature = "rocm")]
pub struct ParamsHash {

}

#[cfg(feature = "rocm")]
impl ParamsHash {
    
    pub fn invoke(&self, params: &ConvolutionParams) -> usize {
        
        todo!();
        /*
            auto ptr = reinterpret_cast<const u8*>(&params);
        u32 value = 0x811C9DC5;
        for (int i = 0; i < (int)sizeof(ConvolutionParams); ++i) {
          value ^= ptr[i];
          value *= 0x01000193;
        }
        return (usize)value;
        */
    }
}

#[cfg(feature = "rocm")]
pub struct ParamsEqual {

}

#[cfg(feature = "rocm")]
impl ParamsEqual {
    
    pub fn invoke(&self, 
        a: &ConvolutionParams,
        b: &ConvolutionParams) -> bool {
        
        todo!();
        /*
            auto ptr1 = reinterpret_cast<const u8*>(&a);
        auto ptr2 = reinterpret_cast<const u8*>(&b);
        return memcmp(ptr1, ptr2, sizeof(ConvolutionParams)) == 0;
        */
    }
}

#[cfg(feature = "rocm")]
pub struct BenchmarkCache<T> {
    mutex: Mutex,
    map:   HashMap<ConvolutionParams,T,ParamsHash,ParamsEqual>,
}

#[cfg(feature = "rocm")]
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

#[cfg(feature = "rocm")]
lazy_static!{
    /*
    BenchmarkCache<miopenConvFwdAlgorithm_t> fwd_algos;
    BenchmarkCache<miopenConvBwdDataAlgorithm_t> bwd_data_algos;
    BenchmarkCache<miopenConvBwdWeightsAlgorithm_t> bwd_filter_algos;

    BenchmarkCache<usize> fwd_wssizes;
    BenchmarkCache<usize> bwd_data_wssizes;
    BenchmarkCache<usize> bwd_filter_wssizes;
    */
}

#[cfg(feature = "rocm")]
pub struct Workspace {
    size: usize,
    data: *mut void,
}

#[cfg(feature = "rocm")]
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

#[cfg(feature = "rocm")]
impl Workspace {
    
    pub fn new(size: usize) -> Self {
    
        todo!();
        /*
        : size(size),
        : data(NULL),

            data = THCudaMalloc(globalContext().lazyInitCUDA(), size);
        */
    }
}

#[cfg(feature = "rocm")]
pub struct AlgorithmSearch<algo_t> {

}

#[cfg(feature = "rocm")]
pub fn get_workspace_size<A>(
    args:                        &ConvolutionArgs,
    miopen_conv_fwd_algorithm_t: A) -> usize {
    
    todo!();
        /*
            usize sz = 0;
        miopenConvolutionForwardGetWorkSpaceSize(
            args.handle,
            args.wdesc.desc(),
            args.idesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            &sz);
        return sz;
        */
}

#[cfg(feature = "rocm")]
pub fn get_workspace_size<A>(
    args:                             &ConvolutionArgs,
    miopen_conv_bwd_data_algorithm_t: A) -> usize {
    
    todo!();
        /*
            usize sz = 0;
        miopenConvolutionBackwardDataGetWorkSpaceSize(
            args.handle,
            args.odesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            &sz);
        return sz;
        */
}

#[cfg(feature = "rocm")]
pub fn get_workspace_size<A>(
    args:                                &ConvolutionArgs,
    miopen_conv_bwd_weights_algorithm_t: A) -> usize {
    
    todo!();
        /*
            usize sz = 0;
        miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            args.handle,
            args.odesc.desc(),
            args.idesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            &sz);
        return sz;
        */
}

/**
  | philosophy begins in wonder
  | 
  | - Aristotle
  |
  */
#[cfg(feature = "rocm")]
pub fn get_best_algorithm<perf_t>(
        perf_results:  *mut Perf,
        deterministic: bool,
        n_algo:        i32) -> Perf {

    todo!();
        /*
            return perfResults[0];
        */
}

#[cfg(feature = "rocm")]
pub struct AlgorithmSearchMiopenConvFwdAlgorithm {

}

#[cfg(feature = "rocm")]
pub mod algorithm_search_miopen_conv_fwd_algorithm {

    use super::*;

    pub type Perf = MiOpenConvAlgoPerf;
    pub type Algo = MiOpenConvFwdAlgorithm;

    pub const DEFAULT_ALGO: Auto = MiOpenConvolutionFwdAlgoGEMM;
}

#[cfg(feature = "rocm")]
impl AlgorithmSearchMiopenConvFwdAlgorithm {
    
    pub fn cache() -> &mut BenchmarkCache<Algo> {
        
        todo!();
        /*
            return fwd_algos;
        */
    }
    
    pub fn wsscache() -> &mut BenchmarkCache<usize> {
        
        todo!();
        /*
            return fwd_wssizes;
        */
    }
    
    pub fn find_algorithm(args: &ConvolutionArgs) -> Perf {
        
        todo!();
        /*
            int perf_count;
        perf_t perf_results;
        usize max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
        Workspace ws(max_ws_size);
        MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
            args.handle,
            args.idesc.desc(), args.input.data_ptr(),
            args.wdesc.desc(), args.weight.data_ptr(),
            args.cdesc.desc(),
            args.odesc.desc(), args.output.data_ptr(),
            1,        // just return the fastest
            &perf_count,
            &perf_results,
            ws.data,
            ws.size,
            false));
        return perf_results;
        */
    }
}

#[cfg(feature = "rocm")]
pub struct AlgorithmSearchMiOpenConvBwdDataAlgorithm {

}

#[cfg(feature = "rocm")]
pub mod algorithm_search_miopen_conv_bwd_data_algorithm {

    use super::*;

    pub type Perf = MiOpenConvAlgoPerf;
    pub type Algo = MiOpenConvBwdDataAlgorithm;

    pub const DEFAULT_ALGO: Auto = MiOpenConvolutionBwdDataAlgoGEMM;
}

#[cfg(feature = "rocm")]
impl AlgorithmSearchMiOpenConvBwdDataAlgorithm {
    
    pub fn cache() -> &mut BenchmarkCache<Algo> {
        
        todo!();
        /*
            return bwd_data_algos;
        */
    }
    
    pub fn wsscache() -> &mut BenchmarkCache<usize> {
        
        todo!();
        /*
            return bwd_data_wssizes;
        */
    }
    
    pub fn find_algorithm(args: &ConvolutionArgs) -> Perf {
        
        todo!();
        /*
            int perf_count;
        perf_t perf_results;
        usize max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
        Workspace ws(max_ws_size);
        MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
            args.handle,
            args.odesc.desc(), args.output.data_ptr(),
            args.wdesc.desc(), args.weight.data_ptr(),
            args.cdesc.desc(),
            args.idesc.desc(), args.input.data_ptr(),
            1,      // just return the fastest
            &perf_count,
            &perf_results,
            ws.data,
            ws.size,
            false));
        return perf_results;
        */
    }
}

#[cfg(feature = "rocm")]
pub struct AlgorithmSearchMiOpenConvBwdWeightsAlgorithm {

}

#[cfg(feature = "rocm")]
pub mod algorithm_search_miopen_conv_bwd_weights_algorithm {

    use super::*;

    pub type Perf = MiOpenConvAlgoPerf;
    pub type Algo = MiOpenConvBwdWeightsAlgorithm;

    pub const DEFAULT_ALGO: Auto = MiOpenConvolutionBwdWeightsAlgoGEMM;
}

#[cfg(feature = "rocm")]
impl AlgorithmSearchMiOpenConvBwdWeightsAlgorithm {
    
    pub fn cache() -> &mut BenchmarkCache<Algo> {
        
        todo!();
        /*
            return bwd_filter_algos;
        */
    }
    
    pub fn wsscache() -> &mut BenchmarkCache<usize> {
        
        todo!();
        /*
            return bwd_filter_wssizes;
        */
    }
    
    pub fn find_algorithm(args: &ConvolutionArgs) -> Perf {
        
        todo!();
        /*
            int perf_count;
        perf_t perf_results;
        usize max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
        Workspace ws(max_ws_size);
        MIOPEN_CHECK(miopenFindConvolutionBackwardWeightsAlgorithm(
            args.handle,
            args.odesc.desc(), args.output.data_ptr(),
            args.idesc.desc(), args.input.data_ptr(),
            args.cdesc.desc(),
            args.wdesc.desc(), args.weight.data_ptr(),
            1,      // just return the fastest
            &perf_count,
            &perf_results,
            ws.data,
            ws.size,
            false));
        return perf_results;
        */
    }
}


#[cfg(feature = "rocm")]
pub fn find_algorithm<algo_t>(
        args:      &ConvolutionArgs,
        benchmark: bool,
        algo:      *mut Algo)  {

    todo!();
        /*
            using search = algorithm_search<algo_t>;
      auto& cache = search::cache();
      auto& wsscache = search::wsscache();

      if (cache.find(args.params, algo)) {
        return;
      }

      if (args.params.deterministic && !benchmark) {
        *algo = search::DEFAULT_ALGO;
      }

      if (cache.find(args.params, algo)) {
        // re-check cache since another thread may have benchmarked the algorithm
        return;
      }

      auto perfResults = search::findAlgorithm(args);
      *algo = reinterpret_cast<algo_t&>(perfResults);

      cache.insert(args.params, *algo);
      wsscache.insert(args.params, perfResults.memory);

      hip::HIPCachingAllocator::emptyCache();
        */
}



#[cfg(feature = "rocm")]
pub fn choose_algorithm<algo_t>(
        args:      &ConvolutionArgs,
        benchmark: bool,
        algo:      *mut Algo) -> Workspace {

    todo!();
        /*
            findAlgorithm(args, benchmark, algo);

      using search = algorithm_search<algo_t>;
      usize workspace_size;
      search::wsscache().find(args.params, &workspace_size);
      try {
        return Workspace(workspace_size);
      } catch (const exception& e) {
        hipGetLastError(); // clear OOM error

        // switch to default algorithm and record it in the cache to prevent
        // further OOM errors
        *algo = search::DEFAULT_ALGO;
        workspace_size = getWorkspaceSize(args, *algo);
        search::cache().insert(args.params, *algo);
        search::wsscache().insert(args.params, workspace_size);
        return Workspace(workspace_size);
      }
        */
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// In-place!
#[cfg(feature = "rocm")]
pub fn miopen_convolution_add_bias(
        c:      CheckedFrom,
        output: &TensorArg,
        bias:   &TensorArg)  {
    
    todo!();
        /*
            checkAllSameType(c, {output, bias});
      checkAllSameGPU(c, {output, bias});
      checkSize(c, bias, { output->size(output_channels_dim) });

      TensorDescriptor bdesc, odesc;
      bdesc.set(bias->expand({1, bias->size(0)}), output->dim());
      odesc.set(*output);

      auto handle = getMiopenHandle();
      auto dataType = getMiopenDataType(*bias);
      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionForwardBias(handle, &one, bdesc.desc(), bias->data_ptr(),
                                         &zero, odesc.desc(), output->data_ptr()));
        */
}

// see NOTE [ Convolution design ] in src/Aten/native/cudnn/Conv.cpp

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

/**
  | The raw API directly invokes MIOpen.
  |
  | There are a few reasons this should never be directly exposed
  | via ATen:
  |
  |    - It takes output as a parameter (this should be computed!)
  |    - It doesn't do input checking
  |    - It doesn't resize output (it is assumed to be correctly sized)
  |
  */
#[cfg(feature = "rocm")]
pub fn raw_miopen_convolution_forward_out(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool)  {
    
    todo!();
        /*
            auto dataType = getMiopenDataType(input);
      miopenConvolutionMode_t c_mode = miopenConvolution;

      ConvolutionArgs args{ input, output, weight };
      args.handle = getMiopenHandle();
      setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
      args.idesc.set(input);
      args.wdesc.set(weight);
      args.odesc.set(output);
      args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

      miopenConvFwdAlgorithm_t fwdAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.data_ptr(),
        args.wdesc.desc(), weight.data_ptr(),
        args.cdesc.desc(), fwdAlg, &zero,
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_forward(
        c:             CheckedFrom,
        input:         &TensorArg,
        weight:        &TensorArg,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            checkAllSameType(c, {input, weight});
      checkAllSameGPU(c, {input, weight});

      auto output_t = empty(
                        conv_output_size(input->sizes(), weight->sizes(),
                                         padding, stride, dilation),
                        input->options());

      if (output_t.numel() == 0) {
        return output_t;
      }

      // Avoid ambiguity of "output" when this is being used as backwards
      TensorArg output{ output_t, "result", 0 };
      convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

      // See #4500
      Tensor weight_contig = weight->contiguous();

      raw_miopen_convolution_forward_out(
          *output, *input, weight_contig,
          padding, stride, dilation, groups, benchmark, deterministic);

      return *output;
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution(
        input_t:       &Tensor,
        weight_t:      &Tensor,
        bias_t_opt:    &Option<Tensor>,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_t_maybe_owned = borrow_from_optional_tensor(bias_t_opt);
      const Tensor& bias_t = *bias_t_maybe_owned;

      TensorArg input  { input_t,  "input",  1 },
                weight { weight_t, "weight", 2 },
                bias   { bias_t,   "bias",   3 };
      CheckedFrom c = "miopen_convolution";
      auto output_t = miopen_convolution_forward(
        c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
      if (bias->defined()) {
        miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
      }
      return output_t;
        */
}

/// Depthwise Convolutions
#[cfg(feature = "rocm")]
pub fn raw_miopen_depthwise_convolution_forward_out(
        output:        &Tensor,
        input:         &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool)  {
    
    todo!();
        /*
            auto dataType = getMiopenDataType(input);
      miopenConvolutionMode_t c_mode = miopenDepthwise;

      ConvolutionArgs args{ input, output, weight };
      args.handle = getMiopenHandle();
      setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
      args.idesc.set(input);
      args.wdesc.set(weight);
      args.odesc.set(output);
      args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

      miopenConvFwdAlgorithm_t fwdAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.data_ptr(),
        args.wdesc.desc(), weight.data_ptr(),
        args.cdesc.desc(), fwdAlg, &zero,
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution_forward(
        c:             CheckedFrom,
        input:         &TensorArg,
        weight:        &TensorArg,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            checkAllSameType(c, {input, weight});
      checkAllSameGPU(c, {input, weight});

      auto output_t = empty(
                        conv_output_size(input->sizes(), weight->sizes(),
                                         padding, stride, dilation),
                        input->options());

      TensorArg output{ output_t, "result", 0 };
      convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

      Tensor weight_contig = weight->contiguous();

      raw_miopen_depthwise_convolution_forward_out(
          *output, *input, weight_contig,
          padding, stride, dilation, groups, benchmark, deterministic);

      return *output;
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution(
        input_t:       &Tensor,
        weight_t:      &Tensor,
        bias_t_opt:    &Option<Tensor>,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_t_maybe_owned = borrow_from_optional_tensor(bias_t_opt);
      const Tensor& bias_t = *bias_t_maybe_owned;

      TensorArg input  { input_t,  "input",  1 },
                weight { weight_t, "weight", 2 },
                bias   { bias_t,   "bias",   3 };
      CheckedFrom c = "miopen_depthwise_convolution";
      auto output_t = miopen_depthwise_convolution_forward(
        c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
      if (bias->defined()) {
        miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
      }
      return output_t;
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_transpose_backward_input(
        grad_output_t: &Tensor,
        weight_t:      &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output { grad_output_t,  "grad_output", 1 },
                weight      { weight_t, "weight", 2 };
      return miopen_convolution_forward(
        "miopen_convolution_transpose_backward_input",
        grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_transpose_backward(
        input:          &Tensor,
        grad_output_t:  &Tensor,
        weight:         &Tensor,
        padding:        &[i32],
        output_padding: &[i32],
        stride:         &[i32],
        dilation:       &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool,
        output_mask:    [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor grad_output = grad_output_t.contiguous();

      Tensor grad_input, grad_weight, grad_bias;
      if (output_mask[0]) {
        grad_input = miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
      }
      if (output_mask[1]) {
        grad_weight = miopen_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
      }
      if (output_mask[2]) {
        grad_bias = miopen_convolution_backward_bias(grad_output);
      }

      return tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
        */
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub fn raw_miopen_convolution_backward_input_out(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool)  {
    
    todo!();
        /*
            auto dataType = getMiopenDataType(grad_output);
      miopenConvolutionMode_t c_mode = miopenConvolution;

      ConvolutionArgs args{ grad_input, grad_output, weight };
      args.handle = getMiopenHandle();
      setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);
      args.idesc.set(grad_input);
      args.wdesc.set(weight);
      args.odesc.set(grad_output);
      args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

      miopenConvBwdDataAlgorithm_t bwdDataAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one, args.odesc.desc(), grad_output.data_ptr(),
          args.wdesc.desc(), weight.data_ptr(),
          args.cdesc.desc(), bwdDataAlg, &zero,
          args.idesc.desc(), grad_input.data_ptr(), workspace.data, workspace.size));
        */
}

/// see NOTE [ Backward vs transpose convolutions ] in src/Aten/native/cudnn/Conv.cpp
///
#[cfg(feature = "rocm")]
pub fn miopen_convolution_backward_input(
        c:             CheckedFrom,
        input_size:    &[i32],
        grad_output:   &TensorArg,
        weight:        &TensorArg,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            checkAllSameType(c, {grad_output, weight});
      checkAllSameGPU(c, {grad_output, weight});

      auto grad_input_t = empty(input_size, grad_output->options());

      // Avoid "grad_input" when this is being used as transposed convolution
      TensorArg grad_input{ grad_input_t, "result", 0 };
      convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

      // See #4500
      Tensor weight_contig = weight->contiguous();

      raw_miopen_convolution_backward_input_out(
          *grad_input, *grad_output, weight_contig,
          padding, stride, dilation, groups, benchmark, deterministic);

      return *grad_input;
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_transpose_forward(
        c:              CheckedFrom,
        grad_output:    &TensorArg,
        weight:         &TensorArg,
        padding:        &[i32],
        output_padding: &[i32],
        stride:         &[i32],
        dilation:       &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool) -> Tensor {
    
    todo!();
        /*
            auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                        padding, output_padding, stride, dilation, groups);
      return miopen_convolution_backward_input(c, input_size, grad_output, weight,
                                        padding, stride, dilation, groups, benchmark, deterministic);
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_backward_input(
        input_size:    &[i32],
        grad_output_t: &Tensor,
        weight_t:      &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output{ grad_output_t, "grad_output", 1 },
                weight{ weight_t, "weight", 2 };
      return miopen_convolution_backward_input(
          "miopen_convolution_backward_input",
          input_size, grad_output, weight,
          padding, stride, dilation, groups, benchmark, deterministic);
        */
}

/// Depthwise convolutions backward data.
#[cfg(feature = "rocm")]
pub fn raw_miopen_depthwise_convolution_backward_input_out(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool)  {
    
    todo!();
        /*
            auto dataType = getMiopenDataType(grad_output);
      miopenConvolutionMode_t c_mode = miopenDepthwise;

      ConvolutionArgs args{ grad_input, grad_output, weight };
      args.handle = getMiopenHandle();
      setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);
      args.idesc.set(grad_input);
      args.wdesc.set(weight);
      args.odesc.set(grad_output);
      args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

      miopenConvBwdDataAlgorithm_t bwdDataAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one, args.odesc.desc(), grad_output.data_ptr(),
          args.wdesc.desc(), weight.data_ptr(),
          args.cdesc.desc(), bwdDataAlg, &zero,
          args.idesc.desc(), grad_input.data_ptr(), workspace.data, workspace.size));
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution_backward_input(
        c:             CheckedFrom,
        input_size:    &[i32],
        grad_output:   &TensorArg,
        weight:        &TensorArg,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            checkAllSameType(c, {grad_output, weight});
      checkAllSameGPU(c, {grad_output, weight});

      auto grad_input_t = empty(input_size, grad_output->options());

      TensorArg grad_input{ grad_input_t, "result", 0 };
      convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

      Tensor weight_contig = weight->contiguous();

      raw_miopen_depthwise_convolution_backward_input_out(
          *grad_input, *grad_output, weight_contig,
          padding, stride, dilation, groups, benchmark, deterministic);

      return *grad_input;
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution_backward_input(
        input_size:    &[i32],
        grad_output_t: &Tensor,
        weight_t:      &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output{ grad_output_t, "grad_output", 1 },
                weight{ weight_t, "weight", 2 };
      return miopen_depthwise_convolution_backward_input(
          "miopen_depthwise_convolution_backward_input",
          input_size, grad_output, weight,
          padding, stride, dilation, groups, benchmark, deterministic);
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_backward(
        input:         &Tensor,
        grad_output_t: &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor grad_output = grad_output_t.contiguous();

      Tensor grad_input, grad_weight, grad_bias;
      if (output_mask[0]) {
        grad_input = miopen_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
      }
      if (output_mask[1]) {
        grad_weight = miopen_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
      }
      if (output_mask[2]) {
        grad_bias = miopen_convolution_backward_bias(grad_output);
      }

      return tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution_backward(
        input:         &Tensor,
        grad_output_t: &Tensor,
        weight:        &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool,
        output_mask:   [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor grad_output = grad_output_t.contiguous();

      Tensor grad_input, grad_weight, grad_bias;
      if (output_mask[0]) {
        grad_input = miopen_depthwise_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
      }
      if (output_mask[1]) {
        grad_weight = miopen_depthwise_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
      }
      if (output_mask[2]) {
        grad_bias = miopen_convolution_backward_bias(grad_output);
      }

      return tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_transpose(
        input_t:        &Tensor,
        weight_t:       &Tensor,
        bias_t_opt:     &Option<Tensor>,
        padding:        &[i32],
        output_padding: &[i32],
        stride:         &[i32],
        dilation:       &[i32],
        groups:         i64,
        benchmark:      bool,
        deterministic:  bool) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_t_maybe_owned = borrow_from_optional_tensor(bias_t_opt);
      const Tensor& bias_t = *bias_t_maybe_owned;

      TensorArg input  { input_t,  "input",  1 },
                weight { weight_t, "weight", 2 },
                bias   { bias_t,   "bias",   3 };
      CheckedFrom c = "miopen_convolution_transpose";
      auto output_t = miopen_convolution_transpose_forward(
        c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      if (bias->defined()) {
        miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
      }
      return output_t;
        */
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub fn raw_miopen_convolution_backward_weight_out(
        grad_weight:   &Tensor,
        grad_output:   &Tensor,
        input:         &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool)  {
    
    todo!();
        /*
            auto dataType = getMiopenDataType(input);
      miopenConvolutionMode_t c_mode = miopenConvolution;

      ConvolutionArgs args{ input, grad_output, grad_weight };
      args.handle = getMiopenHandle();
      setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
      args.idesc.set(input);
      args.wdesc.set(grad_weight);
      args.odesc.set(grad_output);
      args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one, args.odesc.desc(), grad_output.data_ptr(),
          args.idesc.desc(), input.data_ptr(),
          args.cdesc.desc(), bwdFilterAlg, &zero,
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_backward_weight(
        c:             CheckedFrom,
        weight_size:   &[i32],
        grad_output:   &TensorArg,
        input:         &TensorArg,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            checkAllSameType(c, {grad_output, input});
      checkAllSameGPU(c, {grad_output, input});

      auto grad_weight_t = empty(weight_size, grad_output->options());

      // For uniformity with everything else, although it seems grad_weight
      // would be unambiguous too.
      TensorArg grad_weight{ grad_weight_t, "result", 0 };
      convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

      raw_miopen_convolution_backward_weight_out(
          *grad_weight, *grad_output, *input,
          padding, stride, dilation, groups, benchmark, deterministic);

      return grad_weight_t;
        */
}

/// Depthwise backward weights.
#[cfg(feature = "rocm")]
pub fn raw_miopen_depthwise_convolution_backward_weight_out(
        grad_weight:   &Tensor,
        grad_output:   &Tensor,
        input:         &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool)  {
    
    todo!();
        /*
            auto dataType = getMiopenDataType(input);
      miopenConvolutionMode_t c_mode = miopenDepthwise;

      ConvolutionArgs args{ input, grad_output, grad_weight };
      args.handle = getMiopenHandle();
      setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
      args.idesc.set(input);
      args.wdesc.set(grad_weight);
      args.odesc.set(grad_output);
      args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one, args.odesc.desc(), grad_output.data_ptr(),
          args.idesc.desc(), input.data_ptr(),
          args.cdesc.desc(), bwdFilterAlg, &zero,
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution_backward_weight(
        c:             CheckedFrom,
        weight_size:   &[i32],
        grad_output:   &TensorArg,
        input:         &TensorArg,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            checkAllSameType(c, {grad_output, input});
      checkAllSameGPU(c, {grad_output, input});

      auto grad_weight_t = empty(weight_size, grad_output->options());

      // For uniformity with everything else, although it seems grad_weight
      // would be unambiguous too.
      TensorArg grad_weight{ grad_weight_t, "result", 0 };
      convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

      raw_miopen_depthwise_convolution_backward_weight_out(
          *grad_weight, *grad_output, *input,
          padding, stride, dilation, groups, benchmark, deterministic);

      return grad_weight_t;
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_backward_weight(
        weight_size:   &[i32],
        grad_output_t: &Tensor,
        input_t:       &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output{ grad_output_t, "grad_output", 1 },
                input{ input_t, "input", 2 };
      return miopen_convolution_backward_weight(
          "miopen_convolution_backward_weight",
          weight_size, grad_output, input,
          padding, stride, dilation, groups, benchmark, deterministic);
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_convolution_transpose_backward_weight(
        weight_size:   &[i32],
        grad_output_t: &Tensor,
        input_t:       &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output{ grad_output_t, "grad_output", 1 },
                input{ input_t, "input", 2 };
      return miopen_convolution_backward_weight(
          "miopen_convolution_backward_weight",
          weight_size, input, grad_output,
          padding, stride, dilation, groups, benchmark, deterministic);
        */
}

#[cfg(feature = "rocm")]
pub fn miopen_depthwise_convolution_backward_weight(
        weight_size:   &[i32],
        grad_output_t: &Tensor,
        input_t:       &Tensor,
        padding:       &[i32],
        stride:        &[i32],
        dilation:      &[i32],
        groups:        i64,
        benchmark:     bool,
        deterministic: bool) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output{ grad_output_t, "grad_output", 1 },
                input{ input_t, "input", 2 };
      return miopen_depthwise_convolution_backward_weight(
          "miopen_depthwise_convolution_backward_weight",
          weight_size, grad_output, input,
          padding, stride, dilation, groups, benchmark, deterministic);
        */
}

// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub fn miopen_convolution_backward_bias(grad_output_t: &Tensor) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_output{ grad_output_t, "grad_output", 1 };

      auto grad_bias_t = empty( { grad_output->size(output_channels_dim) }, grad_output->options());

      TensorArg grad_bias{ grad_bias_t, "result", 0 };

      TensorDescriptor bdesc{grad_bias->expand({1, grad_bias->size(0)}),
                             static_cast<usize>(grad_output->dim())};
      TensorDescriptor odesc{*grad_output};

      auto handle = getMiopenHandle();
      auto dataType = getMiopenDataType(*grad_bias);
      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardBias(handle, &one, odesc.desc(), grad_output->data_ptr(),
                                                       &zero, bdesc.desc(), grad_bias->data_ptr()));
      return *grad_bias;
        */
}
