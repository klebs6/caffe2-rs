crate::ix!();

define_int!{
    caffe2_tensor_chunk_size, 
    1000000, 
    "Chunk size to split tensor data into"
}

define_int!{
    caffe2_max_tensor_serializer_threads, 
    16,      
    "Maximal number of threads that can be used for tensor serialization"
}

define_bool!{
    caffe2_serialize_fp16_as_bytes, 
    false, 
    "Serialize FLOAT16 tensors using byte_data field"
}

define_bool!{
    caffe2_serialize_using_bytes_as_holder, 
    false, 
    "Serialize BOOL, UINT8, INT8, UINT16, INT16, INT64, FLOAT16 tensors using byte_data field instead of int32"
}

define_int64!{caffe2_test_big_tensor_size, 100000000, ""}

define_string!{caffe2_override_executor, "", "Comma-separated list of executor overrides"}

define_registry!{
    /*
    CPUOperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
}

define_registry!{
    /*
    CUDAOperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
}

define_registry!{
    /*
    HIPOperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
}
define_registry!{
    /*
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<GradientWrapper>&
    */
}

define_int!{
    caffe2_operator_max_engine_name_length,
    10,
    "Maximum engine name length to be stored"
}

define_bool!{
    caffe2_disable_implicit_engine_preference,
    false,
    "If set, disable implicit engine preferences. This is useful for unit testing and debugging cases."
}

define_bool!{
    caffe2_operator_throw_if_fp_exceptions,
    false,
    "If set, throws if floating point exceptions (FE_DIVBYZERO, FE_INVALID) are detected when running any operator. FE_OVERFLOW is handled separately by caffe2_operator_throw_if_fp_overflow_exceptions option."
}

define_bool!{
    caffe2_operator_throw_if_fp_overflow_exceptions,
    false,
    "If set, throws if floating point exception FE_OVERFLOW is detected when running any operator."
}

#[cfg(__gnu_library__)]
define_bool!{
    caffe2_operator_throw_on_first_occurrence_if_fp_exceptions,
    false,
    "If set with caffe2_operator_throw_if_fp_exceptions or caffe2_operator_throw_if_fp_overflow_exceptions, throw on the first occurrence of corresponding floating point exceptions that is detected when running any operator."
}

define_bool!{caffe2_force_shared_col_buffer,
    false,
    "Always use the shared col buffer"
}

define_bool!{
    caffe2_workspace_stack_debug, 
    false, 
    "Enable debug checks for CreateScope's workspace stack"
}

define_int!{
    caffe2_omp_num_threads, 
    0, 
    "The number of openmp threads. 0 to use default value. Does not have effect if OpenMP is disabled." 
}

define_int!{
    caffe2_mkl_num_threads, 
    0, 
    "The number of mkl threads. 0 to use default value. If set, this overrides the caffe2_omp_num_threads flag if both are set. Does not have effect if MKL is not used."
}

define_bool!{
    caffe2_quit_on_unsupported_cpu_feature,
    false,
    "If set, when Caffe2 is built with a CPU feature (like avx2) but the current CPU does not support it, quit early. If not set (by default), log this as an error message and continue execution."
}

define_int!{
    caffe2_ftz,
    false,
    "If true, turn on flushing denormals to zero (FTZ)"
}

define_int!{
    caffe2_daz,
    false,
    "If true, turn on replacing denormals loaded from memory with zero (DAZ)"
}

define_bool!{
    caffe2_version,
    false,
    "Print Caffe2 version and build options on startup"
}

define_bool!{
    caffe2_rnn_executor,
    true,
    "If set, uses special RNN executor for executing RecurrentNetworkOp"
}

define_registry!{
    /*
    Caffe2DBRegistry, 
    DB, 
    const string&, 
    Mode
    */
}

define_bool!{
    caffe2_print_blob_sizes_at_exit,
    false,
    "If true, workspace destructor will print all blob shapes"
}

define_registry!{ 
    /* TransformRegistry, Transform */ 
}

// experimental support for multiple streams per worker per GPU
define_int!{
    caffe2_streams_per_gpu,
    1,
    "Number of streams per worker per GPU to use in GPU thread pool (experimental)"
}

define_bool!{
    caffe2_net_async_inference_mode,
    false,
    "If set, use one single chain containing all ops"
}

define_bool!{
    caffe2_net_async_profile_operators,
    false,
    "If set, profile operators of the net regardless of net being prof_dag."
}

define_int!{
    caffe2_net_async_max_gpus,
    16,
    "Max number of GPUs allowed in net async executor"
}

define_int!{
    caffe2_net_async_max_numa_nodes,
    8,
    "Max number of NUMA nodes allowed in net async executor"
}

define_int!{
    caffe2_net_async_thread_pool_size,
    0,
    "Number of threads in device thread pool by default"
}

define_bool!{
    caffe2_net_async_check_stream_status,
    false,
    "Select next non-busy stream"
}

define_bool!{
    caffe2_net_async_use_single_pool,
    false,
    "Use single thread pool for all devices"
}

define_bool!{
    caffe2_net_async_use_per_net_pools,
    false,
    "Use per net thread pools"
}

define_bool!{
    caffe2_net_async_run_root_tasks_inline,
    false,
    "Run root tasks in current thread instread of scheduling to threadpool"
}

define_bool!{
    caffe2_threadpool_force_inline,
    false,
    "Force to always run jobs on the calling thread"
}

// Whether or not threadpool caps apply to Android
define_int!{caffe2_threadpool_android_cap, true, ""}

// Whether or not threadpool caps apply to iOS
define_int!{caffe2_threadpool_ios_cap, true, ""}

define_int!{pthreadpool_size, 0, "Override the default thread pool size."}

