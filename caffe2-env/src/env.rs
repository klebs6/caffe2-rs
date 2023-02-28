crate::ix!();

declare_int!{caffe2_tensor_chunk_size}

declare_int!{caffe2_max_tensor_serializer_threads}

declare_bool!{caffe2_serialize_fp16_as_bytes}

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

declare_int!{caffe2_tensor_chunk_size}

declare_bool!{caffe2_serialize_fp16_as_bytes}

declare_bool!{caffe2_serialize_using_bytes_as_holder}

declare_bool!{caffe2_report_cpu_memory_usage}

declare_string!{caffe2_override_executor}

define_string!{caffe2_override_executor, "", "Comma-separated list of executor overrides"}

declare_bool!{caffe2_operator_throw_if_fp_exceptions}

declare_bool!{caffe2_operator_throw_if_fp_overflow_exceptions}

declare_bool!{caffe2_operator_throw_on_first_occurrence_if_fp_exceptions}

define_registry!{
    /*
    CPUOperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
}

caffe_register_device_type!{CPU, CPUOperatorRegistry}

define_registry!{
    /*
    CUDAOperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
}

caffe_register_device_type!{CUDA, CUDAOperatorRegistry}

define_registry!{
    /*
    HIPOperatorRegistry,
    OperatorStorage,
    const OperatorDef&,
    Workspace*
    */
}

caffe_register_device_type!{HIP, HIPOperatorRegistry}

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

declare_typed_registry!{
    /*
    ExternalTensorFunctionsBaseRegistry,
    TypeIdentifier,
    ExternalTensorFunctionsBase,
    std::unique_ptr
    */
}

declare_export_caffe2_op_to_c10!{LSTMOp}

export_caffe2_op_to_c10_cpu!{
    InferenceLSTM,
    "_caffe2::InferenceLSTM(
        Tensor[] input_list, 
        int num_layers, 
        bool has_biases, 
        bool batch_first, 
        bool bidirectional) -> (Tensor output, 
        Tensor hidden, 
        Tensor cell)",
        InferenceLSTMOp
}

declare_string!{
    caffe_test_root
}

declare_export_caffe2_op_to_c10!{
    Bucketize
}

export_caffe2_op_to_c10_cpu!{
    Bucketize,
    "_caffe2::Bucketize(
        Tensor data, 
        float[] boundaries) -> Tensor output",
    BucketizeInt
}

declare_export_caffe2_op_to_c10!{
    AliasWithName
}

export_caffe2_op_to_c10_cpu!{
    AliasWithName,
    "_caffe2::AliasWithName(
        Tensor input, 
        str name, 
        bool is_backward = False) -> (Tensor output)",
        AliasWithNameOp<CPUContext>
}

declare_export_caffe2_op_to_c10!{
    BatchBoxCox
}

export_caffe2_op_to_c10_cpu!{
    BatchBoxCox,
    "_caffe2::BatchBoxCox(
        Tensor data, 
        Tensor lambda1, 
        Tensor lambda2, 
        int min_block_size = 256) -> Tensor results",
        BatchBoxCoxOp<CPUContext>
}

declare_export_caffe2_op_to_c10!{
    BatchPermutation
}

export_caffe2_op_to_c10_cpu!{
    BatchPermutation,
    "_caffe2::BatchPermutation(
        Tensor X, 
        Tensor indices) -> Tensor",
    BatchPermutationOpFloatCPU
}

declare_export_caffe2_op_to_c10!{
    BBoxTransform
}

export_caffe2_op_to_c10_cpu!{
    BBoxTransform,
    "_caffe2::BBoxTransform(
        Tensor rois, 
        Tensor deltas, 
        Tensor im_info, 
        float[] weights, 
        bool apply_scale, 
        bool rotated, 
        bool angle_bound_on, 
        int angle_bound_lo, 
        int angle_bound_hi, 
        float clip_angle_thresh, 
        bool legacy_plus_one) -> (Tensor output_0, Tensor output_1)",
        BBoxTransformOpFloatCPU
}

declare_export_caffe2_op_to_c10!{
    BoxWithNMSLimit
}

export_caffe2_op_to_c10_cpu!{
    BoxWithNMSLimit,
    "_caffe2::BoxWithNMSLimit(
        Tensor scores, 
        Tensor boxes, 
        Tensor batch_splits, 
        float score_thresh, 
        float nms, 
        int detections_per_im, 
        bool soft_nms_enabled, 
        str soft_nms_method, 
        float soft_nms_sigma, 
        float soft_nms_min_score_thres, 
        bool rotated, 
        bool cls_agnostic_bbox_reg, 
        bool input_boxes_include_bg_cls, 
        bool output_classes_include_bg_cls, 
        bool legacy_plus_one ) -> (Tensor scores, 
        Tensor boxes, 
        Tensor classes, 
        Tensor batch_splits, 
        Tensor keeps, 
        Tensor keeps_size)",
        BoxWithNMSLimitOp<CPUContext>
}

declare_export_caffe2_op_to_c10!{CollectAndDistributeFpnRpnProposals}
declare_export_caffe2_op_to_c10!{CollectRpnProposals}
declare_export_caffe2_op_to_c10!{DistributeFpnProposals}
declare_bool!{caffe2_force_shared_col_buffer}

define_bool!{caffe2_force_shared_col_buffer,
    false,
    "Always use the shared col buffer"}

declare_bool!{caffe2_force_shared_col_buffer}
declare_bool!{caffe2_force_shared_col_buffer}
declare_bool!{caffe2_force_shared_col_buffer}
declare_export_caffe2_op_to_c10!{CopyGPUToCPU}
declare_export_caffe2_op_to_c10!{CopyCPUToGPU}

export_caffe2_op_to_c10_schema_only!{
    CopyGPUToCPU, 
    "_caffe2::CopyGPUToCPU(Tensor input) -> Tensor"
}

export_caffe2_op_to_c10_schema_only!{
    CopyCPUToGPU, 
    "_caffe2::CopyCPUToGPU(Tensor input) -> Tensor"
}

declare_bool!{caffe2_workspace_stack_debug}

define_bool!{
    caffe2_workspace_stack_debug, 
    false, 
    "Enable debug checks for CreateScope's workspace stack"
}

declare_bool!{caffe2_force_shared_col_buffer}
declare_string!{caffe_test_root}
declare_string!{caffe_test_root}
declare_export_caffe2_op_to_c10!{HeatmapMaxKeypoint}

export_caffe2_op_to_c10_cpu!{
    HeatmapMaxKeypoint,
    "_caffe2::HeatmapMaxKeypoint(
        Tensor heatmaps, 
        Tensor bboxes_in, 
        bool should_output_softmax = True) -> Tensor keypoints",
        HeatmapMaxKeypointOpFloatCPU
}

declare_export_caffe2_op_to_c10!{Fused8BitRowwiseQuantizedToFloat}

export_caffe2_op_to_c10_cpu!{
    Fused8BitRowwiseQuantizedToFloat,
    "_caffe2::Fused8BitRowwiseQuantizedToFloat(
        Tensor scale_bias_quantized_input) -> Tensor",
    Fused8BitRowwiseQuantizedToFloatCPUOp
}

declare_export_caffe2_op_to_c10!{GatherRangesToDense}

export_caffe2_op_to_c10_cpu!{
    GatherRangesToDense,
    "_caffe2::GatherRangesToDense(
        Tensor data, 
        Tensor ranges, 
        Tensor? key, 
        int[] lengths, 
        int min_observation, 
        float max_mismatched_ratio, 
        float max_empty_ratio) -> Tensor[] outputs",
        GatherRangesToDenseCPUOp
}

declare_export_caffe2_op_to_c10!{Gelu}

export_caffe2_op_to_c10_cpu!{
    Gelu,
    "_caffe2::Gelu(
        Tensor input, 
        bool fast_gelu = False) -> (Tensor output)",
        GeluOp<CPUContext>
}

declare_export_caffe2_op_to_c10!{GenerateProposals}

declare_string!{caffe_test_root}

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

declare_export_caffe2_op_to_c10!{PiecewiseLinearTransform}

export_caffe2_op_to_c10_cpu!{
    PiecewiseLinearTransform,
    "_caffe2::PiecewiseLinearTransform(
        Tensor predictions, 
        float[] bounds, 
        float[] slopes, 
        float[] intercepts, 
        bool binary) -> (Tensor output_0)",
        PiecewiseLinearTransformOpFloatCPU
}

export_caffe2_op_to_c10_cpu!{
    Logit,
    "_caffe2::Logit(
        Tensor X, 
        float eps = 1e-6)->Tensor Y",
        LogitOp
}

declare_export_caffe2_op_to_c10!{Logit}
declare_export_caffe2_op_to_c10!{LayerNorm}

export_c10_op_to_caffe2_cpu!{
    "_caffe2::LayerNorm",
    C10LayerNorm_DontUseThisOpYet
}

export_caffe2_op_to_c10_cpu!{
    LayerNorm,
    "_caffe2::LayerNorm(
        Tensor X, 
        Tensor? gamma, 
        Tensor? beta, 
        int axis = 1, 
        float epsilon = 1e-5,
        bool elementwise_affine = False) -> (Tensor Y, 
        Tensor mean, 
        Tensor std)",
        LayerNormOp<CPUContext>
}

declare_int32!{caffe2_dnnlowp_activation_quantization_precision}
declare_int32!{caffe2_dnnlowp_weight_quantization_precision}
declare_int32!{caffe2_dnnlowp_requantization_multiplier_precision}
declare_int32!{caffe2_dnnlowp_eltwise_quantization_precision}
declare_bool!{caffe2_dnnlowp_force_scale_power_of_two}
declare_bool!{caffe2_dnnlowp_preserve_activation_sparsity}
declare_bool!{caffe2_dnnlowp_preserve_weight_sparsity}
declare_string!{caffe2_dnnlowp_activation_quantization_kind}
declare_string!{caffe2_dnnlowp_weight_quantization_kind }
declare_double!{caffe2_dnnlowp_weight_p99_threshold}
declare_double!{caffe2_dnnlowp_activation_p99_threshold}
declare_string!{caffe_test_root}
declare_string!{caffe_test_root}
declare_export_caffe2_op_to_c10!{IndexHash}

export_caffe2_op_to_c10_cpu!{
    IndexHash,
    "_caffe2::IndexHash(Tensor indices, int seed, int modulo) -> Tensor hashed_indices",
    IndexHashOp<CPUContext>
}

define_bool!{
    caffe2_rnn_executor,
    true,
    "If set, uses special RNN executor for executing RecurrentNetworkOp"
}

declare_bool!{caffe2_rnn_executor}

export_caffe2_op_to_c10_cpu!{
    RoIAlignGradient,
    "_caffe2::RoIAlignGradient(
        Tensor features, 
        Tensor rois, 
        Tensor grad, 
        str order, 
        float spatial_scale, 
        int pooled_h, 
        int pooled_w, 
        int sampling_ratio, 
        bool aligned) -> Tensor",
        RoIAlignGradientCPUOp<f32>
}

declare_export_caffe2_op_to_c10!{RoIAlignGradient}

export_caffe2_op_to_c10_cpu!{
    RoIAlign,
    "_caffe2::RoIAlign(
        Tensor features, 
        Tensor rois, 
        str order, 
        float spatial_scale, 
        int pooled_h, 
        int pooled_w, 
        int sampling_ratio, 
        bool aligned) -> Tensor",
        RoIAlignCPUOp<f32>
}

declare_export_caffe2_op_to_c10!{RoIAlign}
declare_export_caffe2_op_to_c10!{GatherRangesOp}
declare_export_caffe2_op_to_c10!{LengthsGatherOp}

export_caffe2_op_to_c10_cpu!{
    GatherRanges,
    "_caffe2::GatherRanges(
        Tensor data, 
        Tensor ranges) -> (Tensor, Tensor)",
    GatherRangesOp<CPUContext>
}

export_caffe2_op_to_c10_cpu!{
    LengthsGather,
    "_caffe2::LengthsGather(
        Tensor data, 
        Tensor lengths, 
        Tensor indices) -> (Tensor)",
    LengthsGatherOp<CPUContext>
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

declare_bool!{caffe2_print_blob_sizes_at_exit}
define_registry!{ /* TransformRegistry, Transform */ }
declare_registry!{ /* TransformRegistry, Transform */}
declare_int!{caffe2_streams_per_gpu}
declare_int!{caffe2_net_async_max_gpus}
declare_int!{caffe2_net_async_max_numa_nodes}
declare_int!{caffe2_net_async_thread_pool_size}
declare_bool!{caffe2_net_async_check_stream_status}
declare_bool!{caffe2_net_async_use_single_pool}
declare_bool!{caffe2_net_async_use_per_net_pools}
declare_bool!{caffe2_net_async_run_root_tasks_inline}
declare_bool!{caffe2_net_async_profile_operators}

register_creator!{
    /*
    ThreadPoolRegistry,
    CPU,
    GetAsyncNetThreadPool<TaskThreadPool, PROTO_CPU>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    CUDA,
    GetAsyncNetThreadPool<TaskThreadPool, PROTO_CUDA>
    */
}

register_creator!{
    /*
    ThreadPoolRegistry,
    HIP,
    GetAsyncNetThreadPool<TaskThreadPool, PROTO_HIP>
    */
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

declare_export_caffe2_op_to_c10!{PackSegments}
declare_export_caffe2_op_to_c10!{UnpackSegments}

export_caffe2_op_to_c10_cpu!{
    PackSegments,
    "_caffe2::PackSegments(
        Tensor lengths, 
        Tensor tensor, 
        int max_length = -1, 
        bool pad_minf = False, 
        bool return_presence_mask = False) -> (Tensor packed_tensor, 
        Tensor presence_mask)",
        PackSegmentsOp<CPUContext>
}

export_caffe2_op_to_c10_cpu!{
    UnpackSegments,
    "_caffe2::UnpackSegments(
        Tensor lengths, 
        Tensor tensor, 
        int max_length = -1) -> (Tensor packed_tensor)",
        UnpackSegmentsOp<CPUContext>
}

declare_export_caffe2_op_to_c10!{BatchBucketOneHot}

export_caffe2_op_to_c10_cpu!{
    BatchBucketOneHot,
    "_caffe2::BatchBucketOneHot(
        Tensor data, 
        Tensor lengths, 
        Tensor boundaries) -> Tensor output",
    BatchBucketOneHotOp<CPUContext>
}

declare_export_caffe2_op_to_c10!{MergeIdLists}

export_caffe2_op_to_c10_cpu!{
    MergeIdLists,
    "_caffe2::MergeIdLists(
        Tensor[] lengths_and_values) -> (Tensor merged_lengths, Tensor merged_values)",
    MergeIdListsOp<CPUContext>
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

declare_string!{caffe_test_root}

declare_export_caffe2_op_to_c10!{LearningRate}

export_caffe2_op_to_c10_cpu!{
    LearningRate,
"_caffe2::LearningRate(
        Tensor iterations, 
        float base_lr, 
        str policy,  
        float? power = 1.0,  
        float? gamma = 1.0,  
        int? stepsize = 1,  
        float? max_lr = 0.005,  
        bool? active_first = True,  
        int? active_period = -1,  
        int? inactive_period = -1,  
        int? max_iter = -1,  
        int? num_iter = 0,  
        float? start_multiplier = 0,  
        float? end_multiplier = 0,  
        float? multiplier = 0.5,  
        float? multiplier_1 = 1.0,  
        float? multiplier_2 = 1.0,  
        int[]? sub_policy_num_iters = None,  
        float? m1 = 0.5,  
        float? n1 = 0,  
        float? m2 = 0.5,  
        float? n2 = 0,  
        float? m3 = 0.5,  
        float? start_warmup_multiplier = 0.1,  
        int? constant_warmup_num_iter = 10000000,  
        int? linear_warmup_num_iter = 10000000,  
        float? cyclical_max_lr = 0.05,  
        int? cyclical_step_size = 1000000,  
        float? cyclical_decay = 0.999,  
        float? cosine_min_lr = 0.01,  
        float? cosine_max_lr = 0.05,  
        int? cosine_period = 50,  
        float? cosine_t_mult = 1.0,  
        float? cosine_lr_shrink = 0.99,  
        float? decay = 1.0) -> Tensor output",
        LearningRateOpFloatCPU
}
