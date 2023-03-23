crate::ix!();

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

export_caffe2_op_to_c10_cpu!{
    Bucketize,
    "_caffe2::Bucketize(
        Tensor data, 
        float[] boundaries) -> Tensor output",
    BucketizeInt
}


export_caffe2_op_to_c10_cpu!{
    AliasWithName,
    "_caffe2::AliasWithName(
        Tensor input, 
        str name, 
        bool is_backward = False) -> (Tensor output)",
        AliasWithNameOp<CPUContext>
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


export_caffe2_op_to_c10_cpu!{
    BatchPermutation,
    "_caffe2::BatchPermutation(
        Tensor X, 
        Tensor indices) -> Tensor",
    BatchPermutationOpFloatCPU
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


export_caffe2_op_to_c10_schema_only!{
    CopyGPUToCPU, 
    "_caffe2::CopyGPUToCPU(Tensor input) -> Tensor"
}

export_caffe2_op_to_c10_schema_only!{
    CopyCPUToGPU, 
    "_caffe2::CopyCPUToGPU(Tensor input) -> Tensor"
}


export_caffe2_op_to_c10_cpu!{
    HeatmapMaxKeypoint,
    "_caffe2::HeatmapMaxKeypoint(
        Tensor heatmaps, 
        Tensor bboxes_in, 
        bool should_output_softmax = True) -> Tensor keypoints",
        HeatmapMaxKeypointOpFloatCPU
}


export_caffe2_op_to_c10_cpu!{
    Fused8BitRowwiseQuantizedToFloat,
    "_caffe2::Fused8BitRowwiseQuantizedToFloat(
        Tensor scale_bias_quantized_input) -> Tensor",
    Fused8BitRowwiseQuantizedToFloatCPUOp
}


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


export_caffe2_op_to_c10_cpu!{
    Gelu,
    "_caffe2::Gelu(
        Tensor input, 
        bool fast_gelu = False) -> (Tensor output)",
        GeluOp<CPUContext>
}


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


export_caffe2_op_to_c10_cpu!{
    IndexHash,
    "_caffe2::IndexHash(Tensor indices, int seed, int modulo) -> Tensor hashed_indices",
    IndexHashOp<CPUContext>
}


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


export_caffe2_op_to_c10_cpu!{
    BatchBucketOneHot,
    "_caffe2::BatchBucketOneHot(
        Tensor data, 
        Tensor lengths, 
        Tensor boundaries) -> Tensor output",
    BatchBucketOneHotOp<CPUContext>
}


export_caffe2_op_to_c10_cpu!{
    MergeIdLists,
    "_caffe2::MergeIdLists(
        Tensor[] lengths_and_values) -> (Tensor merged_lengths, Tensor merged_values)",
    MergeIdListsOp<CPUContext>
}


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
