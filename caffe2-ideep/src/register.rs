crate::ix!();

declare_registry!{
    IDEEPOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

// Boolean operators
register_ideep_compare_operator!{EQ}
register_ideep_compare_operator!{GT}
register_ideep_compare_operator!{GE}
register_ideep_compare_operator!{LT}
register_ideep_compare_operator!{LE}
register_ideep_compare_operator!{NE}

register_ideep_operator!{
    Softmax,
    IDEEPFallbackOp::<SoftmaxOp::<f32,CPUContext>>
}
register_ideep_operator!{
    LabelCrossEntropy,
    IDEEPFallbackOp::<LabelCrossEntropyOp::<f32,CPUContext>>
}
register_ideep_operator!{
    AveragedLoss,
    IDEEPFallbackOp::<AveragedLoss::<f32,CPUContext>,SkipIndices::<0>>
}
register_ideep_operator!{
    Flatten,
    IDEEPFallbackOp::<FlattenOp::<CPUContext>>
}

register_ideep_operator!{
    ResizeLike,
    IDEEPFallbackOp::<ResizeLikeOp::<CPUContext>>
}

register_ideep_operator!{
    Slice,
    IDEEPFallbackOp::<SliceOp::<CPUContext>>
}

register_ideep_operator!{
    Clip,
    IDEEPFallbackOp::<ClipOp::<f32,CPUContext>>
}

register_ideep_operator!{
    ScatterAssign,
    IDEEPFallbackOp::<ScatterAssignOp::<CPUContext>>
}

register_ideep_operator!{
    Cast,
    IDEEPFallbackOp::<CastOp::<CPUContext>>
}

// filter operators
register_ideep_operator!{
    XavierFill,
    IDEEPFallbackOp::<XavierFillOp<f32,CPUContext>>
}

register_ideep_operator!{
    ConstantFill,
    IDEEPFallbackOp::<ConstantFillOp::<CPUContext>>
}

register_ideep_operator!{
    GaussianFill,
    IDEEPFallbackOp::<GaussianFillOp::<f32,CPUContext>>
}

register_ideep_operator!{
    MSRAFill,
    IDEEPFallbackOp::<MSRAFillOp::<f32,CPUContext>>
}

register_ideep_operator!{
    GivenTensorFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<f32,CPUContext>>
}

// Not supported tensor types in below FillOp
register_ideep_operator!{
    GivenTensorDoubleFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<f64,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorBoolFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<bool,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorIntFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<i32,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorInt64Fill,
    IDEEPFallbackOp::<GivenTensorFillOp::<i64,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    GivenTensorStringFill,
    IDEEPFallbackOp::<GivenTensorFillOp::<String,CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    Load,
    IDEEPFallbackOp::<LoadOp::<CPUContext>>
}

register_ideep_operator!{
    Save,
    IDEEPFallbackOp::<SaveOp::<CPUContext>>
}

register_ideep_operator!{
    RMACRegions,
    IDEEPFallbackOp::<RMACRegionsOp::<CPUContext>>
}

register_ideep_operator!{
    RoIPool,
    IDEEPFallbackOp::<RoIPoolOp::<f32,CPUContext>>
}

register_ideep_operator!{
    RoIAlign,
    IDEEPFallbackOp::<RoIAlignOp::<f32,CPUContext>>
}

register_ideep_operator!{
    RoIAlignRotated,
    IDEEPFallbackOp::<RoIAlignRotatedOp::<f32,CPUContext>>
}

register_ideep_operator!{
    GenerateProposals,
    IDEEPFallbackOp::<GenerateProposalsOp::<CPUContext>>
}

register_ideep_operator!{
    GenerateProposalsCPP,
    IDEEPFallbackOp::<GenerateProposalsOp::<CPUContext>>
}

register_ideep_operator!{
    CollectAndDistributeFpnRpnProposals,
    IDEEPFallbackOp::<CollectAndDistributeFpnRpnProposalsOp::<CPUContext>>
}

register_ideep_operator!{
    BoxWithNMSLimit,
    IDEEPFallbackOp::<BoxWithNMSLimitOp::<CPUContext>,SkipIndices::<0,1,2>>
}

register_ideep_operator!{
    BBoxTransform,
    IDEEPFallbackOp::<BBoxTransformOp::<f32,CPUContext>>
}

register_ideep_operator!{
    AffineChannel,
    IDEEPFallbackOp::<AffineChannelOp::<f32,CPUContext>>
}

register_ideep_operator!{
    StopGradient,
    IDEEPFallbackOp::<StopGradientOp::<CPUContext>>
}

register_ideep_operator!{
    PadImage,
    IDEEPFallbackOp::<PadImageOp::<f32,CPUContext>>
}

register_ideep_operator!{
    PRelu,
    IDEEPFallbackOp::<PReluOp::<f32,CPUContext>>
}

// ctc decoder operators
register_ideep_operator!{
    CTCGreedyDecoder,
    IDEEPFallbackOp::<CTCGreedyDecoderOp::<CPUContext>>
}

register_ideep_operator!{
    CTCBeamSearchDecoder,
    IDEEPFallbackOp::<CTCBeamSearchDecoderOp::<CPUContext>>
}

register_ideep_operator!{
    AveragedLossGradient,
    IDEEPFallbackOp::<AveragedLossGradient::<f32,CPUContext>>
}

register_ideep_operator!{
    LabelCrossEntropyGradient,
    IDEEPFallbackOp::<LabelCrossEntropyGradientOp::<f32,CPUContext>>
}

register_ideep_operator!{
    SoftmaxGradient,
    IDEEPFallbackOp::<SoftmaxGradientOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Iter,
    IDEEPFallbackOp::<IterOp::<CPUContext>>
}

register_ideep_operator!{
    LearningRate,
    IDEEPFallbackOp::<LearningRateOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Abs,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,AbsFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Atan,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,AtanFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Sqrt,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,SqrtFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Sign,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,SignFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Div,
    IDEEPFallbackOp::<BinaryElementwiseOp::<NumericTypes,CPUContext,DivFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Mul,
    IDEEPFallbackOp::<BinaryElementwiseOp::<NumericTypes,CPUContext,MulFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Sub,
    IDEEPFallbackOp::<BinaryElementwiseOp::<NumericTypes,CPUContext,SubFunctor::<CPUContext>>>
}

register_ideep_operator!{
    Tanh,
    IDEEPFallbackOp::<UnaryElementwiseOp::<TensorTypes::<f32>,CPUContext,TanhFunctor::<CPUContext>>>
}

register_ideep_operator!{
    L1Distance,
    IDEEPFallbackOp::<L1DistanceOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Scale,
    IDEEPFallbackOp::<ScaleOp::<CPUContext>>
}

register_ideep_operator!{
    Accuracy,
    IDEEPFallbackOp::<AccuracyOp::<f32,CPUContext>>
}

register_ideep_operator!{
    AddGradient,
    IDEEPFallbackOp::<BinaryElementwiseGradientOp::<NumericTypes,CPUContext,AddFunctor::<CPUContext>>>
}

register_ideep_operator!{
    TanhGradient,
    IDEEPFallbackOp::<BinaryElementwiseOp::<TensorTypes::<f32>,CPUContext,TanhGradientFunctor::<CPUContext>>>
}

register_ideep_operator!{
    MulGradient,
    IDEEPFallbackOp::<BinaryElementwiseGradientOp::<NumericTypes,CPUContext,MulFunctor::<CPUContext>>>
}

register_ideep_operator!{
    TensorProtosDBInput,
    IDEEPFallbackOp::<TensorProtosDBInput::<CPUContext>>
}

register_ideep_operator!{
    CloseBlobsQueue,
    IDEEPFallbackOp::<CloseBlobsQueueOp::<CPUContext>>
}

register_ideep_operator!{
    SoftmaxWithLoss,
    IDEEPFallbackOp::<SoftmaxWithLossOp::<f32,CPUContext>>
}

register_ideep_operator!{
    SoftmaxWithLossGradient,
    IDEEPFallbackOp::<SoftmaxWithLossGradientOp::<f32,CPUContext>>
}

register_ideep_operator!{
    Expand,
    IDEEPFallbackOp::<ExpandOp::<TensorTypes::<i32,i64,f32,double>,CPUContext>>
}

register_ideep_operator!{
    Gather,
    IDEEPFallbackOp::<GatherOp::<CPUContext>>
}

register_ideep_operator!{
    Normalize,
    IDEEPFallbackOp::<NormalizeOp::<f32,CPUContext>>
}

register_ideep_operator!{
    ReduceL2,
    IDEEPFallbackOp::<ReduceOp::<TensorTypes::<f32>,CPUContext,L2Reducer::<CPUContext>>>
}

register_ideep_operator!{
    ReduceSum,
    IDEEPFallbackOp::<ReduceOp::<TensorTypes::<i32,i64,f32,f64>,CPUContext,SumReducer::<CPUContext>>>
}

register_ideep_operator!{
    ReduceMean,
    IDEEPFallbackOp::<ReduceOp::<TensorTypes::<f32>,CPUContext,MeanReducer::<CPUContext>>>
}

register_ideep_operator!{
    BatchMatMul,
    IDEEPFallbackOp::<BatchMatMulOp::<CPUContext>>
}

register_ideep_operator!{
    CreateCommonWorld,
    IDEEPFallbackOp::<CreateCommonWorld::<CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    CloneCommonWorld,
    IDEEPFallbackOp::<CloneCommonWorld::<CPUContext>,SkipIndices::<0>>
}

register_ideep_operator!{
    DestroyCommonWorld,
    IDEEPFallbackOp::<DestroyCommonWorld>
}

register_ideep_operator!{
    Broadcast,
    IDEEPFallbackOp::<BroadcastOp::<CPUContext>>
}

register_ideep_operator!{
    Allreduce,
    IDEEPFallbackOp::<AllreduceOp::<CPUContext>>
}

register_ideep_operator!{
    Allgather,
    IDEEPFallbackOp::<AllgatherOp::<CPUContext>>
}

register_ideep_operator!{
    Barrier,
    IDEEPFallbackOp::<BarrierOp::<CPUContext>>
}

register_ideep_operator!{
    ReduceScatter,
    IDEEPFallbackOp::<ReduceScatterOp::<CPUContext>>
}

register_ideep_operator!{
    Adam, 
    IDEEPAdamOp::<f32>
}

register_ideep_operator!{
    ChannelShuffle, 
    IDEEPChannelShuffleOp
}

register_ideep_operator!{
    ChannelShuffleGradient, 
    ChannelShuffleGradientOp
}

register_ideep_operator!{
    Conv, 
    IDEEPConvOp
}

register_ideep_operator!{
    ConvFusion, 
    IDEEPConvFusionOp
}

register_ideep_operator!{
    ConvGradient, 
    IDEEPConvGradientOp
}

register_ideep_operator!{
    ConvTranspose, 
    IDEEPConvTransposeOp
}

register_ideep_operator!{
    ConvTransposeGradient, 
    IDEEPConvTransposeGradientOp
}

register_ideep_operator!{
    Dropout,     
    IDEEPDropoutOp
}

register_ideep_operator!{
    DropoutGrad, 
    IDEEPDropoutGradientOp
}

register_ideep_operator!{
    Sum, 
    IDEEPSumOp
}

register_ideep_operator!{
    Add, 
    IDEEPSumOp
}

register_ideep_operator!{
    ExpandDims, 
    IDEEPExpandDimsOp
}

register_ideep_operator!{
    Squeeze,    
    IDEEPSqueezeOp
}

register_ideep_operator!{
    FC, 
    IDEEPFullyConnectedOp
}

register_ideep_operator!{
    FCGradient, 
    IDEEPFullyConnectedGradientOp
}

register_ideep_operator!{
    LRN,         
    IDEEPLRNOp
}

register_ideep_operator!{
    LRNGradient, 
    IDEEPLRNGradientOp
}

register_ideep_operator!{
    MomentumSGD,       
    IDEEPMomentumSGDOp
}

register_ideep_operator!{
    MomentumSGDUpdate, 
    IDEEPMomentumSGDUpdateOp
}

register_ideep_operator_with_engine!{
    Int8Conv,         
    DNNLOWP, IDEEPInt8ConvOp
}

register_ideep_operator_with_engine!{
    Int8ConvRelu,     
    DNNLOWP, IDEEPInt8ConvReluOp
}

register_ideep_operator_with_engine!{
    Int8ConvSum,      
    DNNLOWP, IDEEPInt8ConvSumOp
}

register_ideep_operator_with_engine!{
    Int8ConvSumRelu,  
    DNNLOWP, IDEEPInt8ConvSumReluOp
}

register_ideep_operator!{
    NHWC2NCHW, 
    IDEEPNHWC2NCHWOp
}

register_ideep_operator!{
    NCHW2NHWC, 
    IDEEPNCHW2NHWCOp
}

register_ideep_operator!{
    MaxPool,             
    IDEEPPoolOp
}

register_ideep_operator!{
    MaxPoolGradient,     
    IDEEPPoolGradientOp
}

register_ideep_operator!{
    AveragePool,         
    IDEEPPoolOp
}

register_ideep_operator!{
    AveragePoolGradient, 
    IDEEPPoolGradientOp
}

register_ideep_operator_with_engine!{
    Int8Sum, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<false>
}

register_ideep_operator_with_engine!{
    Int8Add, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<false>
}

register_ideep_operator_with_engine!{
    Int8SumRelu, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<true>
}

register_ideep_operator_with_engine!{
    Int8AddRelu, 
    DNNLOWP, 
    IDEEPInt8SumReluOp::<true>
}

register_ideep_operator_with_engine!{
    Int8Dequantize, 
    DNNLOWP, IDEEPInt8DequantizeOp
}

register_ideep_operator!{
    Int8GivenTensorFill,    
    IDEEPInt8GivenTensorFillOp
}

register_ideep_operator!{
    Int8GivenIntTensorFill, 
    IDEEPInt8GivenIntTensorFillOp
}

register_ideep_operator_with_engine!{
    Int8MaxPool,     
    DNNLOWP, IDEEPInt8PoolOp
}

register_ideep_operator_with_engine!{
    Int8AveragePool, 
    DNNLOWP, IDEEPInt8PoolOp
}

register_ideep_operator_with_engine!{
    Int8Quantize, 
    DNNLOWP, 
    IDEEPInt8QuantizeOp
}

register_ideep_operator_with_engine!{
    Int8Relu, 
    DNNLOWP, 
    IDEEPInt8ReluOp
}

register_ideep_operator!{
    CreateBlobsQueue, 
    IDEEPCreateBlobsQueueOp
}

register_ideep_operator!{
    SafeEnqueueBlobs, 
    IDEEPSafeEnqueueBlobsOp
}

register_ideep_operator!{Relu,         IDEEPReluOp}
register_ideep_operator!{ReluGradient, IDEEPReluGradientOp}

register_ideep_operator!{LeakyRelu,         IDEEPReluOp}
register_ideep_operator!{LeakyReluGradient, IDEEPReluGradientOp}
register_ideep_operator!{Reshape, IDEEPReshapeOp}
register_ideep_operator!{Shape, IDEEPShapeOp}
register_ideep_operator!{Sigmoid,         IDEEPSigmoidOp}
register_ideep_operator!{SigmoidGradient, IDEEPSigmoidGradientOp}
register_ideep_operator!{SpatialBN,         IDEEPSpatialBNOp}
register_ideep_operator!{SpatialBNGradient, IDEEPSpatialBNGradientOp}
register_ideep_operator!{Transpose, IDEEPTransposeOp}
register_ideep_operator!{CopyCPUToIDEEP, CopyCPUToIDEEPOp}
register_ideep_operator!{CopyIDEEPToCPU, CopyIDEEPToCPUOp}
register_ideep_operator!{Copy, IDEEPCopyOp}
register_ideep_operator!{WeightedSum, IDEEPWeightedSumOp}

register_context!{
    DeviceType::IDEEP, 
    caffe2::IDEEPContext
}

register_copy_bytes_function!{
    DeviceType::IDEEP,
    DeviceType::CPU,
    CopyBytesWrapper
}

register_copy_bytes_function!{
    DeviceType::CPU,
    DeviceType::IDEEP,
    CopyBytesWrapper
}

register_copy_bytes_function!{
    DeviceType::IDEEP,
    DeviceType::IDEEP,
    CopyBytesWrapper
}

caffe_known_type!{ideep::tensor}

define_registry!{
    IDEEPOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

caffe_register_device_type!{
    DeviceType::IDEEP, 
    IDEEPOperatorRegistry 
}

register_event_create_function!{IDEEP,         EventCreateCPU}
register_event_record_function!{IDEEP,         EventRecordCPU}
register_event_wait_function!{IDEEP,           IDEEP, EventWaitCPUCPU}
register_event_wait_function!{IDEEP,           CPU,   EventWaitCPUCPU}
register_event_wait_function!{CPU,             IDEEP, EventWaitCPUCPU}
register_event_finish_function!{IDEEP,         EventFinishCPU}
register_event_query_function!{IDEEP,          EventQueryCPU}
register_event_error_message_function!{IDEEP,  EventErrorMessageCPU}
register_event_set_finished_function!{IDEEP,   EventSetFinishedCPU}
register_event_reset_function!{IDEEP,          EventResetCPU}

