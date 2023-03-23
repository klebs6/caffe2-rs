crate::ix!();

///-----------------------
/// Discriminator for LLVM-style RTTI (isa<>)
pub enum NNDataKind { 
    Generic, 
    Tensor 
}

///--------------------------
/// Discriminator for LLVM-style RTTI (isa<>)
pub enum NNKind {
    Undefined,
    GenericOperator,
    NNPhi,
    While,
    Relu, 
    Conv, 
    ConvRelu, 
    ConvTranspose, 
    AveragePool, 
    AveragePoolRelu, 
    MaxPool,
    MaxPoolRelu, 
    Sum, 
    SumRelu, 
    Send, 
    Receive, 
    BatchNormalization, 
    Clip, 
    FC,
    GivenTensorFill, 
    Concat, 
    Softmax, 
    ChannelShuffle, 
    Add, 
    Reshape, 
    Flatten,
    CopyToOpenCL, 
    CopyFromOpenCL, 
    NCHW2NHWC, 
    NHWC2NCHW, 
    Declare, 
    Export
}

/// An optional tensor-type specifier.
pub enum NNLayout { 
    Undefined, 
    NCHW, 
    NHWC
}

