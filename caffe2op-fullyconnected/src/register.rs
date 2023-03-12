crate::ix!();

register_cuda_operator!{
    FC, 
    FullyConnectedOp<CUDAContext>
}

register_cuda_operator!{
    FCGradient, 
    FullyConnectedGradientOp<CUDAContext>
}

register_cuda_operator!{
    FCTransposed,
    FullyConnectedOp<
        CUDAContext,
        DefaultEngine,
        DontTransposeWeight>
}

register_cuda_operator!{
    FCTransposedGradient,
    FullyConnectedGradientOp<
        CUDAContext,
        DefaultEngine,
        DontTransposeWeight>
}

register_cuda_operator_with_engine!{
    FC,
    TENSORCORE,
    FullyConnectedOp<CUDAContext, TensorCoreEngine>
}

register_cuda_operator_with_engine!{
    FCGradient,
    TENSORCORE,
    FullyConnectedGradientOp<CUDAContext, TensorCoreEngine>
}

register_cuda_operator_with_engine!{
    FCTransposed,
    TENSORCORE,
    FullyConnectedOp<
        CUDAContext,
        TensorCoreEngine,
        DontTransposeWeight>
}

register_cuda_operator_with_engine!{
    FCTransposedGradient,
    TENSORCORE,
    FullyConnectedGradientOp<
        CUDAContext,
        TensorCoreEngine,
        DontTransposeWeight>
}
