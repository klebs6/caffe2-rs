crate::ix!();

register_cpu_operator!{
    ExpandDims, 
    ExpandDimsOp<CPUContext>
}

register_cpu_operator!{
    Squeeze,    
    SqueezeOp<CPUContext>
}

register_gradient!{
    Squeeze, 
    GetSqueezeGradient
}

register_cuda_operator!{
    Squeeze, 
    SqueezeOp<CUDAContext>
}

register_gradient!{
    ExpandDims, 
    GetExpandDimsGradient
}

register_cuda_operator!{
    ExpandDims, 
    ExpandDimsOp<CUDAContext>
}
