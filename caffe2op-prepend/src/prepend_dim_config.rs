crate::ix!();

num_inputs!{PrependDim, 1}

num_outputs!{PrependDim, 1}

inputs!{PrependDim, 
    0 => ("data", "An input tensor.")
}

outputs!{PrependDim, 
    0 => ("reshaped", "Reshaped tensor.")
}

args!{PrependDim, 
    0 => ("dim_size", "Size of the dimension to prepend.")
}

allow_inplace!{PrependDim, vec![(0, 0)]}

register_cpu_operator!{
    PrependDim, 
    PrependDimOp<CPUContext>
}

register_gradient!{
    PrependDim,      
    GetPrependDimGradient
}

register_cuda_operator!{
    PrependDim, 
    PrependDimOp<CUDAContext>
}
