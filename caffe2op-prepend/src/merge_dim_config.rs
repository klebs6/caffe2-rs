crate::ix!();

num_inputs!{MergeDim, 1}

num_outputs!{MergeDim, 1}

inputs!{MergeDim, 
    0 => ("data", "An input tensor.")
}

outputs!{MergeDim, 
    0 => ("reshaped", "Reshaped tensor.")
}

allow_inplace!{MergeDim, vec![(0, 0)]}

inherit_onnx_schema!{MergeDim, "Reshape"}

register_cpu_operator!{
    MergeDim,   
    MergeDimOp<CPUContext>
}

register_cuda_operator!{
    MergeDim,
    MergeDimOp<CUDAContext>
}
