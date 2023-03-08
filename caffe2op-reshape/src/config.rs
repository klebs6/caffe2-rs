crate::ix!();

register_cpu_operator!{
    Reshape, 
    ReshapeOp<f32, CPUContext>
}

register_cuda_operator!{
    Reshape, 
    ReshapeOp<float, CUDAContext>
}

num_inputs!{Reshape, (1,2)}

num_outputs!{Reshape, 2}

inputs!{Reshape, 
    0 => ("data", "*(type: Tensor)* Input tensor."),
    1 => ("new_shape", "*(type: Tensor`<int>`)* [OPTIONAL] Tensor containing new shape.")
}

outputs!{Reshape, 
    0 => ("reshaped", "*(type: Tensor)* Reshaped output tensor."),
    1 => ("old_shape", "*(type: Tensor`<int>`)* Tensor containing old shape of `data`.")
}

args!{Reshape, 
    0 => ("shape", "*(type: Tuple(int))* New shape. Do not set if using `new_shape` input.")
}

allow_inplace!{Reshape, vec![(0, 0)]}

inherit_onnx_schema!{Reshape}
