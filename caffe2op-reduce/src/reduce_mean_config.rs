crate::ix!();

register_cpu_operator!{
    ReduceMean,
    ReduceOp<TensorTypes<float>, CPUContext, MeanReducer<CPUContext>>
}

num_inputs!{ReduceMean, 1}

num_outputs!{ReduceMean, 1}

inputs!{ReduceMean, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceMean, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{
    ReduceMean, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
}

tensor_inference_function!{
    ReduceMean, 
    ReduceShapeInference
}

register_cpu_operator!{
    ReduceMeanGradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, MeanReducer<CPUContext>>
}

num_inputs!{ReduceMeanGradient, 3}

num_outputs!{ReduceMeanGradient, 1}
