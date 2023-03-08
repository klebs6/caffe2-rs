crate::ix!();

register_cpu_operator!{
    Negative,
    UnaryElementwiseOp<NumericTypes, CPUContext, NegativeFunctor<CPUContext>>
}

register_cuda_operator!{
    Negative,
    UnaryElementwiseOp<
        NumericTypes,
        CUDAContext,
        NegativeFunctor<CUDAContext>>
}

num_inputs!{Negative, 1}

num_outputs!{Negative, 1}

inputs!{Negative, 
    0 => ("X", "*(type: Tensor`<float>`)* 1D input tensor.")
}

outputs!{Negative, 
    0 => ("Y", "*(type: Tensor`<float>`)* 1D output tensor.")
}

identical_type_and_shape!{Negative}

inherit_onnx_schema!{Negative, "Neg"}

allow_inplace!{Negative, vec![(0, 0)]}
