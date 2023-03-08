crate::ix!();

register_cpu_operator!{
    Relu,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluFunctor<CPUContext>>
}

register_cpu_gradient_operator!{
    ReluGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReluGradientFunctor<CPUContext>>
}

register_cpu_operator!{
    ReluN,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        ReluNFunctor<CPUContext>>
}

register_cpu_operator!{
    ReluNGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        ReluNGradientFunctor<CPUContext>>
}
