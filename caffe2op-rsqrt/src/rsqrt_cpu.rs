crate::ix!();

register_cpu_operator!{
    Rsqrt,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        RsqrtFunctor<CPUContext>>
}

register_cpu_operator!{
    RsqrtGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        RsqrtGradientFunctor<CPUContext>>
}
