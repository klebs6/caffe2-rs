crate::ix!();

register_cpu_operator!{
    Cosh,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CoshFunctor<CPUContext>>
}

register_cpu_operator!{
    CoshGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CoshGradientFunctor<CPUContext>>
}
