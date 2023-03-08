crate::ix!();

num_inputs!{ReciprocalGradient, 2}

num_outputs!{ReciprocalGradient, 1}

allow_inplace!{ReciprocalGradient, vec![(1, 0)]}

register_cpu_operator!{Reciprocal,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        ReciprocalFunctor<CPUContext>>
}

register_cpu_operator!{
    ReciprocalGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        ReciprocalGradientFunctor<CPUContext>>
}
