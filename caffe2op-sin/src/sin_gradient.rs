crate::ix!();

register_cpu_operator!{SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinGradientFunctor<CPUContext>>}

num_inputs!{SinGradient, 2}

num_outputs!{SinGradient, 1}

identical_type_and_shape!{SinGradient}
