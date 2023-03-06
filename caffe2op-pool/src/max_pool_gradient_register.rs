crate::ix!();

register_cpu_operator!{
    MaxPoolGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

num_inputs!{MaxPoolGradient, 3}

num_outputs!{MaxPoolGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPool1DGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

num_inputs!{MaxPool1DGradient, 3}

num_outputs!{MaxPool1DGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPool2DGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

num_inputs!{MaxPool2DGradient, 3}

num_outputs!{MaxPool2DGradient, 1}

///-----------------------------
register_cpu_operator!{
    MaxPool3DGradient,
    PoolGradientOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

num_inputs!{MaxPool3DGradient, 3}

num_outputs!{MaxPool3DGradient, 1}

register_gradient!{MaxPool, GetPoolGradient}
register_gradient!{MaxPool1D, GetPoolGradient}
register_gradient!{MaxPool2D, GetPoolGradient}
register_gradient!{MaxPool3D, GetPoolGradient}
