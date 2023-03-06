crate::ix!();

register_cpu_operator!{
    AveragePoolGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePoolGradient, 3}

num_outputs!{AveragePoolGradient, 1}

///-----------------------------
register_cpu_operator!{
    AveragePool1DGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePool1DGradient, 3}

num_outputs!{AveragePool1DGradient, 1}

///-----------------------------
register_cpu_operator!{
    AveragePool2DGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePool2DGradient, 3}

num_outputs!{AveragePool2DGradient, 1}

///-----------------------------
register_cpu_operator!{
    AveragePool3DGradient,
    PoolGradientOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
num_inputs!{AveragePool3DGradient, 3}

num_outputs!{AveragePool3DGradient, 1}

register_gradient!{AveragePool, GetPoolGradient}
register_gradient!{AveragePool1D, GetPoolGradient}
register_gradient!{AveragePool2D, GetPoolGradient}
register_gradient!{AveragePool3D, GetPoolGradient}
