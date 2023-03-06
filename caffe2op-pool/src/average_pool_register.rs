crate::ix!();

register_cpu_operator!{
    AveragePool,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}

//.FillUsing(AveragePoolDocGenerator(""))

num_inputs!{AveragePool, 1}

num_outputs!{AveragePool, 1}

tensor_inference_function!{AveragePool, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool}

///-------------
register_cpu_operator!{AveragePool1D,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}

//.FillUsing(AveragePoolDocGenerator("1D"))

num_inputs!{AveragePool1D, 1}

num_outputs!{AveragePool1D, 1}

tensor_inference_function!{AveragePool1D, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool1D, "AveragePool"}


///-------------
register_cpu_operator!{
    AveragePool2D,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}
//.FillUsing(AveragePoolDocGenerator("2D"))

num_inputs!{AveragePool2D, 1}

num_outputs!{AveragePool2D, 1}

tensor_inference_function!{AveragePool2D, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool2D, "AveragePool"}

///-------------
register_cpu_operator!{
    AveragePool3D,
    PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>
}

//.FillUsing(AveragePoolDocGenerator("3D"))

num_inputs!{AveragePool3D, 1}

num_outputs!{AveragePool3D, 1}

tensor_inference_function!{
    AveragePool3D, 
    /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */
}

inherit_onnx_schema!{AveragePool3D, "AveragePool"}
