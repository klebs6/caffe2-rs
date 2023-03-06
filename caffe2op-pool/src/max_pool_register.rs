crate::ix!();

///-------------
register_cpu_operator!{
    MaxPool,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator(""))

num_inputs!{MaxPool, 1}

num_outputs!{MaxPool, 1}

tensor_inference_function!{MaxPool, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool}

///-------------
register_cpu_operator!{
    MaxPool1D,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator("1D"))

num_inputs!{MaxPool1D, 1}

num_outputs!{MaxPool1D, 1}

tensor_inference_function!{MaxPool1D, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool1D, "MaxPool"}

///-------------
register_cpu_operator!{
    MaxPool2D,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator("2D"))

num_inputs!{MaxPool2D, 1}

num_outputs!{MaxPool2D, 1}

tensor_inference_function!{MaxPool2D, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool2D, "MaxPool"}

///-------------
register_cpu_operator!{
    MaxPool3D,
    PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>
}

//.FillUsing(MaxPoolDocGenerator("3D"))

num_inputs!{MaxPool3D, 1}

num_outputs!{MaxPool3D, 1}

tensor_inference_function!{MaxPool3D, /* (ConvPoolOpBase<CPUContext>::TensorInferenceForPool) */}

inherit_onnx_schema!{MaxPool3D, "MaxPool"}
