crate::ix!();

register_cpu_operator!{Conv, ConvOp<float, CPUContext>}
num_inputs!{Conv, (2,3)}
num_outputs!{Conv, 1}
tensor_inference_function!{Conv, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}
cost_inference_function!{Conv, OpSchema::CostInferenceFunctionType(ConvPoolOpBase::<CPUContext>::CostInferenceForConv)}
inherit_onnx_schema!{Conv}

register_cpu_operator!{Conv1D, ConvOp::<f32, CPUContext>}
num_inputs!{Conv1D, (2,3)}
num_outputs!{Conv1D, 1}
inherit_onnx_schema!{Conv1D}
tensor_inference_function!{Conv1D, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}

register_cpu_operator!{Conv2D, ConvOp<f32, CPUContext>}
num_inputs!{Conv2D, (2,3)}
num_outputs!{Conv2D, 1}
inherit_onnx_schema!{Conv2D}
cost_inference_function!{Conv2D, /* OpSchema::CostInferenceFunctionType( ConvPoolOpBase<CPUContext>::CostInferenceForConv) */}
tensor_inference_function!{
    Conv2D, 
    ConvPoolOpBase::<CPUContext>::TensorInferenceForConv
}

register_cpu_operator!{Conv3D, ConvOp<f32, CPUContext>}
num_inputs!{Conv3D, (2,3)}
num_outputs!{Conv3D, 1}
inherit_onnx_schema!{Conv3D}
cost_inference_function!{Conv3D, /* OpSchema::CostInferenceFunctionType( ConvPoolOpBase<CPUContext>::CostInferenceForConv) */}
tensor_inference_function!{Conv3D, ConvPoolOpBase::<CPUContext>::TensorInferenceForConv}
