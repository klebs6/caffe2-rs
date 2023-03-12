crate::ix!();

register_cpu_operator!{
    ConvGradient, 
    ConvGradientOp<f32, CPUContext>
}

num_inputs!{ConvGradient, (2, 3)}
num_outputs!{ConvGradient, (1, 3)}
tensor_inference_function!{ConvGradient, tensor_inference_for_conv_gradient}
cost_inference_function!{ConvGradient, cost_inference_for_conv_gradient}

register_cpu_operator!{ Conv1DGradient, ConvGradientOp<f32, CPUContext> }
num_inputs!{Conv1DGradient, (2, 3)}
num_outputs!{Conv1DGradient, (1, 3)}

register_cpu_operator!{ Conv2DGradient, ConvGradientOp<f32, CPUContext> }
num_inputs!{Conv2DGradient,(2, 3)}
num_outputs!{Conv2DGradient,(1, 3)}

register_cpu_operator!{ Conv3DGradient, ConvGradientOp<f32, CPUContext> }
num_inputs!{Conv3DGradient, (2, 3)}
num_outputs!{Conv3DGradient, (1, 3)}

register_gradient!{Conv,   GetConvGradient}
register_gradient!{Conv1D, GetConvGradient}
register_gradient!{Conv2D, GetConvGradient}
register_gradient!{Conv3D, GetConvGradient}

