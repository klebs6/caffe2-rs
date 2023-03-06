crate::ix!();

register_cpu_operator!{LC, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC, (2,3)}

num_outputs!{LC, 1}

tensor_inference_function!{LC, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LC1D, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC1D, (2,3)}

num_outputs!{LC1D, 1}

tensor_inference_function!{LC1D, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LC2D, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC2D, (2,3)}

num_outputs!{LC2D, 1}

tensor_inference_function!{LC2D, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LC3D, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC3D, (2,3)}

num_outputs!{LC3D, 1}

tensor_inference_function!{LC3D, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LCGradient, LocallyConnectedGradientOp<f32, CPUContext>}

num_inputs!{LCGradient, (2,3)}

num_outputs!{LCGradient, (1,3)}

///----------------------------
register_cpu_operator!{LC1DGradient, LocallyConnectedGradientOp<f32, CPUContext>}

num_inputs!{LC1DGradient, (2,3)}

num_outputs!{LC1DGradient, (1,3)}

///----------------------------
register_cpu_operator!{LC2DGradient, LocallyConnectedGradientOp<f32, CPUContext> }

num_inputs!{LC2DGradient, (2,3)}

num_outputs!{LC2DGradient, (1,3)}

///----------------------------
register_cpu_operator!{LC3DGradient, LocallyConnectedGradientOp<f32, CPUContext> }

num_inputs!{LC3DGradient, (2,3)}

num_outputs!{LC3DGradient, (1,3)}

register_gradient!{LC,   GetLocallyConnectedGradient}
register_gradient!{LC1D, GetLocallyConnectedGradient}
register_gradient!{LC2D, GetLocallyConnectedGradient}
register_gradient!{LC3D, GetLocallyConnectedGradient}

register_cuda_operator!{LC,             LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LCGradient,     LocallyConnectedGradientOp<f32, CUDAContext>}

register_cuda_operator!{LC1D,           LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LC1DGradient,   LocallyConnectedGradientOp<f32, CUDAContext>}

register_cuda_operator!{LC2D,           LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LC2DGradient,   LocallyConnectedGradientOp<f32, CUDAContext>}

register_cuda_operator!{LC3D,           LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LC3DGradient,   LocallyConnectedGradientOp<f32, CUDAContext>}
