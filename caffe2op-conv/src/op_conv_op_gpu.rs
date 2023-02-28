crate::ix!();

register_cuda_operator!{Conv,             ConvOp<f32, CUDAContext>}
register_cuda_operator!{ConvGradient,     ConvGradientOp<f32, CUDAContext>}
register_cuda_operator!{Conv1D,           ConvOp<f32, CUDAContext>}
register_cuda_operator!{Conv1DGradient,   ConvGradientOp<f32, CUDAContext>}
register_cuda_operator!{Conv2D,           ConvOp<f32, CUDAContext>}
register_cuda_operator!{Conv2DGradient,   ConvGradientOp<f32, CUDAContext>}
register_cuda_operator!{Conv3D,           ConvOp<f32, CUDAContext>}
register_cuda_operator!{Conv3DGradient,   ConvGradientOp<f32, CUDAContext>}
