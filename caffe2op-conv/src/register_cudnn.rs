crate::ix!();

register_cudnn_operator!{Conv, CudnnConvOp}
register_cudnn_operator!{ConvGradient, CudnnConvGradientOp}

register_cudnn_operator!{Conv1D, CudnnConvOp}
register_cudnn_operator!{Conv1DGradient, CudnnConvGradientOp}

register_cudnn_operator!{Conv2D, CudnnConvOp}
register_cudnn_operator!{Conv2DGradient, CudnnConvGradientOp}

register_cudnn_operator!{Conv3D, CudnnConvOp}
register_cudnn_operator!{Conv3DGradient, CudnnConvGradientOp}
