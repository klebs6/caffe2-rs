crate::ix!();

pub struct CudnnAveragePoolFunctor {
    avg_pool_functor: AveragePoolFunctor<CUDAContext>,
}

register_cudnn_operator!{AveragePool,            CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePoolGradient,    CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{AveragePool1D,          CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePool1DGradient,  CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{AveragePool2D,          CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePool2DGradient,  CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

register_cudnn_operator!{AveragePool3D,          CudnnPoolOp<CudnnAveragePoolFunctor>}
register_cudnn_operator!{AveragePool3DGradient,  CudnnPoolGradientOp<CudnnAveragePoolFunctor>}

impl CudnnAveragePoolFunctor {
    
    pub fn new(op: &OperatorStorage) -> Self {
        todo!();
        /*
            : avg_pool_functor(op)
        */
    }
    
    #[inline] pub fn get_pooling_mode(&self) -> CudnnPoolingMode {
        
        todo!();
        /*
            return avg_pool_functor.count_include_pad
            ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
            : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        */
    }
    
    #[inline] pub fn global_pooling_forward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:         i32,
        c:         i32,
        hxW:       i32,
        x:         *const T,
        y:         *mut T,
        context:   *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return avg_pool_functor.GlobalPoolingForward<T, kOrder>(
              N, C, HxW, X, Y, context);
        */
    }
    
    #[inline] pub fn forward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:          i32,
        c:          i32,
        x_dims:     &Vec<i32>,
        y_dims:     &Vec<i32>,
        kernel:     &Vec<i32>,
        dilation:   &Vec<i32>,
        stride:     &Vec<i32>,
        pads:       &Vec<i32>,
        x:          *const T,
        y:          *mut T,
        context:    *mut CUDAContext) -> bool {
        todo!();
        /*
            return avg_pool_functor.Forward<T, kOrder>(
              N, C, X_dims, Y_dims, kernel, dilation, stride, pads, X, Y, context);
        */
    }
    
    #[inline] pub fn global_pooling_backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:         i32,
        c:         i32,
        hxW:       i32,
        dY:        *const T,
        x:         *const T,
        y:         *const T,
        dX:        *mut T,
        context:   *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return avg_pool_functor.GlobalPoolingBackward<T, kOrder>(
              N, C, HxW, dY, X, Y, dX, context);
        */
    }
    
    #[inline] pub fn backward<T, const kOrder: StorageOrder>(
        &mut self, 
        n:           i32,
        c:           i32,
        x_dims:      &Vec<i32>,
        y_dims:      &Vec<i32>,
        kernel:      &Vec<i32>,
        dilation:    &Vec<i32>,
        stride:      &Vec<i32>,
        pads:        &Vec<i32>,
        dY:          *const T,
        x:           *const T,
        y:           *const T,
        dX:          *mut T,
        context:     *mut CUDAContext) -> bool 
    {
        todo!();
        /*
            return avg_pool_functor.Backward<T, kOrder>(
              N,
              C,
              X_dims,
              Y_dims,
              kernel,
              dilation,
              stride,
              pads,
              dY,
              X,
              Y,
              dX,
              context);
        */
    }
}
